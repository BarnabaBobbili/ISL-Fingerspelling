"""
src/static/benchmark.py
=======================
Honest evaluation protocol for static-sign models.

The original score (98.45%) comes from a RANDOM split of ~45k images that are
largely near-duplicate video frames, so train and test contain almost the same
pictures. This module defines three tests so every model is reported on all of them:

  in_domain   the original random test split (kept for comparison with old numbers)
  cross_src   train on {static_isl, kaggle}  ->  test on isl_data   (new camera,
              signer, background and lighting -- nothing shared with training)
  real        letters cut from the continuous news videos at CTC-aligned times
              (small hands, full-body shot, real co-articulation). Split by VIDEO.

Data comes from cache/static_v2 (extract_v2.py, align_real_frames.py).

Usage:
  python src/static/benchmark.py                       # evaluate the legacy DH-GAT
  python src/static/benchmark.py --ckpt checkpoints/x.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.training.config import STATIC_LETTERS, StaticConfig  # noqa: E402
from src.preprocessing.normalizer import normalize_hand_static  # noqa: E402
from src.preprocessing.feature_engineer import compute_node_features, compute_global_distances  # noqa: E402

import os
V2 = ROOT / "cache" / os.environ.get("ISL_V2_DIR", "static_v2")
SOURCES = ("static_isl", "kaggle", "isl_data")


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_source(name: str) -> dict:
    """Load one v2 npz as a plain dict of arrays (adds 'source' column)."""
    d = dict(np.load(V2 / f"{name}.npz", allow_pickle=False))
    d["source"] = np.array([name] * len(d["label"]))
    return d


def concat(parts: list) -> dict:
    keys = set.intersection(*[set(p) for p in parts])
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def subset(d: dict, mask) -> dict:
    return {k: v[mask] for k, v in d.items()}


def load_all() -> dict:
    """{'static_isl':…, 'kaggle':…, 'isl_data':…, 'real':…(if built)}"""
    out = {s: load_source(s) for s in SOURCES}
    if (V2 / "real.npz").exists():
        out["real"] = load_source("real")
    return out


def legacy_in_domain_test_mask(d: dict, src: str) -> np.ndarray:
    """Boolean mask over source `src` selecting the files in the ORIGINAL test split."""
    return legacy_split_mask(d, src, "test")


def legacy_split_mask(d: dict, src: str, split: str) -> np.ndarray:
    """Boolean mask over source `src` selecting files of the ORIGINAL train/val/test split."""
    keep = set()
    with open(ROOT / "cache" / "static" / "split.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            stem = Path(row["path"].replace("\\", "/")).stem          # e.g. static_isl_233
            if stem.startswith(src + "_"):
                keep.add((row["label"], stem[len(src) + 1:]))
    names = [Path(n).stem for n in d["name"]]
    labels = [STATIC_LETTERS[int(i)] for i in d["label"]]
    return np.array([(l, n) in keep for l, n in zip(labels, names)])


# ─────────────────────────────────────────────────────────────────────────────
# Legacy DH-GAT adapter (per-hand normalised, slotted by MediaPipe handedness)
# ─────────────────────────────────────────────────────────────────────────────
def legacy_inputs(d: dict) -> dict:
    N = len(d["label"])
    lm = {"r": np.zeros((N, 21, 3), np.float32), "l": np.zeros((N, 21, 3), np.float32)}
    pres = {"r": np.zeros(N, bool), "l": np.zeros(N, bool)}
    for i in range(N):
        for j in range(2):
            if d["present"][i, j]:
                k = "r" if d["hand"][i, j] == 0 else "l"
                lm[k][i] = normalize_hand_static(d["lm"][i, j])
                pres[k][i] = True
    node = {k: np.stack([compute_node_features(x) for x in lm[k]]) for k in lm}
    geo = {k: np.stack([compute_global_distances(x) for x in lm[k]]) for k in lm}
    t = torch.from_numpy
    return dict(right_node=t(node["r"]), left_node=t(node["l"]),
                right_present=t(pres["r"]), left_present=t(pres["l"]),
                right_geo=t(geo["r"]), left_geo=t(geo["l"]))


@torch.no_grad()
def predict_legacy(model, d: dict, bs: int = 512):
    x = legacy_inputs(d)
    probs = []
    for i in range(0, len(d["label"]), bs):
        b = {k: v[i:i + bs] for k, v in x.items()}
        probs.append(torch.softmax(model(b["right_node"], b["left_node"], b["right_present"],
                                         b["left_present"], b["right_geo"], b["left_geo"]), -1))
    return torch.cat(probs).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def summarize(probs: np.ndarray, y: np.ndarray, present: np.ndarray = None) -> dict:
    pred = probs.argmax(1)
    conf = probs.max(1)
    acc = float((pred == y).mean())
    per = {STATIC_LETTERS[c]: float((pred[y == c] == c).mean()) for c in range(26) if (y == c).any()}
    # Expected calibration error (10 bins)
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    conf_pairs = {}
    for t, p in zip(y[pred != y], pred[pred != y]):
        k = f"{STATIC_LETTERS[t]}->{STATIC_LETTERS[p]}"
        conf_pairs[k] = conf_pairs.get(k, 0) + 1
    top = dict(sorted(conf_pairs.items(), key=lambda kv: -kv[1])[:8])
    out = dict(n=int(len(y)), acc=acc, macro_acc=float(np.mean(list(per.values()))),
               ece=float(ece), mean_conf=float(conf.mean()), top_confusions=top, per_class=per)
    if present is not None:
        out["acc_by_hands_detected"] = {int(k): float((pred[present == k] == y[present == k]).mean())
                                        for k in (0, 1, 2) if (present == k).any()}
    return out


def print_table(name: str, res: dict):
    print(f"\n=== {name} ===")
    print(f"{'protocol':<26}{'n':>7}{'acc':>9}{'macro':>9}{'ECE':>8}{'conf':>8}")
    for k, r in res.items():
        print(f"{k:<26}{r['n']:>7}{100*r['acc']:>8.2f}%{100*r['macro_acc']:>8.2f}%{r['ece']:>8.3f}{r['mean_conf']:>8.3f}")
        print(f"   top confusions: {r['top_confusions']}")
        if "acc_by_hands_detected" in r:
            print(f"   acc by #hands detected: { {k: round(100*v,1) for k, v in r['acc_by_hands_detected'].items()} }")


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_legacy(ckpt: Path):
    from src.models.dh_gat import build_dhgat
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_dhgat(ck.get("config", StaticConfig()))
    model.load_state_dict(ck["model_state"])
    model.eval()
    data = load_all()
    res = {}
    # in-domain test, per source and total
    parts = []
    for s in SOURCES:
        m = legacy_in_domain_test_mask(data[s], s)
        parts.append(subset(data[s], m))
    ind = concat(parts)
    res["in_domain (orig. split)"] = summarize(predict_legacy(model, ind), ind["label"].astype(int),
                                               ind["present"].sum(1))
    for s, p in zip(SOURCES, parts):
        if len(p["label"]):
            res[f"  in_domain / {s}"] = summarize(predict_legacy(model, p), p["label"].astype(int))
    # real frames: the checkpoint never saw the news videos
    if "real" in data:
        r = data["real"]
        res["real (all videos)"] = summarize(predict_legacy(model, r), r["label"].astype(int), r["present"].sum(1))
        for sp in ("train", "val", "test"):
            m = r["split"] == sp
            if m.any():
                rs = subset(r, m)
                res[f"  real / {sp} videos"] = summarize(predict_legacy(model, rs), rs["label"].astype(int))
    print_table(f"legacy DH-GAT  ({ckpt.name})", res)
    out = ROOT / "results" / f"benchmark_{ckpt.stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\nsaved {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "dh_gat_static_best.pt"))
    a = ap.parse_args()
    evaluate_legacy(Path(a.ckpt))
