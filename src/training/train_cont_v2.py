"""
src/training/train_cont_v2.py
=============================
Train the continuous (CTC) fingerspelling models on cache/cont_feat.pt.

  python src/training/train_cont_v2.py --kind conformer --parts all --tag conformer_all
  python src/training/train_cont_v2.py --kind bilstm    --parts raw --tag bilstm_raw   # no trunk features (ablation)

--parts all   trunk + probs + raw geometry + motion (351-d)
--parts raw   raw geometry + motion only (133-d): shows what the pretrained trunk adds
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.models.cont_v2 import SeqCTC  # noqa: E402
from src.pipeline import ctc  # noqa: E402
from src.pipeline.features import SLICES, FEAT_DIM  # noqa: E402

torch.set_num_threads(int(os.environ.get("ISL_THREADS", "8")))
DATA = ROOT / "data" / "continious-Fingerspelling-dataset"


def columns(parts: str):
    if parts == "all":
        return list(range(FEAT_DIM))
    cols = []
    for k in ("raw", "motion"):
        s = SLICES[k]
        cols += list(range(s.start, s.stop))
    return cols


def load_data(cols):
    feats = torch.load(ROOT / "cache" / "cont_feat.pt")
    ann = pd.read_csv(DATA / "fingerspelling_annotations.csv")
    split = pd.read_csv(ROOT / "cache" / "final_split.csv").set_index("uid")["split"].to_dict()
    sets = {"train": [], "val": [], "test": []}
    for uid, text in zip(ann["uid"], ann["text"]):
        t = "".join(c for c in str(text).lower() if c in ctc.LETTERS)
        if uid not in feats or uid not in split or not t or len(feats[uid]) < len(t):
            continue
        sets[split[uid]].append((uid, feats[uid][:, cols].float(), t))
    return sets


def augment(x: torch.Tensor, trunk_cols: torch.Tensor):
    """x (T,F) standardised features of one sequence."""
    T = x.shape[0]
    if np.random.rand() < 0.7:                                          # time stretch
        r = np.random.uniform(0.8, 1.25)
        T2 = max(8, int(T * r))
        x = F.interpolate(x.T[None], size=T2, mode="linear", align_corners=False)[0].T
        T = T2
    for _ in range(np.random.randint(0, 3)):                            # time masks
        w = np.random.randint(2, 9)
        if T > w + 2:
            s = np.random.randint(0, T - w)
            x[s:s + w] = 0
    x = x * (torch.rand_like(x) > 0.05).float()                         # feature dropout
    if trunk_cols is not None and np.random.rand() < 0.25:              # force use of raw geometry
        x[:, trunk_cols] = 0
    return x + 0.08 * torch.randn_like(x)


def batchify(items, mean, std, train, trunk_cols):
    xs, ys = [], []
    for _, x, t in items:
        x = (x - mean) / std
        if train:
            x = augment(x.clone(), trunk_cols)
        xs.append(x)
        ys.append(torch.tensor([ctc.LETTERS.index(c) for c in t]))
    lens = torch.tensor([len(x) for x in xs])
    X = torch.zeros(len(xs), int(lens.max()), xs[0].shape[1])
    for i, x in enumerate(xs):
        X[i, :len(x)] = x
    return X, lens, torch.cat(ys), torch.tensor([len(y) for y in ys])


@torch.no_grad()
def evaluate(model, items, mean, std, bs=32):
    model.eval()
    hyps, refs = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        X, lens, _, _ = batchify(chunk, mean, std, False, None)
        lp, _ = model(X, lens)
        for b, (_, _, t) in enumerate(chunk):
            hyps.append(ctc.greedy_decode(lp[:int(lens[b]), b].numpy()))
            refs.append(t)
    return ctc.cer(hyps, refs), hyps, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="conformer", choices=["conformer", "bilstm"])
    ap.add_argument("--parts", default="all", choices=["all", "raw"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--kernel", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tag = a.tag or f"{a.kind}_{a.parts}"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    lr = a.lr or (7e-4 if a.kind == "conformer" else 1.5e-3)

    cols = columns(a.parts)
    sets = load_data(cols)
    print({k: len(v) for k, v in sets.items()}, "feature dims", len(cols))
    cat = torch.cat([x for _, x, _ in sets["train"]])
    mean, std = cat.mean(0), cat.std(0).clamp(min=1e-3)
    trunk_cols = torch.tensor([i for i, c in enumerate(cols) if c < SLICES["probs"].stop]) if a.parts == "all" else None

    model = SeqCTC(a.kind, in_dim=len(cols), kernel=a.kernel)
    print(tag, "params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = (len(sets["train"]) + a.bs - 1) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.epochs * steps, pct_start=0.15)
    ctc_loss = torch.nn.CTCLoss(blank=ctc.BLANK, zero_infinity=True)
    best, bad, t0 = 9.0, 0, time.time()
    ckpt = ROOT / "checkpoints" / f"cont_v2_{tag}.pt"
    for ep in range(1, a.epochs + 1):
        model.train()
        order = np.random.permutation(len(sets["train"]))
        tot = 0.0
        for i in range(0, len(order), a.bs):
            items = [sets["train"][j] for j in order[i:i + a.bs]]
            X, lens, Y, ylen = batchify(items, mean, std, True, trunk_cols)
            lens = torch.clamp(lens, max=X.shape[1])
            lp, inter = model(X, lens)
            loss = ctc_loss(lp, Y, lens, ylen)
            if inter is not None:
                loss = loss + 0.3 * ctc_loss(inter, Y, lens, ylen)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
        vcer, _, _ = evaluate(model, sets["val"], mean, std)
        flag = ""
        if vcer < best:
            best, bad, flag = vcer, 0, " *"
            torch.save({"state": model.state_dict(), "cfg": model.cfg, "mean": mean, "std": std, "cols": cols,
                        "parts": a.parts, "val_cer": vcer, "epoch": ep}, ckpt)
        else:
            bad += 1
        print(f"ep {ep:02d} loss {tot/steps:.3f} | val CER {100*vcer:.2f}%{flag} | {time.time()-t0:.0f}s", flush=True)
        if bad >= a.patience:
            print("early stop")
            break
    ck = torch.load(ckpt, weights_only=False)
    model.load_state_dict(ck["state"])
    tcer, hyps, refs = evaluate(model, sets["test"], mean, std)
    ex = [f"{r} -> {h}" for r, h in list(zip(refs, hyps))[:8]]
    print(f"\n[{tag}] best val CER {100*ck['val_cer']:.2f}% (epoch {ck['epoch']})   TEST CER {100*tcer:.2f}%")
    print("examples (truth -> prediction):", *ex, sep="\n  ")
    ck["test_cer"] = tcer
    torch.save(ck, ckpt)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"cont_v2_{tag}.json").write_text(json.dumps(
        {"tag": tag, "kind": a.kind, "parts": a.parts, "val_cer": ck["val_cer"], "test_cer": tcer, "epoch": ck["epoch"],
         "params": sum(p.numel() for p in model.parameters()), "examples": ex}, indent=1))


if __name__ == "__main__":
    main()
