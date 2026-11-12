"""
src/avatar/build_poses.py
=========================
Builds the 3D avatar's pose library (app/static/data/poses.json) from the real data.

For every letter A-Z:
  1. candidates   = dataset samples of that letter with the right number of hands
                    (hand mode: C I L O U V are ONE-handed in this dataset, all others TWO)
                    that the DH-GAT v2 classifier recognises with confidence > 0.85
  2. reference    = the MEDOID candidate (most typical shared-frame pose)
  3. regularise   = rebuild every finger with canonical bone lengths
  4. contacts     = joint pairs that are in contact in the median candidate
  5. touch        = rigid alignment + FABRIK so those contacts really touch (touch.py)
  6. gaps         = contact gap before / after (hand-size units; 0 = skin on skin)

Output coordinates: x right, y UP, z toward the viewer (Three.js), unit = 1 hand size
(wrist -> middle-finger MCP, about 9.5 cm).

  python -m src.avatar.build_poses              # uses cache/$ISL_V2_DIR (default static_v2)
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.avatar import touch  # noqa: E402
from src.models.dh_gat_v2 import featurize  # noqa: E402
from src.pipeline.classifier import StaticClassifier  # noqa: E402
from src.static import benchmark as bm  # noqa: E402

ONE_HANDED = set("CILOUV")
OUT = ROOT / "app" / "static" / "data" / "poses.json"


def px_of(d):
    wh = d["wh"].astype(np.float32)
    return (d["lm"] * np.stack([wh[:, 0], wh[:, 1], wh[:, 0]], 1)[:, None, None, :]).astype(np.float32)


def shared_frame(px, present):
    """g (shared frame) and per-hand size ratio, both from featurize()."""
    node, _ = featurize(torch.from_numpy(px), torch.from_numpy(present))
    g = node[..., :3].numpy()                                            # (N,2,21,3)
    pw = present.astype(np.float32)
    hs = np.linalg.norm(px[:, :, 9] - px[:, :, 0], axis=-1)              # (N,2)
    mean_hs = (hs * pw).sum(1) / np.maximum(pw.sum(1), 1)
    return g, hs / np.maximum(mean_hs[:, None], 1e-6)


def to_three(P):
    """image coords (y down, z toward camera = negative) -> three.js (y up, z toward viewer)."""
    Q = P.copy()
    Q[..., 1] *= -1
    Q[..., 2] *= -1
    return Q


def main():
    clf = StaticClassifier()
    data = [bm.load_source("static_isl")]
    real_path = bm.V2 / "real.npz"
    if real_path.exists():
        data.append(bm.load_source("real"))
    d = bm.concat(data)
    px, present = px_of(d), d["present"]
    y = d["label"].astype(int)
    probs = clf.predict_batch(px, present)
    conf = probs[np.arange(len(y)), y]
    g, hsr = shared_frame(px, present)
    print(f"samples {len(y)}  (two-hand {int(present.all(1).sum())})")

    # canonical bone lengths from per-hand local coords (own hand-size units)
    local = []
    for h in range(2):
        m = present[:, h]
        wr = px[m, h, 0:1]
        hs = np.linalg.norm(px[m, h, 9] - px[m, h, 0], axis=-1)[:, None, None]
        local.append((px[m, h] - wr) / hs)
    lengths = touch.canonical_bone_lengths(np.concatenate(local)[::4])

    poses, rows = {}, []
    for c in range(26):
        L = chr(65 + c)
        two = L not in ONE_HANDED
        sel = (y == c) & (conf > 0.85) & (present.all(1) if two else (present.sum(1) == 1))
        idx = np.where(sel)[0]
        if len(idx) < 3:
            sel = (y == c) & (present.all(1) if two else (present.sum(1) >= 1))
            idx = np.where(sel)[0][np.argsort(-conf[sel])[:50]]
        if len(idx) > 500:
            idx = idx[np.random.default_rng(0).choice(len(idx), 500, replace=False)]
        hands = [0, 1] if two else [0]
        feat = g[idx][:, hands].reshape(len(idx), -1)
        dist = np.linalg.norm(feat[:, None] - feat[None], axis=-1).sum(1)
        ref = idx[int(dist.argmin())]

        H = [touch.regularise_bones(g[ref, h], lengths * hsr[ref, h]) for h in hands]
        contacts, gap_raw, gap_new = [], [], []
        if two:
            A, B = H
            # contact pairs: smallest median skin gap over the candidate set
            ga, gb = g[idx, 0], g[idx, 1]
            # in-plane distance only: depth between the two hands is not observable
            dd = np.linalg.norm(ga[:, :, None, :2] - gb[:, None, :, :2], axis=-1)  # (N,21,21)
            med = np.median(dd, axis=0) - (touch.RADIUS[:, None] + touch.RADIUS[None, :])
            order = np.dstack(np.unravel_index(np.argsort(med, axis=None), med.shape))[0]
            used_i, used_j = set(), set()
            for i, j in order:
                if med[i, j] > 0.30 or len(contacts) >= 4:
                    break
                if i in used_i or j in used_j:
                    continue
                contacts.append((int(i), int(j)))
                used_i.add(int(i)); used_j.add(int(j))
            gap_raw = touch.gaps(A, B, contacts)
            A2, B2 = touch.make_touch(A, B, contacts, lengths * hsr[ref, 1])
            gap_new = touch.gaps(A2, B2, contacts)
            raw_pair, touched_pair = [A, B], [A2, B2]
        else:
            raw_pair = touched_pair = [H[0]]

        poses[L] = {
            "hands": 2 if two else 1,
            "raw": [to_three(p).round(4).tolist() for p in raw_pair],
            "touched": [to_three(p).round(4).tolist() for p in touched_pair],
            "contacts": [{"a": i, "b": j, "target": round(float(touch.RADIUS[i] + touch.RADIUS[j]), 4),
                          "gap_raw": round(gr, 4), "gap_touched": round(gn, 4)}
                         for (i, j), gr, gn in zip(contacts, gap_raw, gap_new)],
            "n_candidates": int(len(idx)),
        }
        rows.append([L, 2 if two else 1, len(idx), len(contacts),
                     round(float(np.mean(np.abs(gap_raw))), 4) if gap_raw else "",
                     round(float(np.mean(np.abs(gap_new))), 4) if gap_new else ""])
        print(f"{L}: hands={2 if two else 1} cand={len(idx):4d} contacts={len(contacts)} "
              f"gap raw={rows[-1][4]} -> touched={rows[-1][5]}")

    # rest pose: relaxed one-handed 'C' shape, mirrored for the left hand
    rest_r = np.array(poses["C"]["touched"][0], np.float32)
    rest_r = rest_r - rest_r[0]
    rest_l = rest_r * np.array([-1, 1, 1], np.float32)
    poses["_rest"] = {"hands": 2, "raw": [rest_l.round(4).tolist(), rest_r.round(4).tolist()],
                      "touched": [rest_l.round(4).tolist(), rest_r.round(4).tolist()], "contacts": [],
                      "n_candidates": 0}
    meta = {"unit": "hand size (wrist->middle MCP, ~9.5cm)", "radius": touch.RADIUS.round(4).tolist(),
            "parent": touch.PARENT, "bone_lengths": lengths.round(4).tolist(), "one_handed": sorted(ONE_HANDED)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "letters": poses}, separators=(",", ":")))
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "avatar_contacts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["letter", "hands", "candidates", "contacts", "mean_abs_gap_raw", "mean_abs_gap_touched"])
        w.writerows(rows)
    print(f"saved {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
