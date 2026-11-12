"""
src/evaluation/browser_parity.py
================================
Does the in-browser tracker (MediaPipe JS) give the same recognition result as the server-side
tracker (MediaPipe Python) that the models were trained on?

Input: JSON written by the browser harness (tests/browser_parity.mjs): for every video the
landmarks the browser tracker produced frame by frame.  We decode them with the continuous model
and compare character error rate and landmark agreement with the server-side cache (cache/cont_v2).

  python src/evaluation/browser_parity.py lm_video.json [lm_image.json ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import ctc  # noqa: E402
from src.pipeline.continuous import ContinuousModel  # noqa: E402
from src.pipeline.features import to_px  # noqa: E402

DATA = ROOT / "data" / "continious-Fingerspelling-dataset"


def arrays(frames, wh):
    n = len(frames)
    lm = np.zeros((n, 2, 21, 3), np.float32)
    pr = np.zeros((n, 2), bool)
    for i, hands in enumerate(frames):
        hs = sorted((np.asarray(h, np.float32) for h in hands[:2]), key=lambda a: a[0, 0])
        for j, a in enumerate(hs):
            lm[i, j], pr[i, j] = a, True
    return lm, pr


def decode(model, lm, pr, wh):
    keep = pr.any(1)
    if keep.sum() < 2:
        return ""
    px = np.stack([to_px(l, wh) for l in lm])
    return ctc.greedy_decode(model.log_probs(model.video_features(px[keep], pr[keep])))


def main():
    model = ContinuousModel.load()
    ann = pd.read_csv(DATA / "fingerspelling_annotations.csv").set_index("uid")["text"].to_dict()
    runs = {Path(p).stem: json.loads(Path(p).read_text()) for p in sys.argv[1:]}
    uids = list(next(iter(runs.values())))
    rows, refs, server_h = [], [], []
    hyps = {k: [] for k in runs}
    agree = {k: [] for k in runs}
    for u in uids:
        d = np.load(ROOT / "cache" / "cont_v2" / f"{u}.npz")
        ref = "".join(c for c in str(ann[u]).lower() if c in ctc.LETTERS)
        refs.append(ref)
        server_h.append(decode(model, d["lm"], d["present"], d["wh"]))
        for k, r in runs.items():
            lm, pr = arrays(r[u]["frames"], (r[u]["w"], r[u]["h"]))
            n = min(len(lm), len(d["lm"]))
            hyps[k].append(decode(model, lm, pr, np.array([r[u]["w"], r[u]["h"]])))
            both = pr[:n].all(1) & d["present"][:n].all(1)
            if both.any():                                       # mean landmark difference, in hand sizes
                hs = np.linalg.norm(d["lm"][:n, 0, 9] - d["lm"][:n, 0, 0], axis=-1)[both] + 1e-6
                diff = np.linalg.norm(lm[:n][both] - d["lm"][:n][both], axis=-1).mean(axis=(1, 2)) / hs
                agree[k].append(float(np.median(diff)))
    print(f"videos: {len(uids)}")
    print(f"{'tracker':<34}{'CER':>8}   hand-presence agreement / median landmark error")
    print(f"{'server (Python, training pipeline)':<34}{100*ctc.cer(server_h, refs):>7.2f}%")
    for k in runs:
        pres = []
        for u in uids:
            d = np.load(ROOT / "cache" / "cont_v2" / f"{u}.npz")
            lm, pr = arrays(runs[k][u]["frames"], None)
            n = min(len(pr), len(d["present"]))
            pres.append(float((pr[:n].any(1) == d["present"][:n].any(1)).mean()))
        print(f"{'browser: ' + k:<34}{100*ctc.cer(hyps[k], refs):>7.2f}%   "
              f"{100*np.mean(pres):.1f}% / {np.mean(agree[k]) if agree[k] else float('nan'):.3f} hand-sizes")
    for u, r, s in zip(uids, refs, server_h):
        print(f"  {u}: truth={r}  server={s}  " + "  ".join(f"{k}={hyps[k][uids.index(u)]}" for k in runs))


if __name__ == "__main__":
    main()
