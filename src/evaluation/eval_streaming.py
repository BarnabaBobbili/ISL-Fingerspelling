"""
src/evaluation/eval_streaming.py
================================
How well does continuous recognition work LIVE? Replays the held-out test videos through the
streaming recogniser under different conditions and compares with whole-clip decoding.

Conditions
  speed x1 / x1.5 / x2     the signer spells faster (landmark track time-compressed; camera 30 fps)
  camera 15 / 10 fps       a slow camera at normal signing speed (frames dropped, true timestamps)

For each condition:
  whole-clip   decode all frames at once, no resampling  (= the previous live behaviour at a pause)
  streaming    StreamingSession: 30 Hz resampling + stable-prefix commit while frames arrive
and the per-letter latency (peak -> committed) of the streaming decoder.

  python src/evaluation/eval_streaming.py [--ckpt checkpoints/cont_best.pt] [--tag base] [--limit N]
Writes results/streaming_<tag>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import ctc  # noqa: E402
from src.pipeline import features as F  # noqa: E402
from src.pipeline.continuous import ContinuousModel  # noqa: E402
from src.pipeline.streaming import StreamingSession  # noqa: E402

DATA = ROOT / "data" / "continious-Fingerspelling-dataset"
FPS = 29.97


def time_compress(lm: np.ndarray, present: np.ndarray, k: float):
    """signer k times faster, camera unchanged: resample the track at positions i*k."""
    T = len(lm)
    pos = np.arange(0, T - 1, k)
    i0 = np.floor(pos).astype(int)
    a = (pos - i0)[:, None, None, None]
    i1 = np.minimum(i0 + 1, T - 1)
    same = (present[i0] == present[i1]).all(1)
    out = np.where(same[:, None, None, None], lm[i0] * (1 - a) + lm[i1] * a, lm[np.round(pos).astype(int)])
    pr = np.where(same[:, None], present[i0], present[np.round(pos).astype(int)])
    return out.astype(np.float32), pr


def run_stream(model, lm, present, wh, times, commit_lag, resample=True):
    s = StreamingSession(model, commit_lag=commit_lag, resample=resample)
    for i in range(len(lm)):
        s.step({"lm": lm[i], "present": present[i], "wh": wh}, float(times[i]), i)
    text = s.flush().replace(" ", "").lower()
    return text, s.latencies


def whole_clip(model, lm, present, wh):
    px = F.to_px(lm, wh)
    keep = present.any(1)
    if keep.sum() < 2:
        return ""
    lp = model.log_probs(model.video_features(px[keep], present[keep]))
    return ctc.greedy_decode(lp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--commit-lag", type=int, default=12)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    model = ContinuousModel.load(a.ckpt)
    split = pd.read_csv(ROOT / "cache" / "final_split.csv")
    ann = pd.read_csv(DATA / "fingerspelling_annotations.csv").set_index("uid")["text"].to_dict()
    uids = [u for u in split[split.split == a.split].uid if (ROOT / "cache" / "cont_v2" / f"{u}.npz").exists()]
    if a.limit:
        uids = uids[:a.limit]
    conds = [("speed x1", dict(k=1.0, cam=1)), ("speed x1.5", dict(k=1.5, cam=1)), ("speed x2", dict(k=2.0, cam=1)),
             ("camera 15 fps", dict(k=1.0, cam=2)), ("camera 10 fps", dict(k=1.0, cam=3))]
    res = {name: dict(whole=[], stream=[], refs=[], lat=[]) for name, _ in conds}
    t0 = time.time()
    for n, uid in enumerate(uids):
        d = np.load(ROOT / "cache" / "cont_v2" / f"{uid}.npz")
        ref = "".join(c for c in str(ann[uid]).lower() if c in ctc.LETTERS)
        if not ref:
            continue
        for name, c in conds:
            lm, pr = (d["lm"], d["present"]) if c["k"] == 1.0 else time_compress(d["lm"], d["present"], c["k"])
            times = np.arange(len(lm)) / FPS
            lm, pr, times = lm[::c["cam"]], pr[::c["cam"]], times[::c["cam"]]
            r = res[name]
            r["refs"].append(ref)
            r["whole"].append(whole_clip(model, lm, pr, d["wh"]))
            txt, lat = run_stream(model, lm, pr, d["wh"], times, a.commit_lag)
            r["stream"].append(txt)
            r["lat"] += lat
        if (n + 1) % 25 == 0:
            print(f"{n+1}/{len(uids)} videos  {time.time()-t0:.0f}s", flush=True)
    out = {"model": model.name, "ckpt": str(a.ckpt or "cont_best.pt"), "videos": len(res["speed x1"]["refs"]),
           "commit_lag_frames": a.commit_lag, "conditions": {}}
    print(f"\n{'condition':<16}{'whole-clip CER':>16}{'streaming CER':>16}{'latency median':>17}{'p90':>8}")
    for name, _ in conds:
        r = res[name]
        w, s = ctc.cer(r["whole"], r["refs"]), ctc.cer(r["stream"], r["refs"])
        lat = np.array(r["lat"]) / 30.0 if r["lat"] else np.array([0.0])
        out["conditions"][name] = {"whole_cer": w, "stream_cer": s, "latency_median_s": float(np.median(lat)),
                                   "latency_p90_s": float(np.percentile(lat, 90))}
        print(f"{name:<16}{100*w:>15.2f}%{100*s:>15.2f}%{np.median(lat):>15.2f} s{np.percentile(lat, 90):>7.2f}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"streaming_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved results/streaming_{a.tag}.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
