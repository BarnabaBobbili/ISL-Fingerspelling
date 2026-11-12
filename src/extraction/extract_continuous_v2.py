"""
src/extraction/extract_continuous_v2.py
=======================================
Extract RAW TWO-HAND landmarks for every frame of every continuous-fingerspelling
video (the v1 cache kept one hand only, 63-d).

OUTPUT  cache/cont_v2/<uid>.npz
  lm      (T, 2, 21, 3) float32   raw MediaPipe coords, hands ordered by wrist-x
  present (T, 2) bool
  wh      (2,) int32              frame width, height

Run:  python -I src/extraction/extract_continuous_v2.py [--workers 6]
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.static import extract_v2 as ev2  # noqa: E402

VIDEO_DIR = ROOT / "data" / "continious-Fingerspelling-dataset" / "videos"
OUT = ROOT / "cache" / "cont_v2"


def _one(uid: str):
    path = OUT / f"{uid}.npz"
    if path.exists():
        return uid, 0
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{uid}.mp4"))
    lms, prs, wh = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = ev2.detect_hands(frame, ev2._DET)
        lms.append(rec["lm"])
        prs.append(rec["present"])
        wh = rec["wh"]
    cap.release()
    if not lms:
        return uid, -1
    np.savez_compressed(path, lm=np.stack(lms), present=np.stack(prs), wh=wh)
    return uid, len(lms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    uids = sorted(p.stem for p in VIDEO_DIR.glob("*.mp4"))
    t0 = time.time()
    bad = 0
    with Pool(a.workers, initializer=ev2._init_worker) as pool:
        for uid, n in tqdm(pool.imap_unordered(_one, uids, chunksize=4), total=len(uids)):
            bad += n < 0
    print(f"done: {len(uids)} videos, {bad} unreadable, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
