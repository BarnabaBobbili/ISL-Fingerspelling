"""
src/static/extract_v2.py
========================
Parallel RAW two-hand landmark extraction for the static ISL image datasets.

WHY A V2 CACHE
--------------
The original cache (cache/static/*.npz) stores each hand ALREADY wrist-centred
and scale-normalised, so the position of one hand relative to the other is lost.
Many ISL letters are defined by exactly that (e.g. "right index touches left
palm"). v2 keeps the raw MediaPipe image coordinates of every detected hand so
inter-hand geometry can be computed later.

OUTPUT (one file per source in cache/static_v2/<source>.npz)
  lm      (N, 2, 21, 3) float32  raw MediaPipe coords (x,y in [0,1], z relative)
                                 hands ordered by wrist-x (image-left first)
  present (N, 2) bool
  hand    (N, 2) int8            MediaPipe handedness label: 0=Right, 1=Left, -1=none
  score   (N, 2) float32         handedness score
  wh      (N, 2) int32           image (width, height) -> needed to undo aspect ratio
  label   (N,)   int8            0..25 = A..Z
  name    (N,)   str             path relative to the dataset root (provenance)

Run:
  python src/static/extract_v2.py                 # all sources
  python src/static/extract_v2.py --sample 200    # quick test
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.static.extract_static_landmarks import _collect_images, TASK_MODEL_PATH  # noqa: E402

OUT_DIR = ROOT / "cache" / "static_v2"
_DET = None
_RECOVER = False


def _init_worker(recover: bool = False):
    global _DET, _RECOVER
    _RECOVER = recover
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    _DET = mp_vision.HandLandmarker.create_from_options(opts)


def _second_hand(bgr, detector, lm_norm):
    """
    SECOND-HAND RECOVERY. When MediaPipe reports one hand but a sign involves two
    overlapping hands, blank out the found hand (convex hull + margin) and detect again.
    A new hand is accepted only if its wrist lies outside the blanked area. Measured on
    Static_ISL: J 0%->52%, S 30%->76%, B 80%->100% two-hand detection, while one-handed
    letters stay at 0-2% false recoveries (O: 18%).
    """
    h, w = bgr.shape[:2]
    pts = lm_norm[:, :2] * [w, h]
    pad = 0.12 * float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    hull = cv2.convexHull(pts.astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    k = int(pad) * 2 + 1
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    m = bgr.copy()
    m[mask > 0] = np.median(bgr.reshape(-1, 3), 0).astype(np.uint8)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(m, cv2.COLOR_BGR2RGB)))
    for lms, hd in zip(res.hand_landmarks, res.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
        wx, wy = arr[0, 0] * w, arr[0, 1] * h
        if cv2.pointPolygonTest(hull, (float(wx), float(wy)), True) < -pad:
            return arr, 0 if hd[0].display_name == "Right" else 1, float(hd[0].score)
    return None


def detect_hands(bgr: np.ndarray, detector, recover: bool = None) -> dict:
    """Run MediaPipe on one BGR image; return the v2 per-image record."""
    if recover is None:
        recover = _RECOVER
    h, w = bgr.shape[:2]
    rec = {
        "lm": np.zeros((2, 21, 3), np.float32),
        "present": np.zeros(2, bool),
        "hand": np.full(2, -1, np.int8),
        "score": np.zeros(2, np.float32),
        "wh": np.array([w, h], np.int32),
    }
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    dets = []
    for lms, hd in zip(res.hand_landmarks, res.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
        dets.append((arr, 0 if hd[0].display_name == "Right" else 1, float(hd[0].score)))
    if recover and len(dets) == 1:
        extra = _second_hand(bgr, detector, dets[0][0])
        if extra is not None:
            dets.append(extra)
    dets.sort(key=lambda d: d[0][0, 0])            # wrist x ascending
    for i, (arr, hand, sc) in enumerate(dets[:2]):
        rec["lm"][i] = arr
        rec["present"][i] = True
        rec["hand"][i] = hand
        rec["score"][i] = sc
    return rec


def _work(item):
    path, label, _src = item
    img = cv2.imread(str(path))
    if img is None:
        return None
    rec = detect_hands(img, _DET)
    rec["label"] = np.int8(ord(label) - ord("A"))
    return rec


def run(sample: int = None, sources=None, recover: bool = False, out_dir: Path = None):
    global OUT_DIR
    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if not TASK_MODEL_PATH.exists():
        sys.exit(f"hand_landmarker.task not found at {TASK_MODEL_PATH}")
    items = _collect_images()
    if sources:
        items = [it for it in items if it[2] in sources]
    if sample:
        import random
        random.seed(0)
        items = random.sample(items, min(sample, len(items)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_root = ROOT / "data"

    by_src = {}
    for it in items:
        by_src.setdefault(it[2], []).append(it)

    t0 = time.time()
    with Pool(max(1, min(8, (__import__("os").cpu_count() or 4) - 2)), initializer=_init_worker, initargs=(recover,)) as pool:
        for src, its in by_src.items():
            recs = []
            kept = []
            for it, rec in zip(its, tqdm(pool.imap(_work, its, chunksize=32),
                                         total=len(its), desc=src, unit="img")):
                if rec is not None:
                    recs.append(rec)
                    kept.append(it)
            out = OUT_DIR / (f"{src}.npz" if not sample else f"{src}_sample.npz")
            np.savez_compressed(
                out,
                lm=np.stack([r["lm"] for r in recs]),
                present=np.stack([r["present"] for r in recs]),
                hand=np.stack([r["hand"] for r in recs]),
                score=np.stack([r["score"] for r in recs]),
                wh=np.stack([r["wh"] for r in recs]),
                label=np.array([r["label"] for r in recs], np.int8),
                name=np.array([str(it[0].relative_to(data_root)) for it in kept]),
            )
            n2 = sum(int(r["present"].all()) for r in recs)
            n1 = sum(int(r["present"].sum() == 1) for r in recs)
            n0 = sum(int(r["present"].sum() == 0) for r in recs)
            print(f"[{src}] {len(recs)} imgs  two-hand={n2}  one-hand={n1}  none={n0}  -> {out.name}")
    print(f"done in {time.time() - t0:.0f}s")


def _work_path(path):
    img = cv2.imread(str(path))
    return detect_hands(img, _DET)


def run_real(recover: bool, out_dir: Path):
    """Re-detect the CTC-aligned real frames (jpgs from align_real_frames.py), keep metadata."""
    src = dict(np.load(ROOT / "cache" / "static_v2" / "real.npz", allow_pickle=False))
    paths = [ROOT / "data" / n for n in src["name"]]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(max(1, min(8, (__import__("os").cpu_count() or 4) - 2)), initializer=_init_worker, initargs=(recover,)) as pool:
        recs = list(tqdm(pool.imap(_work_path, paths, chunksize=16), total=len(paths), desc="real"))
    for k, key in (("lm", "lm"), ("present", "present"), ("hand", "hand"), ("score", "score"), ("wh", "wh")):
        src[k] = np.stack([r[key] for r in recs])
    np.savez_compressed(out_dir / "real.npz", **src)
    print("real two-hand:", int(src["present"].all(1).sum()), "/", len(paths))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--recover", action="store_true", help="second-hand recovery")
    ap.add_argument("--out", default=None, help="output dir (default cache/static_v2)")
    ap.add_argument("--real", action="store_true",
                    help="re-detect the aligned real frames listed in cache/static_v2/real.npz")
    a = ap.parse_args()
    if a.real:
        run_real(a.recover, Path(a.out) if a.out else OUT_DIR)
    else:
        run(a.sample, a.sources, a.recover, a.out)
