"""
src/extraction/landmark_extractor.py
======================================
Stage 2: MediaPipe Hands landmark extraction pipeline.

WHAT THIS DOES
--------------
Processes every video in the dataset frame-by-frame:
  1. Opens video with OpenCV
  2. Runs MediaPipe Hands on each frame
  3. Extracts 21 landmarks × (x, y, z) = 63 features per frame
  4. Applies missing-detection strategy (forward-fill then zero-fill)
  5. Saves the result as a .npy file in cache/

DESIGN DECISIONS
-----------------
DD-003: Missing landmark strategy
  - Forward-fill last valid detection for ≤ MAX_FORWARD_FILL_FRAMES
  - Zero-fill beyond that; mark frame as invalid in boolean mask
  - Drop segments with > DROP_SEGMENT_THRESHOLD of frames zero-filled

DD-004: Cache format = .npy, shape (T, 63)
  - One file per segment: cache/{uid}.npy
  - Companion mask: cache/{uid}_mask.npy  (shape (T,), True=valid detection)

WHY CACHE?
  MediaPipe takes ~0.05-0.1s per frame. At 30 FPS, each segment needs
  ~3–10 seconds of processing. Running this on all 1308 segments would take
  ~1–2 hours. We cache results so training can be re-run in seconds.

USAGE
-----
  # Full extraction (all segments):
  python src/extraction/landmark_extractor.py

  # Sample only (5 random segments, for quick verification):
  python src/extraction/landmark_extractor.py --sample 5

  # Specific segment:
  python src/extraction/landmark_extractor.py --uid KxqgVJSEiW0_seg000

  # Force re-extract even if cache exists:
  python src/extraction/landmark_extractor.py --force
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    VIDEOS_DIR, CACHE_DIR, EXTRACTION_STATS_CSV,
    NUM_LANDMARKS, LANDMARK_DIM, FEATURE_DIM,
    MAX_FORWARD_FILL_FRAMES, DROP_SEGMENT_THRESHOLD,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE, MP_MAX_NUM_HANDS,
    ROOT, RANDOM_SEED
)

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Tasks API setup (compatible with mediapipe 0.10.x / 1.0.x)
# ─────────────────────────────────────────────────────────────────────────────
# DESIGN NOTE: MediaPipe dropped mp.solutions in Python 3.11+ / mediapipe 0.10+.
# We use the new Tasks API with a downloaded .task model file.
# The model file (hand_landmarker.task) lives at the project root.
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

MP_MODEL_PATH = ROOT / "hand_landmarker.task"
if not MP_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {MP_MODEL_PATH}\n"
        "Download it with:\n"
        "  python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')\""
    )

_BaseOptions       = _mp_python.BaseOptions
_HandLandmarker    = _mp_vision.HandLandmarker
_HandLandmarkerOpt = _mp_vision.HandLandmarkerOptions
_RunningMode       = _mp_vision.RunningMode


def extract_landmarks_from_video(video_path: Path) -> dict:
    """
    Runs MediaPipe HandLandmarker (Tasks API) on a single video.

    WHY TASKS API (not mp.solutions.hands):
      mp.solutions was removed from mediapipe for Python 3.11+ (mediapipe 0.10+).
      The Tasks API is the current supported interface for Python 3.13.
      Functionally equivalent: both give 21 (x, y, z) landmarks per frame.

    Parameters
    ----------
    video_path : Path

    Returns
    -------
    dict with keys:
      'landmarks' : np.ndarray shape (T, 63)
      'mask'      : np.ndarray shape (T,)  True = valid detection
      'n_frames'  : int  total frames
      'n_detected': int  frames where hand was detected
      'fps'       : float
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    landmarks = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)
    mask      = np.zeros(n_frames, dtype=bool)

    frame_idx        = 0
    n_detected       = 0
    last_valid       = None
    no_detect_streak = 0

    # Tasks API uses IMAGE mode for frame-by-frame processing
    options = _HandLandmarkerOpt(
        base_options=_BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
        running_mode=_RunningMode.IMAGE,
        num_hands=MP_MAX_NUM_HANDS,
        min_hand_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )

    with _HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]   # first detected hand
                lm   = np.array(
                    [[l.x, l.y, l.z] for l in hand],
                    dtype=np.float32
                ).flatten()   # shape (63,)

                landmarks[frame_idx] = lm
                mask[frame_idx]      = True
                last_valid           = lm
                n_detected          += 1
                no_detect_streak     = 0

            else:
                # Forward-fill strategy (DD-003)
                if last_valid is not None and no_detect_streak < MAX_FORWARD_FILL_FRAMES:
                    landmarks[frame_idx] = last_valid
                    mask[frame_idx]      = True
                # else: remains zero-filled
                no_detect_streak += 1

            frame_idx += 1

    cap.release()

    actual_frames = frame_idx
    landmarks = landmarks[:actual_frames]
    mask      = mask[:actual_frames]

    return {
        "landmarks": landmarks,
        "mask":      mask,
        "n_frames":  actual_frames,
        "n_detected": n_detected,
        "fps":       fps,
    }


def process_segment(uid: str, force: bool = False) -> dict:
    """
    Extracts landmarks for one segment and saves to cache.

    Parameters
    ----------
    uid   : str   segment UID, e.g. 'KxqgVJSEiW0_seg000'
    force : bool  re-extract even if cache file exists

    Returns
    -------
    dict with extraction statistics
    """
    npy_path  = CACHE_DIR / f"{uid}.npy"
    mask_path = CACHE_DIR / f"{uid}_mask.npy"

    if npy_path.exists() and mask_path.exists() and not force:
        # Already cached — compute stats from existing file for reporting
        lm   = np.load(npy_path)
        mask = np.load(mask_path)
        n_frames   = len(mask)
        n_detected = int(mask.sum())
        pct_missing = 1.0 - (n_detected / n_frames) if n_frames > 0 else 1.0
        return {
            "uid": uid,
            "n_frames": n_frames,
            "n_detected": n_detected,
            "pct_missing": pct_missing,
            "dropped": pct_missing > DROP_SEGMENT_THRESHOLD,
            "cached": True,
        }

    video_path = VIDEOS_DIR / f"{uid}.mp4"
    if not video_path.exists():
        print(f"  [WARN] Video not found: {video_path}")
        return {"uid": uid, "n_frames": 0, "n_detected": 0,
                "pct_missing": 1.0, "dropped": True, "cached": False}

    result = extract_landmarks_from_video(video_path)

    n_frames    = result["n_frames"]
    n_detected  = result["n_detected"]
    pct_missing = 1.0 - (n_detected / n_frames) if n_frames > 0 else 1.0
    dropped     = pct_missing > DROP_SEGMENT_THRESHOLD

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy_path,  result["landmarks"])
    np.save(mask_path, result["mask"])

    return {
        "uid":         uid,
        "n_frames":    n_frames,
        "n_detected":  n_detected,
        "pct_missing": pct_missing,
        "dropped":     dropped,
        "cached":      False,
    }


def run_extraction(uids: list, force: bool = False) -> list:
    """
    Runs landmark extraction on a list of UIDs with progress reporting.

    Parameters
    ----------
    uids  : list of str
    force : bool

    Returns
    -------
    list of stats dicts
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(uids, desc="Extracting landmarks", unit="seg")
    except ImportError:
        iterator = uids
        print(f"Extracting landmarks for {len(uids)} segments...")

    all_stats = []
    for uid in iterator:
        stats = process_segment(uid, force=force)
        all_stats.append(stats)

    # Write extraction stats CSV
    with open(EXTRACTION_STATS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "n_frames", "n_detected",
                                               "pct_missing", "dropped", "cached"])
        writer.writeheader()
        writer.writerows(all_stats)

    # Summary report
    total         = len(all_stats)
    n_dropped     = sum(s["dropped"] for s in all_stats)
    pct_dropped   = 100 * n_dropped / total if total else 0
    avg_missing   = 100 * sum(s["pct_missing"] for s in all_stats) / total if total else 0

    print("\n" + "=" * 60)
    print("  Extraction Summary")
    print("=" * 60)
    print(f"  Total segments processed : {total}")
    print(f"  Segments dropped (>{DROP_SEGMENT_THRESHOLD*100:.0f}% missing) : {n_dropped}  ({pct_dropped:.1f}%)")
    print(f"  Average % missing/frame  : {avg_missing:.1f}%")
    print(f"  Stats saved to           : {EXTRACTION_STATS_CSV}")
    print("=" * 60 + "\n")

    return all_stats


def load_landmarks(uid: str) -> tuple:
    """
    Loads cached landmark array and mask for a segment.

    Returns
    -------
    (landmarks: np.ndarray shape (T, 63),
     mask:      np.ndarray shape (T,))

    Raises FileNotFoundError if cache is missing (run extraction first).
    """
    npy_path  = CACHE_DIR / f"{uid}.npy"
    mask_path = CACHE_DIR / f"{uid}_mask.npy"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"Cache missing for {uid}. Run landmark_extractor.py first."
        )
    landmarks = np.load(npy_path,  mmap_mode="r")
    mask      = np.load(mask_path, mmap_mode="r")
    return landmarks, mask


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Landmark Extractor (Stage 2)")
    parser.add_argument("--uid",    type=str, default=None,
                        help="Extract a single segment by UID")
    parser.add_argument("--sample", type=int, default=None,
                        help="Extract N random segments (for quick verification)")
    parser.add_argument("--force",  action="store_true",
                        help="Re-extract even if cache exists")
    args = parser.parse_args()

    import pandas as pd
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.training.config import ANN_CSV

    all_uids = pd.read_csv(ANN_CSV)["uid"].tolist()

    if args.uid:
        uids = [args.uid]
    elif args.sample:
        random.seed(RANDOM_SEED)
        uids = random.sample(all_uids, min(args.sample, len(all_uids)))
        print(f"Sampling {len(uids)} segments: {uids}")
    else:
        uids = all_uids

    t0 = time.time()
    run_extraction(uids, force=args.force)
    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s  ({elapsed/len(uids):.2f}s per segment)")
