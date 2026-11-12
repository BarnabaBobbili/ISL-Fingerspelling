"""
src/static/extract_static_landmarks.py
=======================================
Extracts MediaPipe hand landmarks (BOTH hands) from all static ISL images.

WHY BOTH HANDS:
  ISL fingerspelling is a BSL-derived two-handed system. The dominant hand
  acts as the "pen" and the secondary hand acts as the "paper". Extracting
  only one hand would discard the inter-hand geometry that defines many letters.

OUTPUT FORMAT (per image → .npz file):
  right_lm      : (21, 3) float32 — right hand landmarks, wrist-relative normalized
  left_lm       : (21, 3) float32 — left hand landmarks, wrist-relative normalized
  right_present : bool
  left_present  : bool
  label         : str   (e.g. 'A')

Run with:
  python src/static/extract_static_landmarks.py
  python src/static/extract_static_landmarks.py --sample 10   # quick test
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    STATIC_ISL_DIR, KAGGLE_ISL_DIR, ISL_DATA_DIR, STATIC_CACHE_DIR,
    STATIC_LETTERS, LETTER_TO_IDX,
    NUM_LANDMARKS, LANDMARK_DIM, MP_MAX_NUM_HANDS_STATIC
)
from src.preprocessing.normalizer import normalize_hand_static

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe HandLandmarker setup (Tasks API — Python 3.13 compatible)
# ─────────────────────────────────────────────────────────────────────────────
TASK_MODEL_PATH = Path(__file__).parent.parent.parent / "hand_landmarker.task"


def _build_detector():
    """Create a MediaPipe HandLandmarker for static image detection."""
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=MP_MAX_NUM_HANDS_STATIC,          # detect BOTH hands
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_from_image(img_path: Path, detector) -> dict:
    """
    Run MediaPipe on one image. Returns a dict with:
      right_lm, left_lm (both (21,3)), right_present, left_present.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return _empty_result()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = detector.detect(mp_image)

    right_lm    = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    left_lm     = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    right_present = False
    left_present  = False

    if result.hand_landmarks:
        for i, handedness_list in enumerate(result.handedness):
            # MediaPipe handedness label is from the image's perspective
            label = handedness_list[0].display_name  # 'Left' or 'Right'
            lms   = result.hand_landmarks[i]
            arr   = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            arr   = normalize_hand_static(arr)

            if label == "Right":
                right_lm      = arr
                right_present = True
            else:
                left_lm       = arr
                left_present  = True

    return {
        "right_lm":      right_lm,
        "left_lm":       left_lm,
        "right_present": right_present,
        "left_present":  left_present,
    }


def _empty_result() -> dict:
    return {
        "right_lm":      np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32),
        "left_lm":       np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32),
        "right_present": False,
        "left_present":  False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset inventory helpers
# ─────────────────────────────────────────────────────────────────────────────
def _collect_images() -> list[tuple[Path, str, str]]:
    """
    Walk all three static datasets and collect (image_path, label, source) tuples.
    Only letters A–Z are included (digits are excluded for fingerspelling task).

    Sources:
      static_isl  — data/static-isl-dataset/Static_ISL/  (~1200 imgs/class, A-Z)
      kaggle      — data/static-isl-dataset/Kaggle_ISL_Dataset/ (~25 imgs/class, 23 letters)
      isl_data    — data/ISL_DATA/  (~460-700 imgs/class, A-Z)
    """
    items = []

    # Static_ISL — folders named A, B, ..., Z
    for letter in STATIC_LETTERS:
        folder = STATIC_ISL_DIR / letter
        if folder.is_dir():
            for img in sorted(folder.glob("*.jpg")):
                items.append((img, letter, "static_isl"))
            for img in sorted(folder.glob("*.png")):
                items.append((img, letter, "static_isl"))

    # Kaggle_ISL_Dataset — folders named A, B, ..., Z (23 letters)
    for letter in STATIC_LETTERS:
        folder = KAGGLE_ISL_DIR / letter
        if folder.is_dir():
            for img in sorted(folder.glob("*.jpg")):
                items.append((img, letter, "kaggle"))
            for img in sorted(folder.glob("*.png")):
                items.append((img, letter, "kaggle"))

    # ISL_DATA — folders named A-Z (also has 1-9 digits which we skip)
    if ISL_DATA_DIR.exists():
        for letter in STATIC_LETTERS:
            folder = ISL_DATA_DIR / letter
            if folder.is_dir():
                for img in sorted(folder.glob("*.jpg")):
                    items.append((img, letter, "isl_data"))
                for img in sorted(folder.glob("*.png")):
                    items.append((img, letter, "isl_data"))
    else:
        print(f"[extract_static] WARNING: ISL_DATA not found at {ISL_DATA_DIR}")

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────
def run_extraction(sample: int = None, force: bool = False):
    if not TASK_MODEL_PATH.exists():
        print(f"[extract_static] ERROR: MediaPipe task model not found at {TASK_MODEL_PATH}")
        print("  Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    items = _collect_images()
    from collections import Counter
    src_counts = Counter(src for _, _, src in items)
    print(f"[extract_static] Found {len(items)} images across all datasets:")
    for src, cnt in sorted(src_counts.items()):
        print(f"  {src:<12}: {cnt:,} images")

    if sample:
        import random
        random.seed(42)
        items = random.sample(items, min(sample, len(items)))
        print(f"[extract_static] Sampling {len(items)} images")

    # Create cache structure
    for letter in STATIC_LETTERS:
        (STATIC_CACHE_DIR / letter).mkdir(parents=True, exist_ok=True)

    detector = _build_detector()

    stats = {lt: {"total": 0, "both": 0, "one": 0, "none": 0} for lt in STATIC_LETTERS}
    t0 = time.time()

    for img_path, label, source in tqdm(items, desc="Extracting static landmarks", unit="img"):
        # Cache path — use stem + source prefix to avoid collisions
        cache_path = STATIC_CACHE_DIR / label / f"{source}_{img_path.stem}.npz"

        if cache_path.exists() and not force:
            # Still update stats from cached file
            cached = np.load(cache_path, allow_pickle=True)
            rp = bool(cached["right_present"])
            lp = bool(cached["left_present"])
        else:
            result = extract_from_image(img_path, detector)
            rp = result["right_present"]
            lp = result["left_present"]
            np.savez_compressed(
                cache_path,
                right_lm      = result["right_lm"],
                left_lm       = result["left_lm"],
                right_present = np.array(rp),
                left_present  = np.array(lp),
                label         = np.array(label),
            )

        stats[label]["total"] += 1
        if rp and lp:
            stats[label]["both"] += 1
        elif rp or lp:
            stats[label]["one"] += 1
        else:
            stats[label]["none"] += 1

    detector.close()
    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'='*65}")
    print(f"  Static Landmark Extraction Summary")
    print(f"{'='*65}")
    total = sum(v["total"] for v in stats.values())
    both  = sum(v["both"]  for v in stats.values())
    one   = sum(v["one"]   for v in stats.values())
    none  = sum(v["none"]  for v in stats.values())
    print(f"  Total images processed : {total}")
    print(f"  Both hands detected    : {both}  ({100*both/max(total,1):.1f}%)")
    print(f"  One hand detected      : {one}   ({100*one/max(total,1):.1f}%)")
    print(f"  No hands detected      : {none}  ({100*none/max(total,1):.1f}%)")
    print(f"  Time: {elapsed:.1f}s  ({elapsed/max(total,1):.2f}s per image)")
    print(f"{'='*65}\n")

    # Save per-letter stats CSV
    stats_path = STATIC_CACHE_DIR / "extraction_stats.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["letter","total","both","one","none"])
        w.writeheader()
        for lt, v in stats.items():
            w.writerow({"letter": lt, **v})
    print(f"[extract_static] Stats saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract static ISL landmarks")
    parser.add_argument("--sample", type=int, default=None,
                        help="If set, extract only N random images (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cache file exists")
    args = parser.parse_args()
    run_extraction(sample=args.sample, force=args.force)
