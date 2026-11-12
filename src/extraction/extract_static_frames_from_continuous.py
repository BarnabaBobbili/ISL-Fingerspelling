"""
src/extraction/extract_static_frames_from_continuous.py
========================================================
Extracts the best single frame for each fingerspelled letter from the
continuous ISL fingerspelling dataset videos.

APPROACH
--------
1. Read localization_timestamps.csv — each row is one video segment with a
   transcript (e.g. "sulochana das") and start/end seconds.
2. The segment time is divided proportionally across each alphabetic character
   in the transcript (spaces & hyphens are skipped).
3. For each character window, we sample N_SAMPLES evenly spaced candidate
   frames and run MediaPipe hand detection.
4. The frame with the highest hand-detection confidence is saved as a JPEG
   to: <OUTPUT_DIR>/<LETTER>/<uid>_char<idx>_<LETTER>.jpg

OUTPUT
------
data/static-frames-extracted/
    A/
        6OlSDCcwM10_seg000_char0_A.jpg
        ...
    B/
    ...
    Z/

Usage
-----
    python src/extraction/extract_static_frames_from_continuous.py
    python src/extraction/extract_static_frames_from_continuous.py --max-videos 50
    python src/extraction/extract_static_frames_from_continuous.py --max-videos 30 --samples-per-char 5
"""

import argparse
import csv
import sys
from pathlib import Path
import random

import cv2
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent.parent
DATASET_DIR    = ROOT / "data" / "continious-Fingerspelling-dataset"
VIDEO_DIR      = DATASET_DIR / "videos"
TIMESTAMPS_CSV = DATASET_DIR / "localization_timestamps.csv"
OUTPUT_DIR     = ROOT / "data" / "static-frames-extracted"
TASK_MODEL     = ROOT / "hand_landmarker.task"

# ── tunables ─────────────────────────────────────────────────────────────────
N_SAMPLES_PER_CHAR = 7     # candidate frames per character window
MIN_HAND_CONF      = 0.30  # minimum detection confidence to accept a frame
RANDOM_SEED        = 42

VALID_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ── MediaPipe detector ───────────────────────────────────────────────────────
def _build_detector(min_conf: float):
    """Build a MediaPipe HandLandmarker (Tasks API) for single-image mode."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    if not TASK_MODEL.exists():
        print(f"[ERROR] hand_landmarker.task not found at {TASK_MODEL}")
        print("  Download: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    base_opts = mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def _best_conf(result) -> float:
    """Return the highest detection score from a HandLandmarker result."""
    if not result.handedness:
        return 0.0
    scores = [h[0].score for h in result.handedness if h]
    return max(scores) if scores else 0.0


# ── letter-time segmentation ─────────────────────────────────────────────────
def _letters_only(text: str) -> list:
    """Return only A-Z chars (uppercased) from transcript."""
    return [c.upper() for c in text if c.upper() in VALID_LETTERS]


def _char_windows(start_sec: float, end_sec: float, letters: list):
    """
    Divide [start_sec, end_sec] proportionally among `letters`.
    Returns list of (letter, char_start, char_end) tuples.
    """
    n = len(letters)
    if n == 0:
        return []
    duration = end_sec - start_sec
    per_char  = duration / n
    windows = []
    for i, letter in enumerate(letters):
        cs = start_sec + i * per_char
        ce = cs + per_char
        windows.append((letter, cs, ce))
    return windows


# ── frame extraction ─────────────────────────────────────────────────────────
def _sample_timestamps(cs: float, ce: float, n: int) -> list:
    """Return n evenly-spaced timestamps inside [cs, ce], skipping edge transitions."""
    if n == 1:
        return [(cs + ce) / 2]
    margin = (ce - cs) * 0.15  # skip first/last 15% (transition frames)
    s = cs + margin
    e = ce - margin
    if s >= e:
        s, e = cs, ce
    return [s + (e - s) * i / (n - 1) for i in range(n)]


def extract_best_frame(cap, ts_list, detector, mp_Image, mp_ImageFormat):
    """
    Given a list of candidate timestamps (seconds), grab each frame,
    run hand detection, return (best_frame_bgr, best_confidence).
    Returns (None, 0.0) if no hand found at any candidate.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    best_frame = None
    best_conf  = 0.0

    for ts in ts_list:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_Image(image_format=mp_ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        conf = _best_conf(result)

        if conf > best_conf:
            best_conf  = conf
            best_frame = frame.copy()

    return best_frame, best_conf


# ── main pipeline ─────────────────────────────────────────────────────────────
def run(max_videos: int = 50, samples_per_char: int = N_SAMPLES_PER_CHAR,
        min_conf: float = MIN_HAND_CONF, seed: int = RANDOM_SEED):
    """
    Main extraction pipeline.

    Parameters
    ----------
    max_videos      : Process at most this many video segment files.
    samples_per_char: Number of candidate frames per character time window.
    min_conf        : Minimum MediaPipe confidence to keep a frame.
    seed            : Random seed for shuffling video order.
    """
    import mediapipe as mp

    # Read timestamps CSV
    rows = []
    with open(TIMESTAMPS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[info] Loaded {len(rows)} segment entries from CSV")

    # Build list of valid (video_file, uid, start_sec, end_sec, letters)
    # IMPORTANT: The .mp4 files are pre-clipped segments — their internal
    # timestamps start at 0, not at the original-video start_sec.
    # We use 0 → duration_sec as the scan window inside each segment file.
    entries = []
    for row in rows:
        vid_id   = row["video_id"]
        seg_idx  = int(row["segment_index"])
        uid      = f"{vid_id}_seg{seg_idx:03d}"
        vid_file = VIDEO_DIR / f"{uid}.mp4"
        if not vid_file.exists():
            continue
        try:
            duration_sec = float(row["duration_sec"])
        except (ValueError, KeyError):
            try:
                duration_sec = float(row["end_sec"]) - float(row["start_sec"])
            except (ValueError, KeyError):
                continue
        if duration_sec <= 0:
            continue
        transcript = row["transcript"].strip()
        letters = _letters_only(transcript)
        if not letters:
            continue
        # Segment-relative: always starts at 0
        entries.append((vid_file, uid, 0.0, duration_sec, letters))

    print(f"[info] {len(entries)} segments have matching video files")

    # Shuffle and cap
    rng = random.Random(seed)
    rng.shuffle(entries)
    entries = entries[:max_videos]
    print(f"[info] Processing {len(entries)} video segments (max_videos={max_videos})\n")

    # Create output dirs
    for letter in VALID_LETTERS:
        (OUTPUT_DIR / letter).mkdir(parents=True, exist_ok=True)

    # Build detector
    detector = _build_detector(min_conf)
    mp_Image = mp.Image
    mp_IFmt  = mp.ImageFormat

    stats = {l: 0 for l in VALID_LETTERS}
    total_saved   = 0
    total_skipped = 0

    for seg_i, (vid_path, uid, start_sec, end_sec, letters) in enumerate(entries):
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open {vid_path.name}")
            continue

        windows = _char_windows(start_sec, end_sec, letters)
        seg_saved = 0

        for char_i, (letter, cs, ce) in enumerate(windows):
            ts_list = _sample_timestamps(cs, ce, samples_per_char)
            frame, conf = extract_best_frame(cap, ts_list, detector, mp_Image, mp_IFmt)

            if frame is None or conf < min_conf:
                total_skipped += 1
                continue

            out_name = f"{uid}_char{char_i:02d}_{letter}.jpg"
            out_path = OUTPUT_DIR / letter / out_name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats[letter] += 1
            total_saved   += 1
            seg_saved     += 1

        cap.release()
        print(f"  [{seg_i+1:3d}/{len(entries)}] {uid}  "
              f"letters={len(letters)}  saved={seg_saved}  "
              f"transcript=\"{''.join(letters)}\"")

    detector.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Static Frame Extraction — Summary")
    print(f"{'='*60}")
    print(f"  Segments processed : {len(entries)}")
    print(f"  Frames saved       : {total_saved}")
    print(f"  Frames skipped     : {total_skipped}  (no/low-conf hand)")
    print(f"\n  Per-letter counts:")
    for letter in sorted(VALID_LETTERS):
        bar = "#" * min(stats[letter], 40)
        print(f"    {letter} : {stats[letter]:4d}  {bar}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Save stats CSV
    stats_path = OUTPUT_DIR / "extraction_stats.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["letter", "count"])
        for letter in sorted(VALID_LETTERS):
            w.writerow([letter, stats[letter]])
    print(f"[info] Stats saved to {stats_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract best static fingerspelling frames from continuous videos"
    )
    parser.add_argument(
        "--max-videos", type=int, default=50,
        help="Maximum number of video segments to process (default: 50)"
    )
    parser.add_argument(
        "--samples-per-char", type=int, default=N_SAMPLES_PER_CHAR,
        help=f"Candidate frames to test per character window (default: {N_SAMPLES_PER_CHAR})"
    )
    parser.add_argument(
        "--min-conf", type=float, default=MIN_HAND_CONF,
        help=f"Minimum MediaPipe hand confidence to accept a frame (default: {MIN_HAND_CONF})"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for video shuffling (default: 42)"
    )
    args = parser.parse_args()

    run(
        max_videos=args.max_videos,
        samples_per_char=args.samples_per_char,
        min_conf=args.min_conf,
        seed=args.seed,
    )
