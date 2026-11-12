"""
src/extraction/visualize_landmarks.py
======================================
Stage 2 Checkpoint: Overlay extracted landmarks on video frames.

PURPOSE
-------
This script is used for the Stage 2 checkpoint: you visually verify that
the extracted landmarks correctly track the signing hand across frames.
If you see the landmark dots drifting off the hand, or snapping to wrong
positions, there is a bug in the extraction that must be fixed before
training on contaminated data.

WHAT TO LOOK FOR
-----------------
CORRECT: 21 dots consistently covering the fingers, palm, and wrist.
  They move smoothly as the hand moves.
WRONG:
  - Dots appear at (0,0) = top-left corner (zero-filled frame)
  - Dots track the face/background instead of the hand
  - Dots freeze while hand moves (forward-fill running too long)

USAGE
-----
  # Visualize 3 random segments from cache:
  python src/extraction/visualize_landmarks.py

  # Visualize a specific segment:
  python src/extraction/visualize_landmarks.py --uid KxqgVJSEiW0_seg000

  # Save to video file instead of displaying:
  python src/extraction/visualize_landmarks.py --uid KxqgVJSEiW0_seg000 --save
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import VIDEOS_DIR, CACHE_DIR, RESULTS_DIR, RANDOM_SEED, NUM_LANDMARKS
from src.extraction.landmark_extractor import load_landmarks

# MediaPipe hand connections (for drawing skeleton lines)
# Defined manually to avoid importing mediapipe just for this
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

DOT_COLOR  = (0, 255, 0)    # green for landmark dots
LINE_COLOR = (0, 200, 255)  # orange for skeleton lines
WRIST_COLOR = (255, 0, 0)   # blue for wrist (landmark 0)


def overlay_landmarks_on_frame(
    frame: np.ndarray,
    landmark_vec: np.ndarray,
    mask_val: bool = True
) -> np.ndarray:
    """
    Draws 21 landmark dots and skeleton lines onto a BGR frame.

    Parameters
    ----------
    frame        : np.ndarray  (H, W, 3) BGR frame
    landmark_vec : np.ndarray  (63,) flat array of [x, y, z] * 21
    mask_val     : bool        if False, the frame had no detection (draw red X)

    Returns
    -------
    annotated frame (copy, not in-place)
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    if not mask_val:
        # Draw red X to indicate missing detection
        cv2.line(vis, (0, 0), (w, h), (0, 0, 255), 2)
        cv2.line(vis, (w, 0), (0, h), (0, 0, 255), 2)
        return vis

    # Reshape (63,) → (21, 3)
    lm = landmark_vec.reshape(NUM_LANDMARKS, 3)
    # x, y are normalized [0,1] in MediaPipe — scale to pixel coords
    pts = [(int(lm[i, 0] * w), int(lm[i, 1] * h)) for i in range(NUM_LANDMARKS)]

    # Draw connections first (behind dots)
    for a, b in HAND_CONNECTIONS:
        cv2.line(vis, pts[a], pts[b], LINE_COLOR, 1)

    # Draw landmark dots
    for i, (px, py) in enumerate(pts):
        color = WRIST_COLOR if i == 0 else DOT_COLOR
        radius = 5 if i == 0 else 3
        cv2.circle(vis, (px, py), radius, color, -1)

    return vis


def visualize_segment(uid: str, save: bool = False, max_frames: int = 300):
    """
    Shows or saves a landmark-overlaid video for one segment.

    Parameters
    ----------
    uid        : str   segment UID
    save       : bool  if True, save to results/ instead of displaying
    max_frames : int   cap at this many frames for display speed
    """
    video_path = VIDEOS_DIR / f"{uid}.mp4"
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    landmarks, mask = load_landmarks(uid)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = None
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{uid}_landmarks.mp4"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"Saving to {out_path}")

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        lm_vec   = landmarks[frame_idx] if frame_idx < len(landmarks) else np.zeros(63)
        mask_val = bool(mask[frame_idx]) if frame_idx < len(mask) else False

        annotated = overlay_landmarks_on_frame(frame, lm_vec, mask_val)

        # Add info overlay
        info = f"UID: {uid}  Frame: {frame_idx}  Detected: {mask_val}"
        cv2.putText(annotated, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if save and out is not None:
            out.write(annotated)
        else:
            cv2.imshow("ISL Landmark Overlay (press Q to quit, Space to pause)", annotated)
            key = cv2.waitKey(int(1000 / fps))
            if key == ord("q"):
                break
            elif key == ord(" "):
                cv2.waitKey(0)   # pause until Space again

        frame_idx += 1

    cap.release()
    if out:
        out.release()
    if not save:
        cv2.destroyAllWindows()

    # Print detection stats for this segment
    n_valid = int(mask[:frame_idx].sum())
    pct     = 100 * n_valid / frame_idx if frame_idx > 0 else 0
    print(f"\n[{uid}]  Frames shown: {frame_idx} | "
          f"Detected: {n_valid} ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize landmarks on ISL video")
    parser.add_argument("--uid",   type=str, default=None)
    parser.add_argument("--n",     type=int, default=3,
                        help="Number of random segments to visualize (if no --uid)")
    parser.add_argument("--save",  action="store_true",
                        help="Save to results/ instead of displaying")
    args = parser.parse_args()

    import pandas as pd
    from src.training.config import ANN_CSV
    all_uids = pd.read_csv(ANN_CSV)["uid"].tolist()

    if args.uid:
        uids = [args.uid]
    else:
        random.seed(RANDOM_SEED)
        uids = random.sample(all_uids, min(args.n, len(all_uids)))
        print(f"Randomly selected segments: {uids}")

    for uid in uids:
        print(f"\nVisualizing: {uid}")
        visualize_segment(uid, save=args.save)
