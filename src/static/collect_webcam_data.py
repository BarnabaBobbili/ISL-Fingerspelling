"""
src/static/collect_webcam_data.py
===================================
Interactive tool for collecting real webcam hand landmark data.

WHY THIS IS NEEDED:
  The Static_ISL training images are small synthetic/sketch-like images.
  MediaPipe extracts very different landmark patterns from sketches vs.
  real webcam footage. This "domain gap" causes the model to fail on live data.

  This script collects YOUR REAL HAND data, which we then use to fine-tune
  the DH-GAT model. Even 30–50 frames per letter dramatically improves
  real-world accuracy. This is called "user calibration" in production systems.

HOW TO USE:
  1. Run this script:  python src/static/collect_webcam_data.py
  2. For each letter displayed:
       - Form the ISL sign with your hand(s) in front of the camera
       - Hold it steady when the skeleton turns SOLID GREEN
       - Press SPACE to capture 30 frames of that letter
       - Press N to skip the current letter
       - Press Q to quit early
  3. After collection, fine-tune the model:
       python src/static/train_static.py --fine-tune

STORAGE:  cache/webcam_data/{LETTER}/*.npz
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    STATIC_LETTERS, NUM_LANDMARKS, LANDMARK_DIM,
    MP_MAX_NUM_HANDS_STATIC,
    HAND_EDGES, ROOT
)
from src.preprocessing.normalizer import normalize_hand_static

TASK_MODEL_PATH  = ROOT / "hand_landmarker.task"
WEBCAM_DATA_DIR  = ROOT / "cache" / "webcam_data"
FRAMES_PER_LETTER = 50   # number of frames to capture per letter


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup (same as extract_static_landmarks.py — unflipped detection)
# ─────────────────────────────────────────────────────────────────────────────
def _build_detector():
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    base_options = mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=MP_MAX_NUM_HANDS_STATIC,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def _detect(frame_bgr, detector):
    """Detect on unflipped frame. Returns right_lm, left_lm, right_present, left_present."""
    img_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect(mp_image)

    right_lm    = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    left_lm     = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    right_present = False
    left_present  = False
    raw_lms       = {}

    if result.hand_landmarks:
        for i, handedness_list in enumerate(result.handedness):
            label = handedness_list[0].display_name
            lms   = result.hand_landmarks[i]
            arr   = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            arr_n = normalize_hand_static(arr)
            raw_lms[label] = lms
            if label == "Right":
                right_lm      = arr_n
                right_present = True
            else:
                left_lm       = arr_n
                left_present  = True

    return right_lm, left_lm, right_present, left_present, raw_lms


def _draw_skeleton(frame, raw_lms, flip_x=True):
    """Draw skeleton for all detected hands on the (flipped) display frame."""
    colors = {"Right": (0, 220, 100), "Left": (100, 180, 255)}
    h, w = frame.shape[:2]
    for label, lms in raw_lms.items():
        color = colors.get(label, (200, 200, 200))
        if flip_x:
            pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in lms]
        else:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for i, j in HAND_EDGES:
            cv2.line(frame, pts[i], pts[j], color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1)
            cv2.circle(frame, pt, 4, (255, 255, 255), 1)


def _draw_collection_ui(frame, letter, captured, total, status, hand_ok):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (15, 15, 25), -1)
    frame[:] = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    # Big letter
    cv2.putText(frame, f"Sign:  {letter}", (20, 70),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 220, 60), 3)

    # Progress bar
    bar_w = int((w - 40) * captured / total)
    cv2.rectangle(frame, (20, 108), (w - 20, 122), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 108), (20 + bar_w, 122), (0, 220, 100), -1)
    cv2.putText(frame, f"{captured}/{total} frames", (w - 160, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Status message
    col = (0, 255, 140) if hand_ok else (100, 100, 100)
    cv2.putText(frame, status, (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1)

    # Controls
    cv2.putText(frame, "SPACE: capture   N: skip   Q: quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────
def collect():
    if not TASK_MODEL_PATH.exists():
        print(f"[collect] ERROR: {TASK_MODEL_PATH} not found")
        sys.exit(1)

    WEBCAM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for letter in STATIC_LETTERS:
        (WEBCAM_DATA_DIR / letter).mkdir(exist_ok=True)

    detector = _build_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[collect] ERROR: Cannot open camera")
        sys.exit(1)

    print(f"[collect] Starting collection. {len(STATIC_LETTERS)} letters × {FRAMES_PER_LETTER} frames.")
    print(f"  Data will be saved to: {WEBCAM_DATA_DIR}")
    print(f"  Controls: SPACE=capture  N=skip  Q=quit\n")

    stats = {}
    for letter in STATIC_LETTERS:
        captured  = 0
        recording = False
        buffer    = []
        frame_idx = 0

        print(f"  → Letter: {letter}  (SPACE to start capturing, N to skip)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Detect on unflipped
            rr, ll, rp, lp, raw_lms = _detect(frame, detector)
            hand_ok = rp or lp

            # Flip display
            display = cv2.flip(frame, 1)
            _draw_skeleton(display, raw_lms, flip_x=True)

            if recording and hand_ok:
                buffer.append((rr, ll, rp, lp))
                captured = len(buffer)
                if captured >= FRAMES_PER_LETTER:
                    # Save all frames
                    for fi, (rr_, ll_, rp_, lp_) in enumerate(buffer):
                        save_path = WEBCAM_DATA_DIR / letter / f"webcam_{frame_idx:06d}_{fi:03d}.npz"
                        np.savez_compressed(
                            save_path,
                            right_lm      = rr_,
                            left_lm       = ll_,
                            right_present = np.array(rp_),
                            left_present  = np.array(lp_),
                            label         = np.array(letter),
                        )
                    stats[letter] = FRAMES_PER_LETTER
                    print(f"    ✓ Captured {FRAMES_PER_LETTER} frames for '{letter}'")
                    recording = False
                    buffer    = []
                    time.sleep(0.5)
                    break

            status = ("● Recording..." if recording
                      else "✋ Hold your sign, then press SPACE" if hand_ok
                      else "⚠ No hand detected")

            _draw_collection_ui(display, letter, captured, FRAMES_PER_LETTER, status, hand_ok)
            cv2.imshow("ISL Data Collection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                detector.close()
                cv2.destroyAllWindows()
                _print_summary(stats)
                return
            elif key == ord("n"):
                print(f"    ⊘ Skipped '{letter}'")
                break
            elif key == ord(" ") and hand_ok:
                recording = True
                buffer    = []
                print(f"    ● Recording '{letter}'...")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    _print_summary(stats)


def _print_summary(stats):
    print(f"\n{'='*50}")
    print(f"  Data Collection Summary")
    print(f"{'='*50}")
    for letter in STATIC_LETTERS:
        n = stats.get(letter, 0)
        mark = "✓" if n > 0 else "⊘"
        print(f"  {mark}  '{letter}'  :  {n} frames")
    total = sum(stats.values())
    print(f"\n  Total frames collected: {total}")
    print(f"  Saved to: {WEBCAM_DATA_DIR}")
    if total > 0:
        print(f"\n  Next step — fine-tune the model:")
        print(f"    python src/static/train_static.py --fine-tune")
    print(f"{'='*50}")


if __name__ == "__main__":
    collect()
