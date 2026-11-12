"""
src/inference/webcam_demo.py
==============================
Stage 10: Real-time webcam fingerspelling recognition.

PIPELINE
--------
  Webcam frame
    ↓
  MediaPipe Hands (per frame, ~30 FPS)
    ↓
  Normalize landmarks (wrist-relative + scale)
    ↓
  Temporal buffer (sliding window, last K frames)
    ↓
  Model inference (every INFERENCE_STRIDE frames)
    ↓
  CTC greedy decode
    ↓
  Display text overlay on OpenCV window

DESIGN DECISIONS (Stage 10)
-----------------------------
1. SLIDING WINDOW (not segment-based):
   WHY: We don't know where one word ends and the next begins in real-time.
   Solution: maintain a circular buffer of the last K frames (~3 seconds at 30 FPS).
   Run inference every INFERENCE_STRIDE frames to balance latency vs. CPU load.

2. INFERENCE STRIDE = 15 frames (~0.5 sec at 30 FPS):
   WHY: Running on every frame is wasteful — the model output changes slowly.
   15 frames gives a new prediction twice per second, which feels responsive.

3. GREEDY DECODE ONLY (not beam search) for real-time:
   WHY: Beam search is ~5× slower. At 30 FPS with 0.5s stride, we have 500ms
   to do inference — greedy is well within budget; beam search might not be.

4. TEXT ACCUMULATION:
   We don't reset the displayed text every inference step. Instead we append
   new non-empty predictions. The user presses SPACE to clear.

USAGE
-----
  python src/inference/webcam_demo.py --model bilstm
  python src/inference/webcam_demo.py --model conformer
  python src/inference/webcam_demo.py --model conformer --camera 1

CONTROLS
--------
  Q      : quit
  SPACE  : clear recognized text
  P      : pause/unpause
"""

import argparse
import sys
import time
import collections
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    CKPT_DIR, FEATURE_DIM, NUM_LANDMARKS,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE, MP_MAX_NUM_HANDS,
    MAX_FORWARD_FILL_FRAMES,
)
from src.preprocessing.normalizer import normalize_frame
from src.evaluation.decoder import ctc_greedy_decode

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BUFFER_SIZE      = 90    # ~3 seconds at 30 FPS
INFERENCE_STRIDE = 15    # run model every N frames (~0.5 sec)
MIN_FRAMES       = 20    # minimum frames before running inference

# UI Colors (BGR)
COLOR_BG         = (20, 20, 20)
COLOR_GREEN      = (0, 220, 100)
COLOR_YELLOW     = (0, 200, 255)
COLOR_WHITE      = (255, 255, 255)
COLOR_RED        = (60, 60, 220)
COLOR_LANDMARK   = (0, 255, 0)
COLOR_WRIST      = (255, 0, 0)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


def load_model(model_name: str, device: str):
    """Loads the best checkpoint for the given model."""
    ckpt_path = CKPT_DIR / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. "
            f"Train the model first: python src/training/train.py --model {model_name}"
        )
    if model_name == "bilstm":
        from src.models.bilstm_ctc import build_bilstm
        model = build_bilstm()
    else:
        from src.models.conformer_ctc import build_conformer
        model = build_conformer()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"[webcam] Loaded {model_name} checkpoint (val CER={ckpt.get('best_val_cer', '?'):.4f})")
    return model


def draw_hand_landmarks(frame: np.ndarray, lm_vec: np.ndarray) -> np.ndarray:
    """Draws landmarks on the frame (same as visualize_landmarks.py)."""
    h, w = frame.shape[:2]
    lm = lm_vec.reshape(NUM_LANDMARKS, 3)
    # Use raw normalized MediaPipe coords (0-1 range)
    pts = [(int(lm[i,0]*w), int(lm[i,1]*h)) for i in range(NUM_LANDMARKS)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], COLOR_GREEN, 1)
    for i, (px, py) in enumerate(pts):
        col = COLOR_WRIST if i == 0 else COLOR_LANDMARK
        cv2.circle(frame, (px, py), 5 if i==0 else 3, col, -1)
    return frame


def draw_ui(frame: np.ndarray, state: dict) -> np.ndarray:
    """
    Renders the HUD overlay: status, current prediction, full recognized text.
    """
    h, w = frame.shape[:2]

    # Dark semi-transparent bar at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 160), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 35), (30, 30, 30), -1)
    cv2.putText(frame, f"ISL Fingerspelling  |  Model: {state['model_name'].upper()}  |  Q=quit  SPACE=clear  P=pause",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

    # FPS and hand status
    fps_text = f"FPS: {state['fps']:.1f}  |  Hand: {'DETECTED' if state['hand_detected'] else 'NOT FOUND'}"
    color = COLOR_GREEN if state['hand_detected'] else COLOR_RED
    cv2.putText(frame, fps_text, (8, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Current live prediction
    cv2.putText(frame, "Current:", (8, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 1)
    cv2.putText(frame, f"  {state['current_pred']}",
                (8, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_YELLOW, 2)

    # Accumulated recognized text
    cv2.putText(frame, "Recognized:", (8, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    # Truncate if too long for display
    full_text = state['full_text'][-60:] if len(state['full_text']) > 60 else state['full_text']
    cv2.putText(frame, full_text, (8, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_GREEN, 2)

    # Inference latency
    if state['inf_ms'] > 0:
        cv2.putText(frame, f"Inference: {state['inf_ms']:.1f}ms",
                    (w - 200, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # Buffer fill bar
    fill = min(1.0, len(state['buffer']) / BUFFER_SIZE)
    bar_w = int((w - 20) * fill)
    cv2.rectangle(frame, (10, h - 155), (10 + bar_w, h - 145), COLOR_GREEN, -1)
    cv2.rectangle(frame, (10, h - 155), (w - 10, h - 145), COLOR_WHITE, 1)
    cv2.putText(frame, "Buffer", (10, h - 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

    if state['paused']:
        cv2.putText(frame, "PAUSED", (w//2 - 60, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLOR_RED, 4)

    return frame


def run_webcam(model_name: str, camera_idx: int = 0):
    """Main real-time inference loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[webcam] Device: {device}")

    model = load_model(model_name, device)

    # MediaPipe setup (Tasks API — compatible with Python 3.11+ / mediapipe 0.10+)
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from src.training.config import ROOT
        mp_model = ROOT / "hand_landmarker.task"
        if not mp_model.exists():
            print(f"[webcam] ERROR: hand_landmarker.task not found at {mp_model}")
            return
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(mp_model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=MP_MAX_NUM_HANDS,
            min_hand_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
        )
        detector = mp_vision.HandLandmarker.create_from_options(options)
        print("[webcam] MediaPipe HandLandmarker initialized (Tasks API)")
    except Exception as e:
        print(f"[webcam] ERROR initializing MediaPipe: {e}")
        return

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[webcam] ERROR: Cannot open camera {camera_idx}")
        return

    print(f"[webcam] Camera {camera_idx} opened. Starting inference loop...")
    print("         Press Q to quit, SPACE to clear text, P to pause.")

    # State
    frame_buffer = collections.deque(maxlen=BUFFER_SIZE)  # raw MediaPipe landmarks
    last_valid   = None
    no_detect_streak = 0
    frame_count  = 0
    inf_latency  = 0.0
    full_text    = ""
    current_pred = ""
    paused       = False

    # FPS tracking
    fps_buffer   = collections.deque(maxlen=30)
    prev_time    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[webcam] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural interaction
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # FPS
        now = time.time()
        fps_buffer.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = sum(fps_buffer) / len(fps_buffer)

        hand_detected = False

        if not paused:
            # MediaPipe inference (Tasks API)
            import mediapipe as mp
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
            lm_vec  = None

            if result.hand_landmarks:
                hand   = result.hand_landmarks[0]
                lm_vec = np.array([[l.x, l.y, l.z] for l in hand],
                                   dtype=np.float32).flatten()
                last_valid       = lm_vec.copy()
                no_detect_streak = 0
                hand_detected    = True
            else:
                # Forward-fill
                if last_valid is not None and no_detect_streak < MAX_FORWARD_FILL_FRAMES:
                    lm_vec = last_valid.copy()
                no_detect_streak += 1

            # Draw landmarks on frame (before normalization)
            if lm_vec is not None:
                draw_hand_landmarks(frame, lm_vec)

            # Add to buffer (normalized)
            if lm_vec is not None:
                frame_buffer.append(normalize_frame(lm_vec))
            else:
                frame_buffer.append(np.zeros(FEATURE_DIM, dtype=np.float32))

            frame_count += 1

            # Run model inference every INFERENCE_STRIDE frames
            if frame_count % INFERENCE_STRIDE == 0 and len(frame_buffer) >= MIN_FRAMES:
                seq = np.stack(list(frame_buffer), axis=0)   # (T, 63)
                seq_t  = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, T, 63)
                length = torch.tensor([seq.shape[0]])

                t_inf = time.time()
                with torch.no_grad():
                    log_probs, out_len = model(seq_t, length)
                    preds = ctc_greedy_decode(log_probs, out_len)
                inf_latency = (time.time() - t_inf) * 1000  # ms

                new_pred = preds[0].strip() if preds else ""
                current_pred = new_pred

                # Simple accumulation: append non-empty unique predictions
                # with a space separator (naïve word boundary heuristic)
                last_word = full_text.split()[-1] if full_text.split() else ""
                if new_pred and new_pred != last_word:
                    if full_text and not full_text.endswith(" "):
                        full_text += " "
                    full_text += new_pred

        # Draw UI
        state = {
            "model_name":    model_name,
            "fps":           fps,
            "hand_detected": hand_detected,
            "current_pred":  current_pred,
            "full_text":     full_text,
            "buffer":        frame_buffer,
            "inf_ms":        inf_latency,
            "paused":        paused,
        }
        draw_ui(frame, state)

        cv2.imshow("ISL Fingerspelling Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            full_text    = ""
            current_pred = ""
            frame_buffer.clear()
            print("[webcam] Text cleared.")
        elif key == ord("p"):
            paused = not paused
            print(f"[webcam] {'Paused' if paused else 'Resumed'}")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print(f"[webcam] Session ended.  Final recognized text: '{full_text}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ISL Fingerspelling Demo")
    parser.add_argument("--model",  type=str, default="conformer",
                        choices=["bilstm", "conformer"])
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (0=default webcam)")
    args = parser.parse_args()
    run_webcam(args.model, args.camera)
