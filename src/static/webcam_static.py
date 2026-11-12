"""
src/static/webcam_static.py
=============================
Real-time ISL static fingerspelling recognition using the trained DH-GAT model.

FEATURES:
  - Detects BOTH hands simultaneously (ISL is two-handed)
  - Skeleton overlay drawn on both hands
  - Stable prediction: only displays a letter after it is consistently
    predicted for `stable_frames` consecutive frames (avoids flickering)
  - Confidence bar shown below predicted letter
  - TTS (Text-to-Speech): speaks each confirmed letter via pyttsx3
  - Press Q to quit, SPACE to clear text, S to speak full history

Run:
  python src/static/webcam_static.py

Install deps if needed:
  pip install pyttsx3
"""

import argparse
import sys
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

# ── TTS setup (non-blocking via background thread) ──────────────────────────
try:
    import pyttsx3 as _pyttsx3
    _tts_engine = _pyttsx3.init()

    # Pick the best available voice (prefer Zira/Hazel/female voices on Windows)
    _voices = _tts_engine.getProperty('voices')
    _preferred_voice = None
    for _v in _voices:
        _vname = _v.name.lower()
        if any(k in _vname for k in ('zira', 'hazel', 'susan', 'female', 'eva', 'helen')):
            _preferred_voice = _v.id
            break
    if _preferred_voice:
        _tts_engine.setProperty('voice', _preferred_voice)
        print(f"[TTS] Voice selected: {_preferred_voice}")

    _tts_engine.setProperty('rate', 130)    # slower = clearer for fingerspelling
    _tts_engine.setProperty('volume', 1.0)
    TTS_AVAILABLE = True
except Exception as _e:
    TTS_AVAILABLE = False
    print(f"[TTS] pyttsx3 not available ({_e}). Install with: pip install pyttsx3")

_tts_lock = threading.Lock()
_tts_busy  = threading.Event()   # set while TTS is speaking


def _speak(text: str):
    """
    Speak text in a background thread.
    Drops the utterance if TTS is still busy (prevents queuing up stale letters).
    """
    if not TTS_AVAILABLE or not text.strip():
        return
    if _tts_busy.is_set():   # still speaking previous — skip, don't queue
        return
    def _run():
        _tts_busy.set()
        with _tts_lock:
            try:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            except Exception:
                pass
        _tts_busy.clear()
    threading.Thread(target=_run, daemon=True).start()


def _speak_word(text: str):
    """
    Speak a full word — always waits for previous speech to finish.
    Used for speaking accumulated words/sentences (S key, word boundary).
    """
    if not TTS_AVAILABLE or not text.strip():
        return
    def _run():
        with _tts_lock:
            try:
                _tts_engine.say(text)
                _tts_engine.runAndWait()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    StaticConfig, NUM_LANDMARKS, LANDMARK_DIM,
    MP_MAX_NUM_HANDS_STATIC, IDX_TO_LETTER,
    HAND_EDGES, ROOT
)
from src.models.dh_gat import build_dhgat
from src.preprocessing.normalizer import normalize_hand_static
from src.preprocessing.feature_engineer import compute_node_features, compute_global_distances

TASK_MODEL_PATH = ROOT / "hand_landmarker.task"
CHECKPOINT_PATH = ROOT / "checkpoints" / "dh_gat_static_best.pt"


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────────────────────────────────────────
def _build_detector():
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    base_options = mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=MP_MAX_NUM_HANDS_STATIC,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
HAND_COLORS = {
    "Right": (0, 220, 100),    # green for right hand
    "Left":  (100, 180, 255),  # blue for left hand
}

def _draw_hand(frame: np.ndarray, lm_raw: list, color: tuple):
    """Draw skeleton on frame using raw (unnormalized) MediaPipe landmark objects."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_raw]
    for i, j in HAND_EDGES:
        cv2.line(frame, pts[i], pts[j], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hand_flipped(frame: np.ndarray, lm_raw: list, color: tuple):
    """
    Draw skeleton on a HORIZONTALLY FLIPPED display frame.
    MediaPipe landmark x coords are in [0,1] for the original (unflipped) frame.
    Since we flip the display frame, we must mirror x: x_display = 1 - x_original.
    """
    h, w = frame.shape[:2]
    pts = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in lm_raw]
    for i, j in HAND_EDGES:
        cv2.line(frame, pts[i], pts[j], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)




def _draw_ui(frame: np.ndarray, letter: str, confidence: float,
             right_det: bool, left_det: bool, stable: bool,
             history_text: str, fps: float, tts_on: bool = True,
             top_preds: list = None):
    """
    top_preds: list of (letter, conf) sorted by confidence descending, e.g.
               [('A', 0.92), ('D', 0.05), ('B', 0.02)]
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar (taller to fit runner-up)
    cv2.rectangle(overlay, (0, 0), (w, 105), (20, 20, 30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Hand detection indicators
    r_col = (0, 220, 100) if right_det else (80, 80, 80)
    l_col = (100, 180, 255) if left_det else (80, 80, 80)
    cv2.circle(frame, (20, 20), 8, r_col, -1)
    cv2.putText(frame, "R", (32, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_col, 1)
    cv2.circle(frame, (65, 20), 8, l_col, -1)
    cv2.putText(frame, "L", (77, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_col, 1)

    # Current predicted letter (big, center top)
    if letter and stable:
        conf_pct = int(confidence * 100)
        txt_color = (0, 255, 140) if confidence > 0.85 else (255, 200, 60)
        cv2.putText(frame, letter, (w//2 - 30, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, txt_color, 3, cv2.LINE_AA)
        # Confidence bar
        bar_w = int(200 * confidence)
        cv2.rectangle(frame, (w//2 - 100, 88), (w//2 - 100 + 200, 95), (60, 60, 60), -1)
        cv2.rectangle(frame, (w//2 - 100, 88), (w//2 - 100 + bar_w, 95), txt_color, -1)
        cv2.putText(frame, f"{conf_pct}%", (w//2 + 105, 97),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)

        # Runner-up predictions (top 2nd and 3rd)
        if top_preds and len(top_preds) >= 2:
            runner_up_parts = []
            for ltr, conf in top_preds[1:3]:
                runner_up_parts.append(f"{ltr} {int(conf*100)}%")
            runner_txt = "  |  ".join(runner_up_parts)
            cv2.putText(frame, runner_txt, (w//2 - 60, 103),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
    elif letter:
        # Detected but not yet stable — show dimmer with live confidence
        conf_pct = int(confidence * 100)
        cv2.putText(frame, letter, (w//2 - 20, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (120, 120, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{conf_pct}%", (w//2 + 25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # History text at bottom
    if history_text:
        cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 30), -1)
        cv2.putText(frame, history_text[-60:], (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # TTS status indicator
    tts_label = "TTS: ON" if (tts_on and TTS_AVAILABLE) else ("TTS: OFF" if TTS_AVAILABLE else "TTS: N/A")
    tts_col   = (0, 200, 100) if (tts_on and TTS_AVAILABLE) else (100, 100, 100)
    cv2.putText(frame, tts_label, (w - 90, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, tts_col, 1)

    # Controls hint
    cv2.putText(frame, "Q:quit  SPC:clear  S:speak  T:TTS toggle", (10, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main webcam loop
# ─────────────────────────────────────────────────────────────────────────────
def run_webcam(checkpoint_path: Path = None):
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    if not TASK_MODEL_PATH.exists():
        print(f"[webcam_static] ERROR: {TASK_MODEL_PATH} not found.")
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"[webcam_static] ERROR: No trained model at {checkpoint_path}")
        print("  Run: python src/static/train_static.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = StaticConfig()

    # Load model
    model = build_dhgat(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    val_acc = ckpt.get("best_val_acc", 0.0)
    print(f"[webcam_static] Loaded DH-GAT (val acc={100*val_acc:.2f}%)")

    detector = _build_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[webcam_static] ERROR: Cannot open camera 0")
        sys.exit(1)
    print("[webcam_static] Camera opened. Press Q to quit, SPACE to clear.")

    # State
    pred_history       = deque(maxlen=cfg.stable_frames)
    history_text       = ""
    current_letter     = ""
    current_conf       = 0.0
    current_top_preds  = []
    stable             = False
    last_stable_letter = ""
    frame_count        = 0
    fps_t0             = time.time()
    fps                = 0.0
    tts_enabled        = True
    # Word-boundary TTS: speak accumulated letters as a word after WORD_PAUSE_SECS
    WORD_PAUSE_SECS    = 2.0   # speak as word after 2s of no new letter
    _last_letter_time  = 0.0   # timestamp of last confirmed letter
    _word_spoken       = True  # flag: have we spoken the current buffer?

    if TTS_AVAILABLE:
        print("[TTS] TTS enabled (word-boundary mode). T=toggle, S=speak, W=speak word.")
    else:
        print("[TTS] Not available — install pyttsx3 for speech output.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # FPS calculation
        if frame_count % 30 == 0:
            fps   = 30 / (time.time() - fps_t0)
            fps_t0 = time.time()

        # ─── CRITICAL: detect on UNFLIPPED frame ───
        # When you flip first, MediaPipe sees mirror-images of hands and
        # swaps the handedness labels (Right ↔ Left). Training data was
        # extracted from unflipped images, so we must match that here.
        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = detector.detect(mp_image)

        # Now flip the display frame so the user sees themselves as a mirror
        frame = cv2.flip(frame, 1)

        right_lm_norm  = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        left_lm_norm   = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        right_present  = False
        left_present   = False

        if result.hand_landmarks:
            for i, handedness_list in enumerate(result.handedness):
                label = handedness_list[0].display_name  # 'Right' or 'Left'
                lms   = result.hand_landmarks[i]
                arr   = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                arr_n = normalize_hand_static(arr)

                # Mirror the x coordinate for drawing (since frame is now flipped)
                # We clone lm objects-like approach: just flip x in pixel space
                _draw_hand_flipped(frame, lms, HAND_COLORS[label])

                if label == "Right":
                    right_lm_norm = arr_n
                    right_present = True
                else:
                    left_lm_norm  = arr_n
                    left_present  = True


        # Run model inference on every frame (DH-GAT is fast on CPU)
        if right_present or left_present:
            # Compute enriched features for the model
            right_node = compute_node_features(right_lm_norm)     # (21, 6)
            left_node  = compute_node_features(left_lm_norm)      # (21, 6)
            right_geo  = compute_global_distances(right_lm_norm)  # (10,)
            left_geo   = compute_global_distances(left_lm_norm)   # (10,)

            with torch.no_grad():
                rn = torch.from_numpy(right_node).unsqueeze(0).to(device)
                ln = torch.from_numpy(left_node).unsqueeze(0).to(device)
                rg = torch.from_numpy(right_geo).unsqueeze(0).to(device)
                lg = torch.from_numpy(left_geo).unsqueeze(0).to(device)
                rp = torch.tensor([right_present], dtype=torch.bool).to(device)
                lp = torch.tensor([left_present],  dtype=torch.bool).to(device)

                logits = model(rn, ln, rp, lp, rg, lg)
                probs  = torch.softmax(logits, dim=1)[0]
                top_confs, top_idxs = probs.topk(3)
                pred_letter = IDX_TO_LETTER[top_idxs[0].item()]
                pred_conf   = top_confs[0].item()
                top_preds   = [(IDX_TO_LETTER[top_idxs[i].item()], top_confs[i].item())
                               for i in range(3)]

            current_letter    = pred_letter
            current_conf      = pred_conf
            current_top_preds = top_preds
            pred_history.append(pred_letter)

            # Adaptive stability window: commit faster when confidence is high
            if pred_conf >= 0.95:
                required_frames = 5
            elif pred_conf >= 0.85:
                required_frames = 8
            else:
                required_frames = cfg.stable_frames   # 10 (default)

            # Stability check: last N frames must all agree AND confidence above threshold
            recent = list(pred_history)[-required_frames:]
            if (len(recent) == required_frames and
                    len(set(recent)) == 1 and
                    pred_conf >= cfg.min_confidence):
                stable = True
                if pred_letter != last_stable_letter:
                    history_text += pred_letter
                    last_stable_letter = pred_letter
                    _last_letter_time  = time.time()
                    _word_spoken       = False
                    # TTS: speak each confirmed letter immediately (brief feedback)
                    if tts_enabled:
                        _speak(pred_letter)
            else:
                stable = False

        else:
            pred_history.clear()
            current_letter    = ""
            current_conf      = 0.0
            current_top_preds = []
            stable            = False
            last_stable_letter = ""

        # ── Word-boundary TTS: if letters stopped coming, speak accumulated word ──
        if (tts_enabled and not _word_spoken
                and history_text.strip()
                and time.time() - _last_letter_time >= WORD_PAUSE_SECS):
            print(f"[TTS] Word boundary — speaking: '{history_text}'")
            _speak_word(history_text)
            _word_spoken = True

        _draw_ui(frame, current_letter, current_conf,
                 right_present, left_present, stable,
                 history_text, fps, tts_on=tts_enabled,
                 top_preds=current_top_preds)

        cv2.imshow("ISL Static Fingerspelling — DH-GAT", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            history_text       = ""
            last_stable_letter = ""
            _word_spoken       = True
            print("[webcam_static] Text cleared")
        elif key in (ord("s"), ord("S")):
            # Speak full history text
            if history_text.strip():
                print(f"[TTS] Speaking: '{history_text}'")
                _speak_word(history_text)
                _word_spoken = True
            else:
                print("[TTS] Nothing to speak yet.")
        elif key in (ord("w"), ord("W")):
            # Speak as word (natural pronunciation), then clear
            if history_text.strip():
                print(f"[TTS] Word: '{history_text}'")
                _speak_word(history_text)
                history_text       = ""
                last_stable_letter = ""
                _word_spoken       = True
        elif key in (ord("t"), ord("T")):
            tts_enabled = not tts_enabled
            _word_spoken = True  # don't auto-speak on toggle
            print(f"[TTS] Toggled {'ON' if tts_enabled else 'OFF'}")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print(f"[webcam_static] Session ended. Text: '{history_text}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Static Fingerspelling Webcam Demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: dh_gat_static_best.pt)")
    args = parser.parse_args()
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    run_webcam(checkpoint_path=ckpt)
