"""
src/static/text_to_sign_avatar.py
===================================
Text → ISL Sign Avatar  (Vice-Versa Direction of the pipeline)

PIPELINE (Direction 2):
  Typed text / Speech input
      ↓
  Split into characters (A-Z)
      ↓
  For each letter: load reference MediaPipe landmarks from cache
      (the same landmark data the DH-GAT model was trained on)
      ↓
  Draw animated hand skeleton using OpenCV (stick figure avatar)
      ↓
  Optional TTS: speak each letter/word as it is displayed

WHY USE CACHED LANDMARKS FOR THE AVATAR?
  The DH-GAT model was trained on these exact landmark distributions.
  Using the same data for the avatar makes the system self-consistent:
  "what the model recognizes" = "what the avatar shows".

CONTROLS (interactive mode):
  Type letters → Press ENTER to animate
  S  : speak the current word via TTS
  Q  : quit

Run:
  python src/static/text_to_sign_avatar.py                # interactive
  python src/static/text_to_sign_avatar.py --word HELLO   # single word
  python src/static/text_to_sign_avatar.py --demo         # A-Z auto demo
"""

import argparse
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    STATIC_LETTERS, STATIC_CACHE_DIR, HAND_EDGES
)

# ─── TTS (optional, non-blocking) ────────────────────────────────────────────
try:
    import pyttsx3 as _pyttsx3
    _tts_engine = _pyttsx3.init()

    # Pick the best available voice (prefer female/natural-sounding)
    _voices = _tts_engine.getProperty('voices')
    _preferred = None
    for v in _voices:
        name = v.name.lower()
        if any(k in name for k in ('zira', 'hazel', 'susan', 'female', 'eva', 'helen')):
            _preferred = v.id
            break
    if _preferred:
        _tts_engine.setProperty('voice', _preferred)

    _tts_engine.setProperty('rate', 130)    # slower = clearer for fingerspelling
    _tts_engine.setProperty('volume', 1.0)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

_tts_lock = threading.Lock()
_tts_queue: list = []          # queued utterances


def _speak(text: str, slow: bool = False):
    """
    Speak text in a background thread.
    If `slow=True`, inserts a short pause between each character
    (useful for spelling out letter names clearly).
    Never blocks the display loop.
    """
    if not TTS_AVAILABLE or not text.strip():
        return

    # Build the utterance: if slow, space the characters out
    if slow and len(text) == 1:
        # Single letter — say the letter name at a slower pace
        utterance = text
    elif slow:
        # Word spelled out: "H... E... L... L... O"
        utterance = ",  ".join(list(text.upper()))  # comma pause between letters
    else:
        utterance = text

    def _run():
        with _tts_lock:
            try:
                _tts_engine.say(utterance)
                _tts_engine.runAndWait()
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()


# ─── Per-letter hand usage (derived from 200-sample statistics per letter) ─────
# Based on actual dataset: % of samples where each hand was detected.
# In ISL fingerspelling: D, P, Z, R are two-handed signs requiring both hands.
# 'R' = right hand only, 'L' = left hand only, 'B' = both hands
HAND_MODE = {
    'A': 'B', 'B': 'B', 'C': 'L', 'D': 'B', 'E': 'B',
    'F': 'B', 'G': 'L', 'H': 'R', 'I': 'L', 'J': 'L',
    'K': 'B', 'L': 'L', 'M': 'B', 'N': 'B', 'O': 'R',
    'P': 'B', 'Q': 'B', 'R': 'B', 'S': 'B', 'T': 'B',
    'U': 'R', 'V': 'R', 'W': 'B', 'X': 'B', 'Y': 'B',
    'Z': 'B',
}



# ─── Load reference landmarks ─────────────────────────────────────────────────

def _medoid(samples: list) -> np.ndarray:
    """Return the sample closest to the centroid (most representative real pose)."""
    arr = np.stack(samples)
    centroid = arr.mean(axis=0)
    dists = np.linalg.norm((arr - centroid).reshape(len(arr), -1), axis=1)
    return arr[np.argmin(dists)]


def _best_by_model(samples: list, target_letter: str,
                   model, device: str = "cpu") -> np.ndarray:
    """
    Pick the sample that DH-GAT classifies as target_letter with highest confidence.
    This guarantees the reference pose is one the model recognises correctly.
    Falls back to medoid if no sample is classified correctly.
    """
    import torch
    from src.training.config import STATIC_LETTERS
    l2i = {l: i for i, l in enumerate(STATIC_LETTERS)}
    target_idx = l2i.get(target_letter, -1)
    if target_idx < 0 or model is None:
        return _medoid(samples)

    best_conf  = -1.0
    best_sample = None
    arr = np.stack(samples)   # (N, 21, 3)

    # batch inference
    t = torch.from_numpy(arr).unsqueeze(0)   # wrong shape — process individually
    model.eval()
    with torch.no_grad():
        for lm in arr:
            r = torch.from_numpy(lm).unsqueeze(0).to(device)
            z = torch.zeros_like(r)
            rp = torch.tensor([True],  dtype=torch.bool).to(device)
            lp = torch.tensor([False], dtype=torch.bool).to(device)
            logits = model(r, z, rp, lp)
            probs  = torch.softmax(logits, dim=1)[0]
            conf   = probs[target_idx].item()
            if conf > best_conf:
                best_conf   = conf
                best_sample = lm

    if best_conf < 0.3:          # model never confident → fall back to medoid
        return _medoid(samples)
    return best_sample


_DHGAT_MODEL = None   # cached model reference for use in load_reference_landmarks


CURATED_DIR = ROOT / "cache" / "static_curated"


def _load_curated(letter: str, hand: str):
    """Return (21,3) array from curated dir, or None if not saved yet."""
    path = CURATED_DIR / f"{letter}_{hand}.npz"
    if path.exists():
        try:
            return np.load(str(path))["lm"].astype(np.float32)
        except Exception:
            pass
    return None


def load_reference_landmarks(model=None, device: str = "cpu") -> dict:
    """
    Load reference landmark poses per letter from the cache.

    Priority order per hand:
      1. Curated pose from cache/static_curated/<LETTER>_<hand>.npz
         (manually adjusted via landmark_editor.py — highest quality)
      2. Model-guided: pick sample DH-GAT classifies with highest confidence
         (ensures avatar shows a sign the model actually recognises)
      3. Geometric medoid: closest sample to the per-class centroid (default)

    Returns dict: letter -> {'right_lm': (21,3) or None,
                             'left_lm':  (21,3) or None,
                             'right_present': bool, 'left_present': bool, 'mode': str}
    """
    global _DHGAT_MODEL
    if model is not None:
        _DHGAT_MODEL = model

    MAX_SAMPLES = 200
    refs = {}
    n_curated = 0

    for letter in STATIC_LETTERS:
        # Check curated poses first (manually adjusted)
        best_r = _load_curated(letter, "right")
        best_l = _load_curated(letter, "left")
        if best_r is not None:
            n_curated += 1

        letter_dir = STATIC_CACHE_DIR / letter
        right_samples, left_samples = [], []

        if letter_dir.exists() and (best_r is None or best_l is None):
            for npz_path in sorted(letter_dir.glob("*.npz")):
                if len(right_samples) >= MAX_SAMPLES and len(left_samples) >= MAX_SAMPLES:
                    break
                try:
                    d = np.load(str(npz_path))
                    rp = bool(d['right_present'])
                    lp = bool(d['left_present'])
                    if rp and best_r is None and len(right_samples) < MAX_SAMPLES:
                        rlm = d['right_lm'].astype(np.float32)
                        if rlm.shape == (21, 3) and not np.all(rlm == 0):
                            if 0.5 < np.linalg.norm(rlm[9]) < 2.0:
                                right_samples.append(rlm)
                    if lp and best_l is None and len(left_samples) < MAX_SAMPLES:
                        llm = d['left_lm'].astype(np.float32)
                        if llm.shape == (21, 3) and not np.all(llm == 0):
                            if 0.5 < np.linalg.norm(llm[9]) < 2.0:
                                left_samples.append(llm)
                except Exception:
                    continue

        # Fill in any non-curated hand using model-guided or medoid
        if best_r is None:
            if _DHGAT_MODEL is not None and right_samples:
                best_r = _best_by_model(right_samples, letter, _DHGAT_MODEL, device)
            elif right_samples:
                best_r = _medoid(right_samples)

        if best_l is None:
            best_l = _medoid(left_samples) if left_samples else None

        entry = {
            'right_lm':      best_r,
            'left_lm':       best_l,
            'right_present': best_r is not None,
            'left_present':  best_l is not None,
            'mode':          HAND_MODE.get(letter, 'B'),
        }
        refs[letter] = entry

    loaded = sum(1 for e in refs.values() if e['right_present'] or e['left_present'])
    print(f"[avatar] Loaded {loaded}/26 letters  "
          f"(R+L: {sum(1 for e in refs.values() if e['right_present'] and e['left_present'])}, "
          f"R-only: {sum(1 for e in refs.values() if e['right_present'] and not e['left_present'])}, "
          f"L-only: {sum(1 for e in refs.values() if not e['right_present'] and e['left_present'])}, "
          f"curated: {n_curated})")
    return refs



# ─── Drawing helpers ──────────────────────────────────────────────────────────

CANVAS_W   = 800
CANVAS_H   = 540
R_WRIST_X  = CANVAS_W // 2 - 160
L_WRIST_X  = CANVAS_W // 2 + 160
WRIST_Y    = CANVAS_H // 2 + 90
SCALE      = 175
AR_CORRECT = 0.75

BG         = (15, 15, 25)
FINGER_COLORS = [
    (60, 160, 255),   # thumb  — blue
    (255, 160, 40),   # index  — orange
    (60, 220, 80),    # middle — green
    (200, 80, 255),   # ring   — purple
    (255, 80, 120),   # pinky  — pink
]
PALM_COL   = (160, 160, 160)
JOINT_COL  = (240, 240, 240)
TIPS       = {4, 8, 12, 16, 20}
ANIM_FPS   = 30

LM_FINGER = [
    0, 1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
]
PALM_EDGES = {(0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)}


def _to_canvas(lm: np.ndarray, anchor_x: int, entry_scale: float = 1.0) -> list:
    """
    Convert normalized (21,3) hand landmarks -> pixel coords on avatar canvas.
    Wrist is placed at (anchor_x, WRIST_Y). Fingers extend upward (negative Y).
    """
    pts = []
    for x, y, _ in lm:
        px = int(np.clip(
            anchor_x + x * AR_CORRECT * SCALE * entry_scale,
            5, CANVAS_W - 5))
        py = int(np.clip(
            WRIST_Y + y * SCALE * entry_scale,
            5, CANVAS_H - 5))
        pts.append((px, py))
    return pts


def _draw_skinned_hand(canvas, pts, alpha: float = 1.0, mirror: bool = False, is_right: bool = True):
    """
    Draw a volumetric 'skinned' hand with natural skin tones for the avatar.
    """
    def _blend(base_bgr, a):
        return tuple(max(0, min(255, int(c * a))) for c in base_bgr)

    # Natural skin tones (BGR format)
    # Warm, neutral medium skin tone
    skin_col = _blend((140, 180, 230), alpha)       # Base skin 
    highlight_col = _blend((170, 200, 245), alpha)  # Highlights for 3D effect
    joint_col = _blend((110, 150, 210), alpha)      # Slightly darker for knuckles / shadows

    # 1. Draw the Palm (Filled Polygon)
    palm_indices = [0, 1, 5, 9, 13, 17]
    palm_pts = np.array([pts[i] for i in palm_indices], dtype=np.int32)
    
    # Outer stroke (shadow/edge)
    cv2.polylines(canvas, [palm_pts], True, joint_col, 8, cv2.LINE_AA)
    # Inner fill
    cv2.fillPoly(canvas, [palm_pts], skin_col)
    
    # 2. Draw the Fingers (Thick Capsules)
    for i, j in HAND_EDGES:
        if (min(i,j), max(i,j)) in PALM_EDGES:
            continue
            
        pt1, pt2 = pts[i], pts[j]
        # Base thick capsule (Shadow/Edge)
        cv2.line(canvas, pt1, pt2, joint_col, 16, cv2.LINE_AA)
        # Inner fill (Main skin)
        cv2.line(canvas, pt1, pt2, skin_col, 12, cv2.LINE_AA)
        # Core highlight for 3D effect
        cv2.line(canvas, pt1, pt2, highlight_col, 4, cv2.LINE_AA)

    # 3. Draw Joints (Knuckles)
    for idx, pt in enumerate(pts):
        if idx == 0:  # Wrist
            r, c = 12, joint_col
        elif idx in TIPS:  # Fingertips
            r, c = 8, skin_col
        else:
            r, c = 9, joint_col
            
        cv2.circle(canvas, pt, r, c, -1, cv2.LINE_AA)
        # Knuckle highlight
        cv2.circle(canvas, pt, r - 3, skin_col, -1, cv2.LINE_AA)


def _draw_ui(canvas, letter, word, li, total, typed, mode="animate",
             right_present=True, left_present=True):
    h, w = canvas.shape[:2]
    ov = canvas.copy()
    cv2.rectangle(ov, (0, 0), (w, 85), (20, 20, 35), -1)
    canvas[:] = cv2.addWeighted(ov, 0.88, canvas, 0.12, 0)
    cv2.putText(canvas, "ISL Sign Avatar  |  Text to Sign",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 140, 190), 1)
    if letter:
        cv2.putText(canvas, letter, (w - 80, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 235, 115), 3, cv2.LINE_AA)
    if total > 0:
        cv2.putText(canvas, f"{li + 1} / {total}", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
    # Hand presence R / L dots
    r_col = (0, 220, 100) if right_present else (55, 55, 55)
    l_col = (100, 180, 255) if left_present else (55, 55, 55)
    cv2.circle(canvas, (12, 70), 6, r_col, -1)
    cv2.putText(canvas, "R", (22, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, r_col, 1)
    cv2.circle(canvas, (48, 70), 6, l_col, -1)
    cv2.putText(canvas, "L", (58, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, l_col, 1)
    if word:
        char_w = 26
        x0 = max(12, w // 2 - len(word) * char_w // 2)
        for ci, ch in enumerate(word):
            if ci == li:
                col, thick = (0, 235, 115), 2
                cv2.line(canvas, (x0 + ci*char_w, 80), (x0 + ci*char_w + char_w-4, 80), col, 2)
            else:
                col, thick = (100, 100, 100), 1
            cv2.putText(canvas, ch, (x0 + ci * char_w, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, thick)
    names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    for fi, (name, col) in enumerate(zip(names, FINGER_COLORS)):
        lx = 12 + fi * 80
        if lx + 70 < w:
            cv2.circle(canvas, (lx, h - 18), 5, col, -1)
            cv2.putText(canvas, name, (lx + 9, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
    ov2 = canvas.copy()
    cv2.rectangle(ov2, (0, h - 55), (w, h), (20, 20, 35), -1)
    canvas[:] = cv2.addWeighted(ov2, 0.88, canvas, 0.12, 0)
    tts_col = (0, 190, 80) if TTS_AVAILABLE else (80, 80, 80)
    if mode == "input":
        cv2.putText(canvas, f"Type: {typed}_",
                    (12, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (210, 210, 210), 1)
        cv2.putText(canvas, "ENTER: sign   BACKSPACE: delete   Q: quit",
                    (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (120, 120, 120), 1)
    else:
        cv2.putText(canvas, f"TTS: {'ON' if TTS_AVAILABLE else 'N/A'}", (w-95, h-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, tts_col, 1)
        cv2.putText(canvas, "SPACE: new word   S: speak   Q: quit",
                    (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (120, 120, 120), 1)



# ─── Animation ────────────────────────────────────────────────────────────────

HOLD_SECS   = 1.2
TRANS_SECS  = 0.2




def animate_word(refs: dict, word: str, speak: bool = True) -> bool:
    """
    Animate ISL fingerspelling for `word`.
    Each letter uses its data-driven hand mode (R / L / BOTH).
    Returns False if user pressed Q.
    """
    letters = [ch.upper() for ch in word if ch.upper() in STATIC_LETTERS]
    if not letters:
        return True

    print(f"[avatar] Signing: {' '.join(letters)}")
    if speak:
        # Speak the whole word naturally first
        _speak(word)

    n_trans = max(1, int(TRANS_SECS * ANIM_FPS))
    n_hold  = max(1, int(HOLD_SECS  * ANIM_FPS))
    delay   = int(1000 / ANIM_FPS)
    WIN     = "ISL Sign Avatar  |  Text to Sign"

    for li, letter in enumerate(letters):
        entry = refs.get(letter)
        if entry is None:
            continue

        mode = entry.get('mode', 'B')      # 'R', 'L', or 'B'
        rlm  = entry['right_lm']
        llm  = entry['left_lm']

        # ── Determine anchor positions based on mode ──────────────────────────
        if mode == 'B' and rlm is not None and llm is not None:
            # Both hands side by side
            r_anchor, l_anchor = R_WRIST_X, L_WRIST_X
            show_r, show_l = True, True
        elif mode == 'R' and rlm is not None:
            # Right hand only — center it
            r_anchor, l_anchor = CANVAS_W // 2, None
            show_r, show_l = True, False
        elif mode == 'L' and llm is not None:
            # Left hand only — center it
            r_anchor, l_anchor = None, CANVAS_W // 2
            show_r, show_l = False, True
        else:
            # Fallback: show whichever is available, centered
            if rlm is not None:
                r_anchor, l_anchor = CANVAS_W // 2, None
                show_r, show_l = True, False
            else:
                r_anchor, l_anchor = None, CANVAS_W // 2
                show_r, show_l = False, True

        # ── Entry animation (scale 0 → 1) ─────────────────────────────────────
        for fi in range(n_trans):
            s = (fi + 1) / n_trans
            canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
            if show_r:
                _draw_skinned_hand(canvas, _to_canvas(rlm, r_anchor, s), alpha=s, is_right=True)
            if show_l:
                _draw_skinned_hand(canvas, _to_canvas(llm, l_anchor, s), alpha=s, mirror=True, is_right=False)
            _draw_ui(canvas, letter, word, li, len(letters), "",
                     right_present=show_r, left_present=show_l)
            cv2.imshow(WIN, canvas)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                return False

        # ── Hold frame ────────────────────────────────────────────────────────
        r_pts = _to_canvas(rlm, r_anchor) if show_r else None
        l_pts = _to_canvas(llm, l_anchor) if show_l else None
        for _ in range(n_hold):
            canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
            if r_pts: _draw_skinned_hand(canvas, r_pts, is_right=True)
            if l_pts: _draw_skinned_hand(canvas, l_pts, mirror=True, is_right=False)
            _draw_ui(canvas, letter, word, li, len(letters), "",
                     right_present=show_r, left_present=show_l)
            cv2.imshow(WIN, canvas)
            k = cv2.waitKey(delay) & 0xFF
            if k == ord("q"):  return False
            if k in (ord("s"), ord("S")): _speak(letter, slow=True)

    return True


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_interactive(refs: dict):
    typed = ""
    WIN = "ISL Sign Avatar  |  Text to Sign"
    print("[avatar] Interactive: type a word then press ENTER.")
    while True:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
        # Ghost circles where each hand would appear
        for ax, lbl in [(R_WRIST_X, "Right"), (L_WRIST_X, "Left")]:
            for r, a in [(80, 18), (55, 30), (28, 48)]:
                ov = canvas.copy()
                cv2.circle(ov, (ax, WRIST_Y - 50), r, (0, 180, 90), -1)
                canvas[:] = cv2.addWeighted(ov, a/100, canvas, 1-a/100, 0)
            cv2.putText(canvas, lbl, (ax - 20, WRIST_Y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
        cv2.putText(canvas, "Type a word & press ENTER to see the ISL signs",
                    (CANVAS_W // 2 - 230, WRIST_Y - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1)
        _draw_ui(canvas, "", "", 0, 0, typed, mode="input")
        cv2.imshow(WIN, canvas)

        k = cv2.waitKey(50) & 0xFF
        if k in (ord("q"), 27):  break
        elif k == 13 and typed.strip():
            if not animate_word(refs, typed.strip()):
                break
            typed = ""
        elif k == 8:   typed = typed[:-1]
        elif 32 <= k <= 126: typed += chr(k)

    cv2.destroyAllWindows()


def run_demo(refs: dict):
    WIN = "ISL Sign Avatar  |  Text to Sign"
    for word in ["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXYZ", "HELLO", "ISL", "SIGN"]:
        print(f"[avatar] Demo: {word}")
        if not animate_word(refs, word, speak=len(word) <= 6):
            break
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
        cv2.putText(canvas, word,
                    (CANVAS_W // 2 - len(word) * 14, CANVAS_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2)
        cv2.imshow(WIN, canvas)
        if cv2.waitKey(1200) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ISL Text-to-Sign Avatar")
    parser.add_argument("--word",        type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--demo",        action="store_true",
                        help="Auto demo: letters A-Z then HELLO ISL SIGN")
    args = parser.parse_args()

    print("[avatar] Loading & averaging reference landmarks from cache...")
    refs = load_reference_landmarks()
    if not refs:
        print("[avatar] ERROR: No landmarks found. Run training pipeline first.")
        sys.exit(1)

    win = "ISL Sign Avatar  |  Text to Sign"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CANVAS_W, CANVAS_H)
    # Show window immediately so OS registers it before first imshow
    cv2.imshow(win, np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8))
    cv2.waitKey(1)

    if args.demo:
        run_demo(refs)
    elif args.word:
        animate_word(refs, args.word)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    else:
        run_interactive(refs)  # default
