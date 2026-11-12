"""
src/pipeline/overlay.py
=======================
Draws the recognition HUD ON TOP of a video frame: skeleton, big letter with a confidence
ring/bar, top-3 alternatives, and the text typed so far. Used for annotated video export
and the OpenCV webcam window (the web app draws the same HUD on a <canvas>).
"""

import cv2
import numpy as np

from src.pipeline.hands import draw_skeleton

FONT = cv2.FONT_HERSHEY_DUPLEX
STATUS_COL = {"no_hand": (150, 150, 150), "moving": (60, 180, 255),
              "holding": (255, 200, 60), "committed": (90, 220, 120)}


def conf_color(c: float):
    """red -> amber -> green (BGR)."""
    if c < 0.5:
        return (70, 70, 230)
    if c < 0.8:
        return (50, 190, 250)
    return (100, 220, 100)


def _panel(img, x, y, w, h, alpha=0.55, color=(18, 18, 24)):
    sub = img[y:y + h, x:x + w]
    if sub.size == 0:
        return
    cv2.addWeighted(np.full_like(sub, color), alpha, sub, 1 - alpha, 0, sub)


def draw_hud(frame: np.ndarray, rec: dict, state: dict, show_skeleton: bool = True) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max(w, h) / 960.0
    if show_skeleton and rec is not None:
        draw_skeleton(frame, rec, thickness=max(1, int(2 * s)))

    # --- big letter card (top-left) ---
    cw, ch = int(210 * s), int(192 * s)
    _panel(frame, int(14 * s), int(14 * s), cw, ch)
    letter, conf, status = state.get("letter"), state.get("conf", 0.0), state.get("status", "no_hand")
    col = STATUS_COL.get(status, (200, 200, 200))
    cv2.rectangle(frame, (int(14 * s), int(14 * s)), (int(14 * s) + int(6 * s), int(14 * s) + ch), col, -1)
    if letter and status != "no_hand":
        (tw, th), _ = cv2.getTextSize(letter, FONT, 4.2 * s, int(5 * s))
        cv2.putText(frame, letter, (int(40 * s), int(14 * s) + int(110 * s)), FONT, 4.2 * s,
                    (255, 255, 255), max(2, int(5 * s)), cv2.LINE_AA)
        # confidence bar + percentage
        bx, by, bw, bh = int(34 * s), int(14 * s) + int(128 * s), int(170 * s), int(10 * s)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 70), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * conf), by + bh), conf_color(conf), -1)
        cv2.putText(frame, f"{conf*100:.0f}%", (bx, by + int(26 * s)), FONT, 0.6 * s,
                    (235, 235, 235), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "show a sign", (int(30 * s), int(14 * s) + int(90 * s)), FONT, 0.8 * s,
                    (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, status.upper(), (int(110 * s), int(14 * s) + int(ch) - int(10 * s)), FONT, 0.5 * s,
                col, 1, cv2.LINE_AA)

    # --- top-3 chips ---
    for i, (lt, p) in enumerate(state.get("top3", [])[:3]):
        x0 = int(14 * s) + cw + int(10 * s)
        y0 = int(14 * s) + i * int(34 * s)
        _panel(frame, x0, y0, int(120 * s), int(28 * s))
        cv2.putText(frame, f"{lt}  {p*100:4.0f}%", (x0 + int(8 * s), y0 + int(20 * s)), FONT, 0.55 * s,
                    conf_color(p), 1, cv2.LINE_AA)

    # --- text banner (bottom) ---
    text = state.get("text", "")
    bh_ = int(70 * s)
    _panel(frame, 0, h - bh_, w, bh_, alpha=0.65)
    shown = text[-34:] if text else "..."
    cv2.putText(frame, shown + ("_" if text else ""), (int(20 * s), h - int(22 * s)), FONT, 1.5 * s,
                (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(frame, "TEXT", (int(20 * s), h - bh_ + int(18 * s)), FONT, 0.45 * s, (160, 160, 170), 1, cv2.LINE_AA)
    return frame
