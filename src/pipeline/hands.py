"""
src/pipeline/hands.py
=====================
Two-hand MediaPipe tracker used by the webcam, video and web-app pipelines.
Output format is identical to the training cache (src/static/extract_v2.py):
hands are ordered by wrist-x (image-left first), a lone hand sits in slot 0.
"""

import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TASK_MODEL = ROOT / "hand_landmarker.task"
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


class HandTracker:
    """Thread-safe wrapper around MediaPipe HandLandmarker (IMAGE mode, up to 2 hands)."""

    def __init__(self, min_conf: float = 0.4, recover: bool = False):
        """recover=True enables second-hand recovery (see src/static/extract_v2.py: _second_hand)."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        self.recover = recover
        self._lock = threading.Lock()
        self._det = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
                min_tracking_confidence=min_conf))

    def __call__(self, bgr: np.ndarray, flip: bool = False) -> dict:
        """
        bgr  : frame (H,W,3).  flip=True mirrors the frame first (use for selfie webcams
               when the model was trained on non-mirrored footage; the static v2 model is
               mirror-augmented so this is normally unnecessary).
        returns dict(lm (2,21,3) normalised image coords, present (2,) bool, wh (2,) int)
        """
        if flip:
            bgr = cv2.flip(bgr, 1)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        rec = {"lm": np.zeros((2, 21, 3), np.float32), "present": np.zeros(2, bool),
               "wh": np.array([w, h], np.int32), "score": np.zeros(2, np.float32)}
        dets = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            dets.append((np.array([[p.x, p.y, p.z] for p in lms], np.float32), float(hd[0].score)))
        if self.recover and len(dets) == 1:
            from src.static.extract_v2 import _second_hand
            with self._lock:
                extra = _second_hand(bgr, self._det, dets[0][0])
            if extra is not None:
                dets.append((extra[0], extra[2]))
        dets.sort(key=lambda d: d[0][0, 0])
        for i, (arr, sc) in enumerate(dets[:2]):
            rec["lm"][i] = arr
            rec["present"][i] = True
            rec["score"][i] = sc
        return rec

    def close(self):
        self._det.close()


def to_px(rec: dict) -> np.ndarray:
    """normalised coords -> aspect-corrected (x*W, y*H, z*W), shape (2,21,3)."""
    w, h = rec["wh"]
    return rec["lm"] * np.array([w, h, w], np.float32)


def hand_size(px: np.ndarray, present: np.ndarray) -> float:
    """mean wrist->middle-MCP distance of present hands (pixels)."""
    d = [np.linalg.norm(px[i, 9] - px[i, 0]) for i in range(2) if present[i]]
    return float(np.mean(d)) if d else 0.0


def draw_skeleton(img: np.ndarray, rec: dict, color=(80, 230, 120), thickness: int = 2):
    """Draw both detected hands on `img` (BGR) with a soft dark outline for contrast."""
    h, w = img.shape[:2]
    for i in range(2):
        if not rec["present"][i]:
            continue
        pts = [(int(x * w), int(y * h)) for x, y, _ in rec["lm"][i]]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (20, 20, 20), thickness + 2, cv2.LINE_AA)
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)
        for j, p in enumerate(pts):
            cv2.circle(img, p, 4 if j in (4, 8, 12, 16, 20) else 3, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, p, 3 if j in (4, 8, 12, 16, 20) else 2, (255, 255, 255), -1, cv2.LINE_AA)
    return img
