"""
src/pipeline/video.py
=====================
Offline analysis of an uploaded video.

  analyze_static(path, ...)      hold-based recognition with the DH-GAT v2 classifier
  analyze_continuous(path, ...)  CTC recognition with the continuous model (see continuous.py)

Both return a JSON-serialisable dict:
  {mode, fps, n_frames, duration, width, height, text,
   letters:[{letter, conf, frame, t, thumb}],     # one entry per recognised letter
   frames:[{f, t, letter, conf, status, text}],   # per-frame HUD data (browser overlay)
   annotated: "annotated.webm" | None}
and write keyframe thumbnails (and the annotated video) into `out_dir`.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.hold import Recognizer, HoldConfig  # noqa: E402
from src.pipeline.overlay import draw_hud  # noqa: E402


def hands_json(rec):
    """compact per-frame landmarks (normalised x,y) for the browser overlay."""
    return [[[round(float(x), 3), round(float(y), 3)] for x, y, _ in rec["lm"][i]]
            for i in range(2) if rec["present"][i]]


def _open(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video file: {Path(path).name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps < 1 or fps > 120:
        fps = 30.0
    return cap, fps


def _writer(out_dir: Path, fps: float, size):
    """Try browser-playable codecs; returns (writer, filename) or (None, None)."""
    for fourcc, name in (("VP80", "annotated.webm"), ("VP90", "annotated.webm"), ("mp4v", "annotated.mp4")):
        wr = cv2.VideoWriter(str(out_dir / name), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if wr.isOpened():
            return wr, name
    return None, None


def save_keyframes(path, letters, out_dir: Path, width: int = 320):
    """Re-read the video and save a thumbnail for every recognised letter."""
    kdir = out_dir / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    for i, ev in enumerate(letters):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ev["frame"]))
        ok, img = cap.read()
        if not ok:
            ev["thumb"] = None
            continue
        s = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * s)))
        name = f"{i:02d}_{ev['letter']}.jpg"
        cv2.imwrite(str(kdir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ev["thumb"] = f"keyframes/{name}"
    cap.release()


def analyze_static(path, tracker, clf, out_dir, cfg: HoldConfig = None, write_video: bool = True,
                   progress_cb=None, flip: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap, fps = _open(path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rcg = Recognizer(cfg or HoldConfig())
    frames, wr, wr_name = [], None, None
    idx, t0 = 0, time.time()
    W = H = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx == 0:
            H, W = frame.shape[:2]
            if write_video:
                wr, wr_name = _writer(out_dir, fps, (W, H))
        rec = tracker(frame, flip=flip)
        probs = clf.predict(rec)
        st = rcg.step(rec, probs, idx / fps, idx)
        frames.append({"f": idx, "t": round(idx / fps, 3), "letter": st["letter"],
                       "conf": round(st["conf"], 3), "status": st["status"], "text": st["text"],
                       "top3": [(a, round(b, 3)) for a, b in st["top3"]], "hands": hands_json(rec)})
        if wr is not None:
            wr.write(draw_hud(frame.copy(), rec, st))
        idx += 1
        if progress_cb and idx % 10 == 0:
            progress_cb(idx / max(n_total, idx))
    cap.release()
    if wr is not None:
        wr.release()
    letters = [{"letter": l, "conf": round(c, 3), "frame": int(f), "t": round(t, 3)}
               for l, c, f, t in rcg.letters]
    save_keyframes(path, letters, out_dir)
    return {"mode": "static", "fps": fps, "n_frames": idx, "duration": round(idx / fps, 2),
            "width": W, "height": H, "text": rcg.text.strip(), "letters": letters,
            "frames": frames, "annotated": wr_name, "elapsed": round(time.time() - t0, 1)}
