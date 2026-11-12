"""
src/pipeline/cli.py
===================
Command-line front end for the recognition pipeline.

  python -m src.pipeline.cli video  path/to/video.mp4 [--mode static|continuous] [--out out_dir]
  python -m src.pipeline.cli webcam [--camera 0] [--no-mirror]

Webcam keys:  q quit   c clear text   space add a space   b backspace
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def cmd_video(a):
    from src.pipeline.hands import HandTracker
    out = Path(a.out or ROOT / "results" / "video_out" / Path(a.path).stem)
    if a.mode == "static":
        from src.pipeline.classifier import StaticClassifier
        from src.pipeline.video import analyze_static
        clf = StaticClassifier()
        res = analyze_static(a.path, HandTracker(recover=clf.recover), clf, out,
                             progress_cb=lambda p: print(f"\r{p*100:5.1f}%", end="", flush=True))
    else:
        from src.pipeline.continuous import analyze_continuous
        res = analyze_continuous(a.path, HandTracker(), out,
                                 progress_cb=lambda p: print(f"\r{p*100:5.1f}%", end="", flush=True))
    print(f"\n[{res['mode']}] text      : {res['text']!r}")
    print(f"letters   : {' '.join(l['letter'] for l in res['letters'])}")
    print(f"keyframes : {out/'keyframes'}   annotated: {res['annotated']}   ({res['elapsed']}s)")


def cmd_webcam(a):
    from src.pipeline.hands import HandTracker
    from src.pipeline.classifier import StaticClassifier
    from src.pipeline.hold import Recognizer
    from src.pipeline.overlay import draw_hud
    clf = StaticClassifier()
    tracker, rcg = HandTracker(recover=clf.recover), Recognizer()
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("cannot open camera")
    t0 = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not a.no_mirror:
            frame = cv2.flip(frame, 1)
        rec = tracker(frame)
        st = rcg.step(rec, clf.predict(rec), time.time() - t0, n)
        n += 1
        cv2.imshow("ISL fingerspelling", draw_hud(frame, rec, st))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            rcg.reset()
        elif k == ord(" "):
            rcg.text += " "
        elif k == ord("b"):
            rcg.text = rcg.text[:-1]
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video")
    v.add_argument("path")
    v.add_argument("--mode", choices=["static", "continuous"], default="static")
    v.add_argument("--out", default=None)
    v.set_defaults(fn=cmd_video)
    w = sub.add_parser("webcam")
    w.add_argument("--camera", type=int, default=0)
    w.add_argument("--no-mirror", action="store_true")
    w.set_defaults(fn=cmd_webcam)
    a = ap.parse_args()
    a.fn(a)
