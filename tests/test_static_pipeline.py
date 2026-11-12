"""
tests/test_static_pipeline.py
=============================
End-to-end check of the static (hold-based) recogniser on a SYNTHETIC video: every letter
of a word is shown as a still image for ~0.8 s, optionally with an empty frame between
letters. Verifies that the typed text equals the word (including the double letter in HELLO).

  python tests/test_static_pipeline.py [WORD ...]
"""

import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.classifier import StaticClassifier  # noqa: E402
from src.pipeline.hands import HandTracker  # noqa: E402
from src.pipeline.video import analyze_static  # noqa: E402

SRC = ROOT / "data" / "ISL_DATA"
SIZE = (640, 480)


def letter_frame(letter: str, k: int):
    """letterbox (never squash: the classifier is sensitive to hand-shape distortion)."""
    img = cv2.imread(str(SRC / letter / f"{k}.jpg"))
    s = min(SIZE[0] / img.shape[1], SIZE[1] / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    out = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    y, x = (SIZE[1] - img.shape[0]) // 2, (SIZE[0] - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    return out


def make_video(word: str, path: Path, gaps: bool, hold: int = 24, gap_frames: int = 8, fps: int = 30):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, SIZE)
    blank = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    for i, ch in enumerate(word):
        img = letter_frame(ch, 10 + 7 * i)
        for _ in range(hold):
            wr.write(np.clip(img.astype(np.int16) + np.random.randint(-3, 4, img.shape), 0, 255).astype(np.uint8))
        if gaps:
            for _ in range(gap_frames):
                wr.write(blank)
    wr.release()


def run(word: str, gaps: bool, tracker, clf) -> str:
    with tempfile.TemporaryDirectory() as d:
        vid = Path(d) / "t.mp4"
        make_video(word, vid, gaps)
        res = analyze_static(vid, tracker, clf, Path(d) / "out", write_video=False)
    return res["text"].replace(" ", "")


if __name__ == "__main__":
    words = [w.upper() for w in sys.argv[1:]] or ["HELLO", "INDIA", "ABCDE"]
    clf = StaticClassifier()
    tracker = HandTracker(recover=clf.recover)
    ok = True
    for w in words:
        for gaps in (True, False):
            got = run(w, gaps, tracker, clf)
            # Known limitation: with NO pause at all, a repeated letter (LL) is indistinguishable
            # from one long hold unless the hold exceeds HoldConfig.repeat_hold_s (1.6 s).
            expected = w if gaps else re.sub(r"(.)\1+", r"\1", w)
            flag = "OK " if got == expected else "DIFF"
            ok &= got == expected
            print(f"{flag} word={w:<8} gaps={str(gaps):<5} -> {got}   (expected {expected})")
    print("ALL PASSED" if ok else "SOME DIFFERENCES (see above)")
