"""
tests/test_ctc.py
=================
Checks of the CTC helpers (src/pipeline/ctc.py): greedy decoding, key-frame positions,
edit distance / CER and Viterbi forced alignment on a synthetic posterior matrix.

  python tests/test_ctc.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import ctc  # noqa: E402


def synthetic(word: str, spacing: int = 6, lead: int = 3):
    """(T,27) log-probs: each letter peaks for 2 frames, blanks elsewhere."""
    ids = [ctc.LETTERS.index(c) for c in word]
    T = lead + spacing * len(ids) + 3
    p = np.full((T, 27), 0.01)
    p[:, ctc.BLANK] = 0.9
    peaks = []
    for k, i in enumerate(ids):
        f = lead + spacing * k
        p[f:f + 2, :] = 0.01
        p[f:f + 2, i] = 0.95
        peaks.append(f)
    return np.log(p / p.sum(1, keepdims=True)), peaks


def test_greedy_and_positions():
    lp, peaks = synthetic("hello")                      # includes a repeated letter
    assert ctc.greedy_decode(lp) == "hello"
    found = ctc.greedy_with_pos(lp)
    assert "".join(c for c, _, _ in found) == "hello"
    assert [f for _, f, _ in found] == [p for p in peaks] or all(abs(f - p) <= 1 for (_, f, _), p in zip(found, peaks))
    assert all(pr > 0.7 for _, _, pr in found)   # peaks normalise to ~0.79


def test_forced_alignment_matches_peaks():
    lp, peaks = synthetic("india")
    spans = ctc.forced_align(lp, [ctc.LETTERS.index(c) for c in "india"])
    assert spans is not None and len(spans) == 5
    for (a, b), p in zip(spans, peaks):
        assert a <= p + 1 and b >= p, (a, b, p)


def test_cer():
    assert ctc.edit_distance("kitten", "sitting") == 3
    assert abs(ctc.cer(["abc", "xx"], ["abd", "xx"]) - 1 / 5) < 1e-9


def test_alignment_impossible():
    lp, _ = synthetic("ab")
    assert ctc.forced_align(lp[:1], [0, 1, 2]) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("ALL PASSED")
