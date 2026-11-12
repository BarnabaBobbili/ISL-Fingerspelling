"""
src/pipeline/ctc.py
===================
CTC helpers shared by training, evaluation and the app: greedy decoding, edit distance and
Viterbi FORCED ALIGNMENT (used to pick one key frame per recognised letter).
Vocabulary: 26 letters a-z = classes 0..25, CTC blank = 26.
"""

import numpy as np

BLANK = 26
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def greedy_decode(log_probs: np.ndarray) -> str:
    """(T, 27) -> collapsed string (repeats merged, blanks removed)."""
    ids = log_probs.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK:
            out.append(LETTERS[int(i)])
        prev = i
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def cer(hyps, refs) -> float:
    """corpus-level character error rate."""
    return sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / max(1, sum(len(r) for r in refs))


def forced_align(log_probs: np.ndarray, target: list):
    """
    Viterbi CTC forced alignment of `target` (class ids, no blanks) to (T, 27) log-probs.
    Returns [(start, end_inclusive)] per target symbol, or None if it cannot fit.
    """
    T, L = log_probs.shape[0], len(target)
    S = 2 * L + 1
    if L == 0 or T < L:
        return None
    ext = np.full(S, BLANK, dtype=np.int64)
    ext[1::2] = target
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = log_probs[0, ext[0]]
    dp[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], 1
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], 2
            dp[t, s] = best + log_probs[t, ext[s]]
            bp[t, s] = arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] <= NEG / 2:
        return None
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = []
    for k in range(L):
        fr = np.where(path == 2 * k + 1)[0]
        if len(fr) == 0:
            return None
        spans.append((int(fr[0]), int(fr[-1])))
    return spans


def greedy_with_pos(log_probs: np.ndarray):
    """
    Greedy CTC decoding that also reports WHERE each letter fired.
    returns [(char, peak_frame, peak_prob)] - for each run of one class the frame of maximum
    probability inside the run (this is the key frame of that letter).
    """
    ids = log_probs.argmax(-1)
    out, t, T = [], 0, len(ids)
    while t < T:
        c = int(ids[t])
        s = t
        while t < T and int(ids[t]) == c:
            t += 1
        if c != BLANK:
            seg = log_probs[s:t, c]
            k = int(seg.argmax())
            out.append((LETTERS[c], s + k, float(np.exp(seg[k]))))
    return out


def greedy_runs(log_probs: np.ndarray):
    """
    Greedy CTC decoding with full run information.
    returns [(char, start, end_inclusive, peak_frame, peak_prob)] for every non-blank run.
    Used by the streaming recogniser to decide which letters are old enough to commit.
    """
    ids = log_probs.argmax(-1)
    out, t, T = [], 0, len(ids)
    while t < T:
        c = int(ids[t])
        s = t
        while t < T and int(ids[t]) == c:
            t += 1
        if c != BLANK:
            seg = log_probs[s:t, c]
            k = int(seg.argmax())
            out.append((LETTERS[c], s, t - 1, s + k, float(np.exp(seg[k]))))
    return out
