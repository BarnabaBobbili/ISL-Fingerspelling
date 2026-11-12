"""
src/pipeline/streaming.py
=========================
Real-time continuous fingerspelling.

  Resampler          puts landmark samples that arrive at ANY camera rate (10-60 fps, jittery)
                     onto the fixed 30 Hz grid the sequence model was trained on
  StreamingSession   sliding-window CTC decoding with a STABLE-PREFIX COMMIT:
                       * a letter becomes final ("committed") once its CTC run has ended and
                         its peak is at least `commit_lag` frames old - it never changes again
                       * newer letters are reported as a TENTATIVE tail that may still change
                       * the window slides past committed letters (keeping `left_ctx` frames of
                         context), so cost and latency stay bounded for arbitrarily long spelling
                       * hands away for `gap_s` seconds ends the word (the vocabulary has no space)

Latency of a letter = commit time - peak time ~= commit_lag / 30 s + one decode interval.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import ctc  # noqa: E402
from src.pipeline import features as F  # noqa: E402


class Resampler:
    """Linear interpolation of two-hand landmarks onto a fixed-rate time grid."""

    def __init__(self, hz: float = 30.0, max_gap: float = 0.25):
        self.dt, self.max_gap = 1.0 / hz, max_gap
        self.reset()

    def reset(self):
        self.pt = None          # previous sample time
        self.ppx = None
        self.ppr = None
        self.g = None           # next grid time

    def push(self, t: float, px: np.ndarray, present: np.ndarray):
        """add one sample (px (2,21,3), present (2,)); returns the grid frames it completes."""
        if self.pt is None or t <= self.pt or t - self.pt > self.max_gap:
            self.pt, self.ppx, self.ppr = t, px, present.copy()     # (re)start the grid here
            self.g = t + self.dt
            return [(px, present.copy())]
        out = []
        same = bool((present == self.ppr).all())
        while self.g <= t + 1e-9:
            a = (self.g - self.pt) / (t - self.pt)
            if same:                                # same hands in both samples -> interpolate
                q = self.ppx * (1 - a) + px * a
                pr = present.copy()
            elif a < 0.5:                           # hand set changed -> nearest sample
                q, pr = self.ppx, self.ppr.copy()
            else:
                q, pr = px, present.copy()
            out.append((q.astype(np.float32), pr))
            self.g += self.dt
        self.pt, self.ppx, self.ppr = t, px, present.copy()
        return out


class StreamingSession:
    def __init__(self, model, hz: float = 30.0, every: int = 2, commit_lag: int = 12, left_ctx: int = 60,
                 max_win: int = 240, gap_s: float = 1.0, min_frames: int = 6, resample: bool = True):
        self.m = model
        self.hz, self.every, self.commit_lag = hz, every, commit_lag
        self.left_ctx, self.max_win, self.gap_s, self.min_frames = left_ctx, max_win, gap_s, min_frames
        self.resample = resample
        self.rs = Resampler(hz)
        self.latencies = []                 # frames between a letter's peak and its commit
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.done = ""                      # finished words (with trailing spaces)
        self._new_word()

    def _new_word(self):
        self.word = ""                      # committed letters of the current word
        self.tent = []                      # [(char, prob)] tentative tail
        self.feats = []                     # features of grid frames in the window
        self.base = 0                       # absolute index of feats[0]
        self.n = 0                          # absolute index of the next grid frame
        self.boundary = -1                  # letters peaking at/before this absolute frame are final
        self.since = 0                      # grid frames since the last decode
        self.prev = None                    # previous grid frame (px, present) for motion features
        self.last_hand_t = None
        self.lp = None
        self.last = (None, 0.0)             # most recent letter and its probability
        self.rs.reset()

    @property
    def text(self):
        return self.done + self.word.upper()

    @text.setter
    def text(self, v):                      # space / backspace buttons of the web app
        self.done = v
        self._new_word()

    @property
    def tentative(self):
        return "".join(c for c, _ in self.tent).upper()

    # ---------------------------------------------------------------- decoding
    def _decode(self, final: bool = False):
        self.lp = self.m.log_probs(np.stack(self.feats))
        W = len(self.feats)
        tent, open_tail = [], False
        for ch, s, e, pk, pr in ctc.greedy_runs(self.lp):
            if self.base + pk <= self.boundary:
                continue                                        # already committed
            old_enough = e < W - 1 and pk <= W - 1 - self.commit_lag
            if (final or old_enough) and not open_tail:
                self.word += ch
                self.boundary = self.base + e
                self.latencies.append(self.n - 1 - (self.base + pk))
                self.last = (ch.upper(), pr)
            else:
                open_tail = True                                # keep order: everything after stays tentative
                tent.append((ch, pr))
        self.tent = tent
        if tent:
            self.last = (tent[-1][0].upper(), tent[-1][1])
        # slide the window: keep left context before the commit boundary, cap total length
        keep = max(self.base, self.boundary + 1 - self.left_ctx, self.n - self.max_win)
        if keep > self.base:
            del self.feats[:keep - self.base]
            self.base = keep
        self.since = 0

    def _end_word(self):
        if len(self.feats) >= 2:
            self._decode(final=True)
        if self.word:
            self.done += self.word.upper() + " "
        done = self.done
        self._new_word()
        self.done = done

    # -------------------------------------------------------------------- step
    def step(self, rec: dict, t: float, n: int = 0) -> dict:
        present = rec["present"]
        px = F.to_px(rec["lm"], rec["wh"])
        frames = self.rs.push(t, px, present) if self.resample else [(px, present.copy())]
        out = {"letter": None, "conf": 0.0, "top3": [], "status": "no_hand", "event": None, "motion": 0.0}
        for q, pr in frames:
            if not pr.any():
                self.prev = None
                continue
            pq, ppr = (self.prev if self.prev is not None else (None, None))
            feat = np.concatenate([F.static_part(self.m.trunk, self.m.temp, q[None], pr[None]),
                                   F.motion_part(q[None], pr[None], pq, ppr)], axis=1)[0]
            self.prev = (q, pr)
            self.feats.append(feat)
            self.n += 1
            self.since += 1
        if present.any():
            self.last_hand_t = t
            out["status"] = "reading"
            if self.since >= self.every and len(self.feats) >= self.min_frames:
                self._decode()
        elif self.feats and self.last_hand_t is not None and t - self.last_hand_t >= self.gap_s:
            self._end_word()
            out["event"] = {"type": "word"}
        out["letter"], out["conf"] = self.last
        if self.lp is not None and present.any():
            p = np.exp(self.lp[-1, :26])
            out["top3"] = [(ctc.LETTERS[i].upper(), float(p[i])) for i in np.argsort(-p)[:3]]
        out["text"] = self.text
        out["tentative"] = self.tentative
        return out

    def flush(self) -> str:
        """finish the current word (end of a clip) and return the full text."""
        if self.feats:
            self._end_word()
        return self.done
