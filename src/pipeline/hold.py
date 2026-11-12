"""
src/pipeline/hold.py
====================
Online "static fingerspelling" recogniser: turns per-frame class probabilities into typed
letters by detecting HOLDS (hand roughly still + stable prediction).

Per frame:
  1. motion   = mean landmark displacement vs. previous frame, in HAND SIZES per frame (EMA)
  2. p_smooth = EMA of calibrated class probabilities
  3. a letter is COMMITTED when its smoothed argmax has been the same for `stable_frames`
     frames, confidence >= `min_conf` and motion < `still_thr`
  4. after a commit the recogniser must be RELEASED (prediction changes, a clear motion
     burst, or the hand leaves) before it can type the same letter again -> "LL" works
  5. hands absent for `word_gap_s` seconds -> a space is typed

For uploaded videos the frame with the highest confidence inside the hold is stored as
the keyframe of that letter.
"""

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.hands import to_px, hand_size

LETTERS = [chr(65 + i) for i in range(26)]


@dataclass
class HoldConfig:
    alpha_p: float = 0.55        # EMA weight of the newest probability vector
    alpha_m: float = 0.5         # EMA weight of the newest motion value
    still_thr: float = 0.060     # motion (hand-sizes/frame) below which the hand counts as still
    release_motion: float = 0.10 # motion burst that re-arms the same letter
    min_conf: float = 0.60
    stable_frames: int = 5
    word_gap_s: float = 1.4
    release_frames: int = 4      # frames of a different, confident letter that also count as release
    release_conf: float = 0.5    # a different letter must reach this confidence to count (ignores flicker)
    repeat_hold_s: float = 1.6   # keep the same sign this long and it types again (like key auto-repeat)


@dataclass
class Recognizer:
    cfg: HoldConfig = field(default_factory=HoldConfig)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.text = ""
        self.events = []
        self._p = None
        self._prev_px = None
        self._prev_present = None
        self._motion = 0.0
        self._cand = None
        self._streak = 0
        self._best = (0.0, -1)          # (conf, frame_idx) inside current streak
        self._armed = True
        self._committed = None
        self._commit_t = 0.0
        self._raw_motion = 0.0
        self._diff_streak = 0
        self._last_hand_t = None
        self.letters = []               # [(letter, conf, frame_idx, t)]

    # ------------------------------------------------------------------ main step
    def step(self, rec: dict, probs, t: float, frame_idx: int = 0) -> dict:
        cfg = self.cfg
        out = {"letter": None, "conf": 0.0, "top3": [], "status": "no_hand",
               "text": self.text, "event": None, "motion": self._motion}
        if probs is None or not rec["present"].any():
            self._prev_px = None
            self._p = None
            self._cand, self._streak, self._best = None, 0, (0.0, -1)
            self._armed = True
            self._committed = None
            if (self._last_hand_t is not None and t - self._last_hand_t >= cfg.word_gap_s
                    and self.text and not self.text.endswith(" ")):
                self.text += " "
                out["event"] = {"type": "space", "t": t}
                self.events.append(out["event"])
            out["text"] = self.text
            return out
        self._last_hand_t = t

        # motion in hand sizes / frame
        px = to_px(rec)
        hs = max(hand_size(px, rec["present"]), 1e-6)
        if self._prev_px is not None and np.array_equal(rec["present"], self._prev_present):
            d = np.linalg.norm(px[rec["present"]] - self._prev_px[rec["present"]], axis=-1).mean() / hs
        else:
            d = self._motion
        d = float(d)
        self._raw_motion = d
        self._motion = float(cfg.alpha_m * d + (1 - cfg.alpha_m) * self._motion)
        self._prev_px, self._prev_present = px, rec["present"].copy()

        # probability smoothing
        self._p = probs if self._p is None else cfg.alpha_p * probs + (1 - cfg.alpha_p) * self._p
        k = int(self._p.argmax())
        conf = float(self._p[k])
        top = np.argsort(-self._p)[:3]
        out.update(letter=LETTERS[k], conf=conf, motion=float(self._motion),
                   top3=[(LETTERS[i], float(self._p[i])) for i in top])

        # candidate streak
        if k == self._cand:
            self._streak += 1
            if conf > self._best[0]:
                self._best = (conf, frame_idx)
        else:
            self._cand, self._streak, self._best = k, 1, (conf, frame_idx)

        # release logic
        if self._committed is not None:
            if k != self._committed and conf >= cfg.release_conf:     # a convincing OTHER letter
                self._diff_streak += 1
                if self._diff_streak >= cfg.release_frames:
                    self._armed = True
            elif k == self._committed:
                self._diff_streak = 0
            if max(self._motion, self._raw_motion) > cfg.release_motion:
                self._armed = True
            if t - self._commit_t >= cfg.repeat_hold_s:
                self._armed = True

        still = self._motion < cfg.still_thr
        out["status"] = "holding" if still else "moving"
        if (self._streak >= cfg.stable_frames and conf >= cfg.min_conf and still
                and (self._armed or k != self._committed)):
            letter = LETTERS[k]
            self.text += letter
            ev = {"type": "letter", "letter": letter, "conf": float(self._best[0] if self._best[1] >= 0 else conf),
                  "frame": int(self._best[1] if self._best[1] >= 0 else frame_idx), "t": float(t)}
            self.letters.append((letter, ev["conf"], ev["frame"], t))
            self.events.append(ev)
            out["event"] = ev
            out["status"] = "committed"
            self._committed, self._armed, self._diff_streak, self._commit_t = k, False, 0, t
        elif self._committed == k and not self._armed:
            out["status"] = "committed"
        out["text"] = self.text
        return out
