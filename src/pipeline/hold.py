"""
src/pipeline/hold.py
====================
Online "static fingerspelling" recogniser: turns per-frame class probabilities into typed
letters by detecting HOLDS (hand roughly still + stable prediction).

Everything is expressed in SECONDS, so behaviour is the same at any camera frame rate
(15 fps webcam, 30 fps video, 60 fps browser tracking):

  1. motion   = mean landmark speed in HAND SIZES PER SECOND (time-constant smoothing)
  2. p_smooth = time-constant smoothing of the calibrated class probabilities
  3. a letter is COMMITTED when its smoothed arg-max has been the same for `stable_s`
     seconds, confidence >= `min_conf` and motion < `still_thr`
  4. after a commit the recogniser must be RE-ARMED before it can type the same letter again:
     a different CONFIDENT letter for `release_s`, a motion burst, the hand leaving, or the
     same sign held for `repeat_hold_s` (auto-repeat)  -> "LL" works
  5. hands absent for `word_gap_s` seconds -> a space is typed

For uploaded videos the frame with the highest confidence inside the hold is stored as
the key frame of that letter.
"""

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.hands import to_px, hand_size

LETTERS = [chr(65 + i) for i in range(26)]
REF_DT = 1.0 / 30.0           # the smoothing weights below are defined at 30 fps


@dataclass
class HoldConfig:
    alpha_p: float = 0.55         # weight of the newest probability vector at 30 fps
    alpha_m: float = 0.5          # weight of the newest motion value at 30 fps
    still_thr: float = 1.8        # speed (hand sizes / second) below which the hand counts as still
    release_motion: float = 3.0   # speed burst (hand sizes / second) that re-arms the same letter
    min_conf: float = 0.60
    stable_s: float = 0.13        # the same letter must persist this long before it is typed
    word_gap_s: float = 1.4
    release_s: float = 0.10       # a different, confident letter for this long also re-arms
    release_conf: float = 0.5     # ...and must reach this confidence (ignores flicker)
    repeat_hold_s: float = 1.6    # keep the same sign this long and it types again (key auto-repeat)


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
        self._prev_t = None
        self._motion = 0.0
        self._raw_motion = 0.0
        self._cand = None
        self._cand_t0 = 0.0
        self._best = (0.0, -1)          # (conf, frame_idx) inside the current candidate run
        self._armed = True
        self._committed = None
        self._commit_t = 0.0
        self._diff_t0 = None
        self._last_hand_t = None
        self.letters = []               # [(letter, conf, frame_idx, t)]

    # ------------------------------------------------------------------ main step
    def step(self, rec: dict, probs, t: float, frame_idx: int = 0) -> dict:
        cfg = self.cfg
        out = {"letter": None, "conf": 0.0, "top3": [], "status": "no_hand",
               "text": self.text, "event": None, "motion": self._motion}
        if probs is None or not rec["present"].any():
            self._prev_px = None
            self._prev_t = None
            self._p = None
            self._cand, self._best = None, (0.0, -1)
            self._armed = True
            self._committed = None
            self._diff_t0 = None
            if (self._last_hand_t is not None and t - self._last_hand_t >= cfg.word_gap_s
                    and self.text and not self.text.endswith(" ")):
                self.text += " "
                out["event"] = {"type": "space", "t": float(t)}
                self.events.append(out["event"])
            out["text"] = self.text
            return out
        self._last_hand_t = t
        dt = REF_DT if self._prev_t is None else min(max(t - self._prev_t, 1.0 / 120.0), 0.25)
        self._prev_t = t
        k_dt = dt / REF_DT                               # how many 30-fps frames this step spans
        a_m = 1.0 - (1.0 - cfg.alpha_m) ** k_dt
        a_p = 1.0 - (1.0 - cfg.alpha_p) ** k_dt

        # motion in hand sizes / second
        px = to_px(rec)
        hs = max(hand_size(px, rec["present"]), 1e-6)
        if self._prev_px is not None and np.array_equal(rec["present"], self._prev_present):
            v = float(np.linalg.norm(px[rec["present"]] - self._prev_px[rec["present"]], axis=-1).mean() / hs / dt)
        else:
            v = self._motion
        self._raw_motion = v
        self._motion = float(a_m * v + (1 - a_m) * self._motion)
        self._prev_px, self._prev_present = px, rec["present"].copy()

        # probability smoothing
        self._p = probs if self._p is None else a_p * probs + (1 - a_p) * self._p
        k = int(self._p.argmax())
        conf = float(self._p[k])
        top = np.argsort(-self._p)[:3]
        out.update(letter=LETTERS[k], conf=conf, motion=float(self._motion),
                   top3=[(LETTERS[i], float(self._p[i])) for i in top])

        # candidate run
        if k == self._cand:
            if conf > self._best[0]:
                self._best = (conf, frame_idx)
        else:
            self._cand, self._cand_t0, self._best = k, t, (conf, frame_idx)

        # re-arming
        if self._committed is not None:
            if k != self._committed and conf >= cfg.release_conf:     # a convincing OTHER letter
                if self._diff_t0 is None:
                    self._diff_t0 = t
                if t - self._diff_t0 >= cfg.release_s:
                    self._armed = True
            elif k == self._committed:
                self._diff_t0 = None
            if max(self._motion, self._raw_motion) > cfg.release_motion:
                self._armed = True
            if t - self._commit_t >= cfg.repeat_hold_s:
                self._armed = True

        still = self._motion < cfg.still_thr
        out["status"] = "holding" if still else "moving"
        if (t - self._cand_t0 >= cfg.stable_s and conf >= cfg.min_conf and still
                and (self._armed or k != self._committed)):
            letter = LETTERS[k]
            self.text += letter
            ev = {"type": "letter", "letter": letter, "conf": float(self._best[0] if self._best[1] >= 0 else conf),
                  "frame": int(self._best[1] if self._best[1] >= 0 else frame_idx), "t": float(t)}
            self.letters.append((letter, ev["conf"], ev["frame"], t))
            self.events.append(ev)
            out["event"] = ev
            out["status"] = "committed"
            self._committed, self._armed, self._diff_t0, self._commit_t = k, False, None, t
        elif self._committed == k and not self._armed:
            out["status"] = "committed"
        out["text"] = self.text
        return out
