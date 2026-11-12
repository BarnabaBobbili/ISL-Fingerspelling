"""
src/pipeline/continuous.py
==========================
Continuous (CTC) fingerspelling recognition for uploaded videos and the live webcam.

  ContinuousModel     frozen DH-GAT trunk + trained sequence model (Conformer/BiLSTM-CTC)
  analyze_continuous  offline: decode a whole video, one key frame per letter, annotated video
  ContinuousSession   online: sliding-window re-decoding of the last seconds of webcam frames

The vocabulary has no space symbol (the dataset transcripts were stripped of spaces), so
words are separated by a PAUSE: hands leaving the picture for ~1.2 s commits the word.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.models.cont_v2 import SeqCTC  # noqa: E402
from src.pipeline import ctc  # noqa: E402
from src.pipeline import features as F  # noqa: E402
from src.pipeline.overlay import draw_hud  # noqa: E402
from src.pipeline.video import _open, _writer, save_keyframes, hands_json  # noqa: E402

CKPT = ROOT / "checkpoints" / "cont_best.pt"


class ContinuousModel:
    def __init__(self, ck: dict, trunk_ckpt=None):
        self.cfg = ck["cfg"]
        self.model = SeqCTC(**self.cfg)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.mean, self.std, self.cols = ck["mean"], ck["std"], ck["cols"]
        self.trunk, self.temp = F.load_trunk(trunk_ckpt)
        self.name = f"{'Conformer' if self.cfg['kind'] == 'conformer' else 'BiLSTM'}-CTC v2"
        self.val_cer, self.test_cer = ck.get("val_cer"), ck.get("test_cer")

    @classmethod
    def load(cls, path=None, trunk_ckpt=None):
        ck = torch.load(Path(path or CKPT), map_location="cpu", weights_only=False)
        return cls(ck, trunk_ckpt)

    @torch.no_grad()
    def log_probs(self, feats: np.ndarray) -> np.ndarray:
        """(T, 351) raw features -> (T, 27) log-probabilities."""
        x = (torch.from_numpy(feats[:, self.cols]).float() - self.mean) / self.std
        lp, _ = self.model(x[None], torch.tensor([len(x)]))
        return lp[:, 0].numpy()

    def video_features(self, px: np.ndarray, present: np.ndarray) -> np.ndarray:
        return F.sequence_features(self.trunk, self.temp, px, present)


# ─────────────────────────────────────────────────────────────────────────────
def analyze_continuous(path, tracker, out_dir, model: ContinuousModel = None, progress_cb=None,
                       write_video: bool = True) -> dict:
    model = model or ContinuousModel.load()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cap, fps = _open(path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    recs, hands = [], []
    while True:                                                       # pass 1: hands in every frame
        ok, frame = cap.read()
        if not ok:
            break
        r = tracker(frame)
        recs.append(r)
        hands.append(hands_json(r))
        if progress_cb and len(recs) % 10 == 0:
            progress_cb(0.85 * len(recs) / n_total)
    cap.release()
    T = len(recs)
    if T == 0:
        raise ValueError("no frames could be read")
    W, H = int(recs[0]["wh"][0]), int(recs[0]["wh"][1])
    px = np.stack([F.to_px(r["lm"], r["wh"]) for r in recs])
    present = np.stack([r["present"] for r in recs])
    lp = model.log_probs(model.video_features(px, present))
    found = ctc.greedy_with_pos(lp)                                   # [(char, frame, prob)]
    letters = [{"letter": c.upper(), "conf": round(p, 3), "frame": int(f), "t": round(f / fps, 3)}
               for c, f, p in found]

    # per-frame HUD state: text typed so far, last letter, current top-3
    frames, k, text = [], 0, ""
    last = (None, 0.0)
    for f in range(T):
        while k < len(found) and found[k][1] <= f:
            text += found[k][0].upper()
            last = (found[k][0].upper(), found[k][2])
            k += 1
        pr = np.exp(lp[f, :26])
        top = np.argsort(-pr)[:3]
        frames.append({"f": f, "t": round(f / fps, 3), "letter": last[0], "conf": round(last[1], 3),
                       "status": "reading" if present[f].any() else "no_hand", "text": text,
                       "top3": [(ctc.LETTERS[i].upper(), round(float(pr[i]), 3)) for i in top],
                       "hands": hands[f]})
    wr_name = None
    if write_video:                                                   # pass 2: annotated video
        cap = cv2.VideoCapture(str(path))
        wr, wr_name = _writer(out_dir, fps, (W, H))
        if wr is not None:
            for f in range(T):
                ok, frame = cap.read()
                if not ok:
                    break
                st = dict(frames[f])
                st["status"] = "committed" if (st["status"] == "reading" and any(
                    abs(l["frame"] - f) <= 3 for l in letters)) else ("holding" if st["status"] == "reading" else "no_hand")
                wr.write(draw_hud(frame, recs[f], st))
            wr.release()
        cap.release()
        if progress_cb:
            progress_cb(0.95)
    save_keyframes(path, letters, out_dir)
    return {"mode": "continuous", "model": model.name, "fps": fps, "n_frames": T, "duration": round(T / fps, 2),
            "width": W, "height": H, "text": text, "letters": letters, "frames": frames,
            "annotated": wr_name, "elapsed": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────
class ContinuousSession:
    """Streaming recogniser: re-decodes a sliding window of the most recent frames."""

    def __init__(self, model: ContinuousModel, gap_s: float = 1.2, max_frames: int = 360, every: int = 3):
        self.m, self.gap_s, self.max_frames, self.every = model, gap_s, max_frames, every
        self.reset()

    def reset(self):
        self.committed = ""
        self._clear_window()

    def _clear_window(self):
        self.buf, self.hyp, self.found = [], "", []
        self.prev_px, self.prev_present = None, None
        self.last_hand_t = None
        self.lp = None

    @property
    def text(self):
        return self.committed + self.hyp.upper()

    @text.setter
    def text(self, v):                       # used by the web app's space / backspace buttons
        self.committed = v
        self._clear_window()

    def _commit(self):
        if self.hyp:
            self.committed += self.hyp.upper() + " "
        self._clear_window()

    def step(self, rec: dict, t: float, n: int) -> dict:
        present = bool(rec["present"].any())
        out = {"letter": None, "conf": 0.0, "top3": [], "status": "no_hand", "event": None, "motion": 0.0}
        if present:
            px = F.to_px(rec["lm"], rec["wh"])[None]
            pr = rec["present"][None]
            feat = np.concatenate([F.static_part(self.m.trunk, self.m.temp, px, pr),
                                   F.motion_part(px, pr, self.prev_px, self.prev_present)], axis=1)[0]
            self.prev_px, self.prev_present = px[0], rec["present"].copy()
            self.buf.append(feat)
            self.last_hand_t = t
            if len(self.buf) > self.max_frames:
                self._commit()
                self.buf.append(feat)
            if len(self.buf) >= 6 and n % self.every == 0:
                self.lp = self.m.log_probs(np.stack(self.buf))
                self.found = ctc.greedy_with_pos(self.lp)
                self.hyp = "".join(c for c, _, _ in self.found)
            out["status"] = "reading"
        elif self.buf and self.last_hand_t is not None and t - self.last_hand_t >= self.gap_s:
            self.lp = self.m.log_probs(np.stack(self.buf))             # final pass over the whole word
            self.hyp = "".join(c for c, _, _ in ctc.greedy_with_pos(self.lp))
            self._commit()
            out["event"] = {"type": "word"}
        if self.found:
            out["letter"], out["conf"] = self.found[-1][0].upper(), self.found[-1][2]
        if self.lp is not None and present:
            pr = np.exp(self.lp[-1, :26])
            out["top3"] = [(ctc.LETTERS[i].upper(), float(pr[i])) for i in np.argsort(-pr)[:3]]
        out["text"] = self.text
        return out
