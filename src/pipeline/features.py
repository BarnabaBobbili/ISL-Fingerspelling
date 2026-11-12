"""
src/pipeline/features.py
========================
Per-frame feature vector for the CONTINUOUS (CTC) models. The same code is used for
training (cache building), offline video analysis and the live webcam session, so the
model always sees identical features.

  trunk   (192)  pooled DH-GAT v2 embedding           learned hand-shape descriptor
  probs   ( 26)  calibrated static class posteriors    "what letter does this frame look like"
  raw     (126)  shared-frame joint coordinates g      both hands, hand-size units
  motion  (  7)  wrist velocity x2 hands (xy), mean landmark speed, 2 presence flags
  -> 351 numbers per frame

Frames with no hand: trunk/probs/raw = 0, presence = 0.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.models.dh_gat_v2 import DHGATv2, featurize  # noqa: E402

TRUNK_DIM, PROB_DIM, RAW_DIM, MOTION_DIM = 192, 26, 126, 7
FEAT_DIM = TRUNK_DIM + PROB_DIM + RAW_DIM + MOTION_DIM
SLICES = {"trunk": slice(0, 192), "probs": slice(192, 218), "raw": slice(218, 344), "motion": slice(344, 351)}


def load_trunk(ckpt=None):
    ck = torch.load(Path(ckpt or ROOT / "checkpoints" / "dh_gat_v2_main.pt"), map_location="cpu", weights_only=False)
    m = DHGATv2(**ck["cfg"])
    m.load_state_dict(ck["model_state"])
    return m.eval(), float(ck.get("temperature", 1.0))


@torch.no_grad()
def static_part(trunk, temperature, px: np.ndarray, present: np.ndarray, bs: int = 1024) -> np.ndarray:
    """(N,2,21,3) aspect-corrected px + (N,2) present -> (N, 192+26+126) float32."""
    out = np.zeros((len(px), TRUNK_DIM + PROB_DIM + RAW_DIM), np.float32)
    idx = np.where(present.any(1))[0]
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        lm, pr = torch.from_numpy(px[b]).float(), torch.from_numpy(present[b])
        node, glob = featurize(lm, pr)
        emb = trunk.features(node, pr, glob)
        probs = torch.softmax(trunk.head(emb) / temperature, -1)
        out[b, :TRUNK_DIM] = emb.numpy()
        out[b, TRUNK_DIM:TRUNK_DIM + PROB_DIM] = probs.numpy()
        out[b, TRUNK_DIM + PROB_DIM:] = node[..., :3].reshape(len(b), -1).numpy()
    return out


def motion_part(px: np.ndarray, present: np.ndarray, prev_px=None, prev_present=None) -> np.ndarray:
    """(N,2,21,3) -> (N, 7). `prev_*` give the frame before px[0] (for streaming use)."""
    N = len(px)
    out = np.zeros((N, MOTION_DIM), np.float32)
    pp = present.astype(np.float32)
    hs = np.linalg.norm(px[:, :, 9] - px[:, :, 0], axis=-1)                    # (N,2)
    msize = np.maximum((hs * pp).sum(1) / np.maximum(pp.sum(1), 1), 1e-6)      # (N,)
    for t in range(N):
        if t == 0:
            if prev_px is None:
                q, qp = None, None
            else:
                q, qp = prev_px, prev_present
        else:
            q, qp = px[t - 1], present[t - 1]
        for h in range(2):
            if q is not None and present[t, h] and qp[h]:
                v = (px[t, h, 0, :2] - q[h, 0, :2]) / msize[t]
                out[t, 2 * h:2 * h + 2] = np.clip(v, -2, 2)
        if q is not None:
            both = present[t] & qp
            if both.any():
                d = np.linalg.norm(px[t][both] - q[both], axis=-1).mean() / msize[t]
                out[t, 4] = min(d, 2.0)
        out[t, 5:7] = present[t]
    return out


def sequence_features(trunk, temperature, px: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Full (T, 351) feature matrix for one video."""
    return np.concatenate([static_part(trunk, temperature, px, present), motion_part(px, present)], axis=1)


def to_px(lm: np.ndarray, wh: np.ndarray) -> np.ndarray:
    """raw normalised coords (...,21,3) + (w,h) -> aspect-corrected."""
    w, h = float(wh[0]), float(wh[1])
    return (lm * np.array([w, h, w], np.float32)).astype(np.float32)
