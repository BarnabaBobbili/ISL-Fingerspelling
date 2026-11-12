"""
src/pipeline/classifier.py
==========================
Calibrated static-letter classifier (DH-GAT v2) for live frames.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.models.dh_gat_v2 import DHGATv2  # noqa: E402
from src.pipeline.hands import to_px  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]

# Checkpoint used by the app. Models whose file name ends in "_r" were trained on landmarks
# extracted WITH second-hand recovery and must be fed by HandTracker(recover=True).
DEFAULT_CKPT = ROOT / "checkpoints" / "dh_gat_v2_main_r.pt"


class StaticClassifier:
    def __init__(self, ckpt: Path = None, device: str = "cpu"):
        ckpt = Path(ckpt or DEFAULT_CKPT)
        self.recover = ckpt.stem.endswith("_r")          # which tracker mode this model expects
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model = DHGATv2(**ck["cfg"])
        self.model.load_state_dict(ck["model_state"])
        self.model.eval().to(device)
        self.temperature = float(ck.get("temperature", 1.0))
        self.device = device

    @torch.no_grad()
    def predict(self, rec: dict):
        """rec from HandTracker -> probs (26,) float32 or None when no hand is visible."""
        if not rec["present"].any():
            return None
        lm = torch.from_numpy(to_px(rec)).float()[None]
        pr = torch.from_numpy(rec["present"])[None]
        logits = self.model.forward_raw(lm.to(self.device), pr.to(self.device))
        return torch.softmax(logits / self.temperature, -1)[0].cpu().numpy()

    @torch.no_grad()
    def predict_batch(self, lm_px: np.ndarray, present: np.ndarray, bs: int = 512) -> np.ndarray:
        """lm_px (N,2,21,3) aspect-corrected, present (N,2) -> probs (N,26)."""
        out = []
        for i in range(0, len(lm_px), bs):
            lg = self.model.forward_raw(torch.from_numpy(lm_px[i:i + bs]).float(),
                                        torch.from_numpy(present[i:i + bs]))
            out.append(torch.softmax(lg / self.temperature, -1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 26), np.float32)
