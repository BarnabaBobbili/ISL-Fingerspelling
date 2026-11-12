"""
src/models/cont_v2.py
=====================
Sequence models for CONTINUOUS ISL fingerspelling (CTC), operating on the per-frame features
of src/pipeline/features.py (351-d). Two encoders share the same input stack and CTC head so
the comparison is fair:

  bilstm     Linear -> 2-layer BiLSTM -> CTC            (baseline)
  conformer  Linear -> 4 Conformer blocks -> CTC        (main model; reuses ConformerBlock)

Improvements over the v1 models (see docs):
  * input = trunk embedding + class posteriors + raw two-hand geometry + motion
  * intermediate CTC loss after the middle Conformer block (helps small data)
  * smaller convolution kernel (15 instead of 31) matching ~0.5 s letters at 30 fps
  * feature-level SpecAugment, time-stretch, noise (training script)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.conformer_ctc import ConformerBlock  # noqa: E402
from src.pipeline.features import FEAT_DIM  # noqa: E402

NUM_CLASSES = 27   # 26 letters + CTC blank (index 26)


class SeqCTC(nn.Module):
    def __init__(self, kind: str = "conformer", in_dim: int = FEAT_DIM, d: int = 144, layers: int = 4,
                 heads: int = 4, kernel: int = 15, dropout: float = 0.15, hidden: int = 192,
                 inter_ctc: bool = True):
        super().__init__()
        self.cfg = dict(kind=kind, in_dim=in_dim, d=d, layers=layers, heads=heads, kernel=kernel,
                        dropout=dropout, hidden=hidden, inter_ctc=inter_ctc)
        self.kind = kind
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        if kind == "bilstm":
            self.rnn = nn.LSTM(d, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, NUM_CLASSES))
        else:
            self.blocks = nn.ModuleList([ConformerBlock(d_model=d, num_heads=heads, conv_kernel=kernel,
                                                        ff_expansion=4, dropout=dropout) for _ in range(layers)])
            self.head = nn.Linear(d, NUM_CLASSES)
            self.inter_ctc = inter_ctc
            if inter_ctc:
                self.inter_head = nn.Linear(d, NUM_CLASSES)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x (B,T,F), lengths (B,) -> log_probs (T,B,C), inter_log_probs (T,B,C) | None"""
        h = self.proj(self.in_norm(x))
        inter = None
        if self.kind == "bilstm":
            packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            lp = torch.log_softmax(self.head(out), -1).transpose(0, 1)
        else:
            T = x.shape[1]
            pad = torch.arange(T, device=x.device)[None] >= lengths[:, None]
            for i, blk in enumerate(self.blocks):
                h = blk(h, key_padding_mask=pad)
                if self.inter_ctc and self.training and i == len(self.blocks) // 2 - 1:
                    inter = torch.log_softmax(self.inter_head(h), -1).transpose(0, 1)
            lp = torch.log_softmax(self.head(h), -1).transpose(0, 1)
        return lp, inter


def build(kind: str, **kw) -> SeqCTC:
    return SeqCTC(kind=kind, **kw)


if __name__ == "__main__":
    for k in ("bilstm", "conformer"):
        m = SeqCTC(k)
        x = torch.randn(3, 50, FEAT_DIM)
        lp, inter = m.train()(x, torch.tensor([50, 40, 30]))
        print(k, sum(p.numel() for p in m.parameters()), lp.shape, None if inter is None else inter.shape)
