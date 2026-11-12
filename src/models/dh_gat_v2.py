"""
src/models/dh_gat_v2.py
=======================
DH-GAT v2: Dual-Hand Graph Attention network that SEES BOTH HANDS IN ONE FRAME.

WHY v2 (findings from the Phase-1 benchmark)
  v1 normalised each hand separately (wrist-centred), so a letter such as
  "right index touches left palm" looked the same as "right index far from left
  hand". v2 keeps a shared coordinate frame and adds explicit inter-hand features.

GRAPH
  42 nodes = 2 hands x 21 MediaPipe joints. Layers alternate:
    * INTRA-hand attention  - restricted to the skeleton (nodes within 2 bones),
                              i.e. a graph-attention layer over the hand graph
    * CROSS-hand attention  - every present node attends to every present node
                              (learns which finger touches which part of the other hand)
  Absent hands are masked out (a node can always attend to itself).

NODE FEATURES (11 per joint, see featurize())
  g(3)      position in the shared frame (centre = mean wrist, unit = mean hand size)
  l(3)      position relative to own wrist, scaled by own hand size  (v1-style)
  b(3)+1    unit vector + length of the bone to the parent joint
  m(1)      distance to the nearest joint of the OTHER hand ("contact" cue)
GLOBAL FEATURES (4): both-hands flag + wrist-to-wrist vector (in hand sizes)

Hands are slotted by wrist-x (image-left first), NOT by MediaPipe's handedness
label, because that label flips between mirrored webcam and normal video.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

PARENT = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
NODE_DIM = 11
GLOB_DIM = 4


# ─────────────────────────────────────────────────────────────────────────────
def featurize(lm: torch.Tensor, present: torch.Tensor):
    """
    lm      : (B, 2, 21, 3) raw landmarks in ASPECT-CORRECTED units (x*W, y*H, z*W)
    present : (B, 2) bool
    returns : node (B, 2, 21, 11), glob (B, 4)
    """
    pw = present.float()
    n = pw.sum(1).clamp(min=1.0)                                       # (B,)
    wrist = lm[:, :, 0]                                                # (B,2,3)
    hs = (lm[:, :, 9] - wrist).norm(dim=-1).clamp(min=1e-6)            # (B,2)
    center = (wrist * pw[..., None]).sum(1) / n[:, None]               # (B,3)
    hsize = ((hs * pw).sum(1) / n).clamp(min=1e-6)                     # (B,)

    g = (lm - center[:, None, None, :]) / hsize[:, None, None, None]
    local = (lm - wrist[:, :, None, :]) / hs[:, :, None, None]
    vec = lm - lm[:, :, PARENT]
    blen = vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    unit = vec / blen
    blen = blen / hs[:, :, None, None]

    d = torch.cdist(g[:, 0], g[:, 1])                                  # (B,21,21)
    both = (present[:, 0] & present[:, 1]).float()                     # (B,)
    m0 = d.min(2).values * both[:, None]
    m1 = d.min(1).values * both[:, None]
    mind = torch.stack([m0, m1], 1)[..., None]                         # (B,2,21,1)

    node = torch.cat([g, local, unit, blen, mind], dim=-1) * pw[:, :, None, None]
    wv = (wrist[:, 1] - wrist[:, 0]) / hsize[:, None] * both[:, None]
    glob = torch.cat([both[:, None], wv], dim=1)
    return node, glob


def _allow_masks():
    """(42,42) bool: intra-hand (<=2 hops, same hand) and cross (all)."""
    A = torch.eye(21)
    for i, j in HAND_EDGES:
        A[i, j] = A[j, i] = 1
    A2 = ((A @ A + A) > 0)
    intra = torch.zeros(42, 42, dtype=torch.bool)
    intra[:21, :21] = A2
    intra[21:, 21:] = A2
    return intra, torch.ones(42, 42, dtype=torch.bool)


class DHGATv2(nn.Module):
    def __init__(self, d: int = 96, heads: int = 4, blocks: int = 2,
                 num_classes: int = 26, dropout: float = 0.1):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, blocks=blocks, num_classes=num_classes, dropout=dropout)
        self.heads = heads
        self.embed = nn.Sequential(nn.Linear(NODE_DIM, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
        self.joint_emb = nn.Embedding(21, d)
        self.slot_emb = nn.Embedding(2, d)
        self.glob = nn.Linear(GLOB_DIM, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(2 * blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, num_classes))
        intra, cross = _allow_masks()
        self.register_buffer("intra", intra, persistent=False)
        self.register_buffer("cross", cross, persistent=False)
        self.register_buffer("joint_idx", torch.arange(21).repeat(2), persistent=False)
        self.register_buffer("slot_idx", torch.arange(2).repeat_interleave(21), persistent=False)
        self.register_buffer("eye", torch.eye(42, dtype=torch.bool), persistent=False)

    def features(self, node, present, glob):
        """pooled embedding (B, 2d) = [masked mean | masked max] over present nodes."""
        B = node.shape[0]
        x = self.embed(node.reshape(B, 42, NODE_DIM))
        x = x + self.joint_emb(self.joint_idx) + self.slot_emb(self.slot_idx) + self.glob(glob)[:, None]
        tok = present.repeat_interleave(21, dim=1)                       # (B,42) bool
        for i, layer in enumerate(self.layers):
            base = self.intra if i % 2 == 0 else self.cross
            allow = (base[None] & tok[:, None, :]) | self.eye[None]      # (B,42,42)
            mask = (~allow).repeat_interleave(self.heads, dim=0)         # True = blocked
            x = layer(x, src_mask=mask)
        x = self.norm(x)
        tf = tok.float()[..., None]
        mean = (x * tf).sum(1) / tf.sum(1).clamp(min=1.0)
        mx = x.masked_fill(~tok[..., None], -1e4).max(1).values
        mx = torch.where(tok.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=-1)

    def forward(self, node, present, glob):
        return self.head(self.features(node, present, glob))

    def forward_raw(self, lm, present):
        node, glob = featurize(lm, present)
        return self(node, present, glob)


def build_dhgat_v2(**kw) -> DHGATv2:
    return DHGATv2(**kw)


if __name__ == "__main__":
    m = DHGATv2()
    print("params:", sum(p.numel() for p in m.parameters()))
    lm = torch.rand(4, 2, 21, 3) * 100
    pr = torch.tensor([[1, 1], [1, 0], [0, 0], [1, 1]]).bool()
    out = m.forward_raw(lm, pr)
    print(out.shape, torch.isfinite(out).all().item())
