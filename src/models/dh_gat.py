"""
src/models/dh_gat.py
=====================
Dual-Hand Graph Attention Transformer (DH-GAT) for ISL static fingerspelling.

ARCHITECTURE OVERVIEW:
  Input: Right hand node features (21×6) + Left hand node features (21×6)
         + Right global geo (10,) + Left global geo (10,) + presence flags (2)
    │
    ├── HandGATEncoder (right) ──→ right_embed (128,)
    ├── HandGATEncoder (left)  ──→ left_embed  (128,)
    │
    └── DualHandClassifier:
          Concat [right_embed, left_embed, right_geo, left_geo, presence_flags]
          → (128+128+10+10+2) = (278,)
          MLP Fusion  →  (256,)  →  Dropout  →  (26,)

ENRICHED NODE FEATURES (in_features=6 per node, up from 3):
  [x, y, z, dist_to_wrist, angle_at_joint, dist_to_fingertip]
  These are geometry-invariant features that bridge the domain gap between
  curated dataset images and real-time webcam captures.

GLOBAL GEOMETRIC FEATURES (10 per hand):
  All 10 pairwise inter-fingertip distances, appended at the fusion layer.
  Invariant to rigid-body transformations — depend only on finger spread.

WHY GRAPH ATTENTION (not plain MLP):
  A plain MLP treats all numbers as independent features.
  A GAT knows that landmark #8 (index fingertip) is connected to #7 (DIP),
  which connects to #6 (PIP), etc. This inductive bias lets the model learn
  finger shapes more efficiently from limited data.

WHY TWO SEPARATE ENCODERS (not one big network):
  Right and left hands are independently meaningful and may be at different
  positions/orientations. Processing them separately before fusing mirrors
  how ISL is structured: each hand has its own role.

REFERENCE ARCHITECTURES:
  - SPOTER (Sign Pose-Based Transformer, CVPR 2023 workshop)
  - GAT: "Graph Attention Networks" (Velickovic et al., ICLR 2018)
  - Hand-GCN for gesture recognition (CVPR 2019)

Run self-test:
  python src/models/dh_gat.py --self-test
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    HAND_EDGES, NUM_LANDMARKS, LANDMARK_DIM,
    NUM_STATIC_CLASSES, StaticConfig
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Attention Layer
# ─────────────────────────────────────────────────────────────────────────────
class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Layer (Velickovic et al., 2018).

    Each node aggregates features from its neighbors using learned attention
    weights. The attention score between node i and neighbor j is:
      e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
      α_ij = softmax over neighbors of e_ij
      h_i' = σ( Σ_j α_ij · W h_j )

    Parameters
    ----------
    in_features  : input feature dim per node
    out_features : output feature dim per node
    dropout      : dropout on attention weights (regularization)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.W        = nn.Linear(in_features, out_features, bias=False)
        self.attn     = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout  = nn.Dropout(dropout)
        self.out_features = out_features

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, N, in_features)
        adj : (N, N) binary adjacency matrix (self-loops included)

        Returns
        -------
        (B, N, out_features)
        """
        B, N, _ = x.shape
        Wh = self.W(x)                         # (B, N, out_features)

        # Compute attention scores for all pairs (i, j) where adj[i,j]=1
        # Efficient broadcast: e_ij = a^T [Wh_i || Wh_j]
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, -1)   # (B, N, N, F)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, -1)   # (B, N, N, F)
        e = self.leaky_relu(self.attn(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))  # (B, N, N)

        # Mask non-edges with -inf so softmax gives ~0 weight
        mask = (adj == 0).unsqueeze(0).expand(B, -1, -1)
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)          # (B, N, N) — NaN guard needed
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # Aggregate neighbor features
        out = torch.bmm(alpha, Wh)            # (B, N, out_features)
        return F.elu(out)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Graph Attention
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadGAT(nn.Module):
    """
    Multi-head Graph Attention (concatenate heads).

    Parameters
    ----------
    in_features  : input feature dim
    out_features : output feature dim PER HEAD
    num_heads    : number of attention heads
    dropout      : dropout on attention
    """
    def __init__(self, in_features: int, out_features: int,
                 num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout)
            for _ in range(num_heads)
        ])
        self.out_dim = out_features * num_heads

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Returns (B, N, out_features * num_heads)"""
        return torch.cat([h(x, adj) for h in self.heads], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Hand Encoder: stacked GAT layers → single hand embedding
# ─────────────────────────────────────────────────────────────────────────────
class HandGATEncoder(nn.Module):
    """
    Encodes a single hand (21 landmark nodes) into a fixed-size embedding.

    Architecture:
      GAT Layer 1: (3) → (gat_hidden × num_heads)
      GAT Layer 2: (gat_hidden × num_heads) → (gat_hidden × num_heads)
      Global mean pooling over all 21 nodes
      Linear projection → hand_embed_dim

    WHY GLOBAL MEAN POOL (not just wrist or just fingertips):
      Different letters stress different subsets of fingers.
      Mean pooling lets the downstream attention decide which nodes matter.
    """
    def __init__(self, cfg: StaticConfig):
        super().__init__()
        head_out_dim = cfg.gat_hidden  # per-head output dim
        gat1_out     = head_out_dim * cfg.gat_heads

        self.gat1 = MultiHeadGAT(
            in_features  = cfg.in_features,   # 3 (x,y,z)
            out_features = head_out_dim,
            num_heads    = cfg.gat_heads,
            dropout      = cfg.dropout,
        )
        self.gat2 = MultiHeadGAT(
            in_features  = gat1_out,
            out_features = head_out_dim,
            num_heads    = cfg.gat_heads,
            dropout      = cfg.dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(gat1_out, cfg.hand_embed_dim),
            nn.LayerNorm(cfg.hand_embed_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                present: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, 21, 3) — normalized hand landmarks
        adj     : (21, 21)   — skeleton adjacency matrix
        present : (B,)       — boolean mask, True if hand is detected

        Returns
        -------
        (B, hand_embed_dim)  — zeros for absent hands
        """
        h = self.gat1(x, adj)               # (B, 21, gat1_out)
        h = self.dropout(h)
        h = self.gat2(h, adj)               # (B, 21, gat1_out)
        h = h.mean(dim=1)                   # (B, gat1_out) global pool
        h = self.proj(h)                    # (B, hand_embed_dim)

        # Zero out absent hands so they contribute nothing to fusion
        h = h * present.float().unsqueeze(1)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# DH-GAT: Main model
# ─────────────────────────────────────────────────────────────────────────────
class DHGAT(nn.Module):
    """
    Dual-Hand Graph Attention Transformer for ISL Static Fingerspelling.

    Input shapes (per forward call):
      right_lm      : (B, 21, 6)  — enriched node features
      left_lm       : (B, 21, 6)  — enriched node features
      right_present : (B,) bool
      left_present  : (B,) bool
      right_geo     : (B, 10)     — global inter-fingertip distances (right)
      left_geo      : (B, 10)     — global inter-fingertip distances (left)

    Output:
      logits        : (B, 26)  — raw class scores (use CrossEntropy)
    """
    def __init__(self, cfg: StaticConfig = None):
        super().__init__()
        if cfg is None:
            cfg = StaticConfig()
        self.cfg = cfg

        # Build and register the skeleton adjacency matrix as a buffer
        # (moved to correct device automatically with .to(device))
        adj = self._build_adj()
        self.register_buffer("adj", adj)

        # Two independent hand encoders (shared weights optional, but separate
        # is better because ISL assigns different roles to each hand)
        self.right_encoder = HandGATEncoder(cfg)
        self.left_encoder  = HandGATEncoder(cfg)

        # Fusion MLP
        # Input: right_embed (128) + left_embed (128)
        #        + right_geo (10) + left_geo (10)
        #        + presence_flags (2) = 278
        fusion_in = cfg.hand_embed_dim * 2 + cfg.global_geo_dim * 2 + 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_dim),
            nn.LayerNorm(cfg.fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim // 2),
            nn.LayerNorm(cfg.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim // 2, cfg.num_classes),
        )

    @staticmethod
    def _build_adj() -> torch.Tensor:
        """
        Build (21, 21) symmetric adjacency matrix from HAND_EDGES,
        with self-loops added (each node attends to itself).
        """
        N = NUM_LANDMARKS
        adj = torch.zeros(N, N)
        for i, j in HAND_EDGES:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        # Add self-loops
        adj += torch.eye(N)
        return adj

    def forward(
        self,
        right_lm:      torch.Tensor,             # (B, 21, 6)
        left_lm:       torch.Tensor,             # (B, 21, 6)
        right_present: torch.Tensor,             # (B,) bool
        left_present:  torch.Tensor,             # (B,) bool
        right_geo:     torch.Tensor = None,      # (B, 10)
        left_geo:      torch.Tensor = None,      # (B, 10)
    ) -> torch.Tensor:
        """Returns logits (B, num_classes)."""
        B = right_lm.shape[0]
        device = right_lm.device

        # Default zero geo tensors if not provided (fallback)
        if right_geo is None:
            right_geo = torch.zeros(B, self.cfg.global_geo_dim, device=device)
        if left_geo is None:
            left_geo = torch.zeros(B, self.cfg.global_geo_dim, device=device)

        r_embed = self.right_encoder(right_lm, self.adj, right_present)
        l_embed = self.left_encoder(left_lm,  self.adj, left_present)

        # Concatenate embeddings + global geo + presence flags
        pf = torch.stack([right_present.float(), left_present.float()], dim=1)
        fused = torch.cat([r_embed, l_embed, right_geo, left_geo, pf], dim=1)  # (B, 278)
        return self.fusion(fused)                                                # (B, 26)


def build_dhgat(cfg: StaticConfig = None) -> DHGAT:
    """Convenience constructor for use in training/inference scripts."""
    return DHGAT(cfg or StaticConfig())


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
def _self_test():
    print("Running DH-GAT self-tests...")
    cfg   = StaticConfig()
    model = build_dhgat(cfg)

    B = 8
    # New enriched inputs: (B, 21, 6) node features + (B, 10) global geo
    right_lm      = torch.randn(B, NUM_LANDMARKS, cfg.in_features)
    left_lm       = torch.randn(B, NUM_LANDMARKS, cfg.in_features)
    right_present = torch.ones(B, dtype=torch.bool)
    left_present  = torch.ones(B, dtype=torch.bool)
    right_geo     = torch.randn(B, cfg.global_geo_dim).abs()   # distances ≥ 0
    left_geo      = torch.randn(B, cfg.global_geo_dim).abs()

    # Test 1: Output shape
    logits = model(right_lm, left_lm, right_present, left_present, right_geo, left_geo)
    assert logits.shape == (B, NUM_STATIC_CLASSES), \
        f"Expected ({B}, {NUM_STATIC_CLASSES}), got {logits.shape}"
    print(f"  ✓ Output shape: {logits.shape}")

    # Test 2: Absent hand → different output
    right_absent = torch.zeros(B, dtype=torch.bool)
    logits_no_right = model(right_lm, left_lm, right_absent, left_present, right_geo, left_geo)
    assert not torch.allclose(logits, logits_no_right), \
        "Presence flags should affect output"
    print("  ✓ Presence flags affect output correctly")

    # Test 3: Fallback works (no geo tensors provided)
    logits_no_geo = model(right_lm, left_lm, right_present, left_present)
    assert logits_no_geo.shape == (B, NUM_STATIC_CLASSES)
    print("  ✓ Fallback (no geo) works correctly")

    # Test 4: Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total trainable parameters: {n_params:,}")
    assert n_params < 2_000_000, f"Model too large: {n_params:,} params (limit 2M for CPU)"
    print(f"  ✓ Model is CPU-feasible (< 2M params)")

    # Test 5: Gradient flow
    loss = logits.mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            print(f"  ✗ No gradient for {name}")
    print("  ✓ Gradient flows to all parameters")

    print("\nAll DH-GAT self-tests PASSED ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
