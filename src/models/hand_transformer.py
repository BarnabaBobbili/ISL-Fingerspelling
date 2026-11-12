"""
src/models/hand_transformer.py
================================
HandTransformer: Lightweight Transformer for ISL static fingerspelling.

MOTIVATION:
  This model tests whether self-attention (without explicit graph edges)
  can learn finger relationships from the enriched 6-dim node features.
  Unlike GAT, the Transformer has no inductive bias about finger connectivity —
  it must learn which landmarks to attend to from data alone.

  This is inspired by SPOTER (Sign Pose-Based Transformer) and SAM-SLR,
  which achieve strong results on sign language recognition without explicit
  graph structure.

INPUT (per hand): 21 landmark tokens, each with 6-dim enriched features
  [x, y, z, dist_to_wrist, angle_at_joint, dist_to_fingertip]

ARCHITECTURE (per hand):
  Linear(6 → 64) + Learnable Positional Encoding (21 positions)
  → Transformer Encoder (3 layers, d_model=64, 4 heads, ff=128)
  → Mean Pooling over 21 tokens
  → hand_embed (64,)

DUAL-HAND FUSION:
  right_embed (64) + left_embed (64)
  + right_geo (10) + left_geo (10) + presence (2) = 150
  → MLP → 26-class output

WHY MEAN POOL (not CLS token):
  With only 21 tokens and limited data, a learned CLS token adds parameters.
  Mean pooling is simpler and equally effective for small sequences.

Run self-test:
  python src/models/hand_transformer.py --self-test
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import NUM_LANDMARKS, NUM_STATIC_CLASSES, StaticConfig

# Transformer hyperparameters
D_MODEL    = 64
NUM_HEADS  = 4
NUM_LAYERS = 3
FF_DIM     = 128
GLOBAL_GEO_DIM = 10


class HandTransformerEncoder(nn.Module):
    """
    Encodes one hand's 21 landmark tokens into a fixed-size embedding.

    Parameters
    ----------
    in_features : int   — per-node input dim (6 for enriched features)
    d_model     : int   — Transformer model dim
    num_heads   : int   — multi-head attention heads
    num_layers  : int   — number of Transformer encoder layers
    ff_dim      : int   — feed-forward expansion dim
    dropout     : float — dropout
    """
    def __init__(self, in_features: int = 6, d_model: int = D_MODEL,
                 num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS,
                 ff_dim: int = FF_DIM, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Project input features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional encoding for 21 landmark positions
        self.pos_embed = nn.Parameter(torch.zeros(NUM_LANDMARKS, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, T, D) convention
            norm_first=True,    # Pre-LN (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, 21, in_features) — enriched node features
        present : (B,) bool            — hand presence mask

        Returns
        -------
        (B, d_model) — hand embedding, zeros for absent hands
        """
        B = x.shape[0]

        # Project + add positional encoding
        h = self.input_proj(x)                          # (B, 21, d_model)
        h = h + self.pos_embed.unsqueeze(0)             # broadcast over batch

        # Transformer encoding
        h = self.transformer(h)                         # (B, 21, d_model)
        h = self.norm(h)

        # Mean pooling over 21 tokens
        embed = h.mean(dim=1)                           # (B, d_model)

        # Zero out absent hands
        embed = embed * present.float().unsqueeze(1)
        return embed


class HandTransformer(nn.Module):
    """
    Dual-hand Transformer classifier for ISL static fingerspelling.

    Input shapes (per forward call):
      right_lm      : (B, 21, 6)  — enriched node features
      left_lm       : (B, 21, 6)  — enriched node features
      right_present : (B,) bool
      left_present  : (B,) bool
      right_geo     : (B, 10)     — global inter-fingertip distances
      left_geo      : (B, 10)     — global inter-fingertip distances

    Output:
      logits : (B, 26)
    """
    def __init__(self, cfg: StaticConfig = None,
                 d_model: int = D_MODEL, num_heads: int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS, ff_dim: int = FF_DIM):
        super().__init__()
        if cfg is None:
            cfg = StaticConfig()
        self.cfg = cfg
        self.d_model = d_model

        self.right_encoder = HandTransformerEncoder(
            in_features=cfg.in_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=cfg.dropout,
        )
        self.left_encoder = HandTransformerEncoder(
            in_features=cfg.in_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=cfg.dropout,
        )

        # Fusion MLP
        # right_embed (d_model) + left_embed (d_model)
        # + right_geo (10) + left_geo (10) + presence (2)
        fusion_in = d_model * 2 + cfg.global_geo_dim * 2 + 2
        fusion_hidden = max(fusion_in * 2, 128)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(fusion_hidden // 2, cfg.num_classes),
        )

    def forward(
        self,
        right_lm:      torch.Tensor,            # (B, 21, 6)
        left_lm:       torch.Tensor,            # (B, 21, 6)
        right_present: torch.Tensor,            # (B,) bool
        left_present:  torch.Tensor,            # (B,) bool
        right_geo:     torch.Tensor = None,     # (B, 10)
        left_geo:      torch.Tensor = None,     # (B, 10)
    ) -> torch.Tensor:
        B      = right_lm.shape[0]
        device = right_lm.device

        if right_geo is None:
            right_geo = torch.zeros(B, self.cfg.global_geo_dim, device=device)
        if left_geo is None:
            left_geo  = torch.zeros(B, self.cfg.global_geo_dim, device=device)

        r_embed = self.right_encoder(right_lm, right_present)
        l_embed = self.left_encoder(left_lm,   left_present)

        pf    = torch.stack([right_present.float(), left_present.float()], dim=1)
        fused = torch.cat([r_embed, l_embed, right_geo, left_geo, pf], dim=1)
        return self.fusion(fused)


def build_hand_transformer(cfg: StaticConfig = None) -> HandTransformer:
    """Convenience constructor."""
    return HandTransformer(cfg or StaticConfig())


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
def _self_test():
    print("Running HandTransformer self-tests...")
    cfg   = StaticConfig()
    model = build_hand_transformer(cfg)
    model.eval()

    B = 8
    right_lm      = torch.randn(B, NUM_LANDMARKS, cfg.in_features)
    left_lm       = torch.randn(B, NUM_LANDMARKS, cfg.in_features)
    right_present = torch.ones(B, dtype=torch.bool)
    left_present  = torch.ones(B, dtype=torch.bool)
    right_geo     = torch.rand(B, cfg.global_geo_dim)
    left_geo      = torch.rand(B, cfg.global_geo_dim)

    # Test 1: Output shape
    logits = model(right_lm, left_lm, right_present, left_present, right_geo, left_geo)
    assert logits.shape == (B, NUM_STATIC_CLASSES), \
        f"Expected ({B}, {NUM_STATIC_CLASSES}), got {logits.shape}"
    print(f"  ✓ Output shape: {logits.shape}")

    # Test 2: Absent hand → different output
    right_absent    = torch.zeros(B, dtype=torch.bool)
    logits_no_right = model(right_lm, left_lm, right_absent, left_present, right_geo, left_geo)
    assert not torch.allclose(logits, logits_no_right)
    print("  ✓ Presence flags affect output")

    # Test 3: Fallback (no geo)
    logits_no_geo = model(right_lm, left_lm, right_present, left_present)
    assert logits_no_geo.shape == (B, NUM_STATIC_CLASSES)
    print("  ✓ Fallback (no geo) works")

    # Test 4: Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Parameters: {n_params:,}")
    assert n_params < 2_000_000, f"Too many params: {n_params:,}"

    # Test 5: Gradient flow
    model.train()
    logits2 = model(right_lm, left_lm, right_present, left_present, right_geo, left_geo)
    logits2.mean().backward()
    print("  ✓ Gradient flows")

    print("\nAll HandTransformer self-tests PASSED ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
