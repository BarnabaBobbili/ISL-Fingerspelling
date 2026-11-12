"""
src/models/geo_mlp.py
======================
GeoMLP: Pure geometric feature MLP for ISL static fingerspelling.

MOTIVATION:
  This model tests the hypothesis that geometric features alone (distances +
  angles, NO raw coordinates, NO graph structure) are sufficient to distinguish
  all 26 ISL letters. If this simple model performs well, it validates the
  feature engineering as the key contribution — not the architecture.

INPUT (per hand): 35-dimensional geometric feature vector
  - 10 inter-fingertip distances
  -  5 fingertip-to-wrist distances
  -  5 fingertip-to-MCP (curl) distances
  - 15 joint angles (normalized to [0,1])

DUAL-HAND INPUT:
  right_geo  : (B, 35)
  left_geo   : (B, 35)
  presence   : (B, 2)
  Total      : (B, 72)

ARCHITECTURE:
  Linear(72 → 256) → BN → GELU → Dropout
  Linear(256 → 256) → BN → GELU → Dropout
  Linear(256 → 128) → BN → GELU → Dropout
  Linear(128 → 26)

WHY NO RAW COORDINATES:
  This is a deliberately minimal model to isolate the effect of geometry.
  Raw coordinates add camera-pose-dependent noise; excluding them makes the
  model maximally robust to domain gap (dataset images vs. webcam).

Run self-test:
  python src/models/geo_mlp.py --self-test
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import NUM_STATIC_CLASSES, StaticConfig

# Feature dims per hand from compute_geo_vector()
GEO_DIM_PER_HAND = 35   # 10 inter-fingertip + 5 tip-wrist + 5 curl + 15 angles
DUAL_GEO_DIM = GEO_DIM_PER_HAND * 2 + 2   # both hands + presence flags = 72


class GeoMLP(nn.Module):
    """
    Pure geometric-feature MLP for ISL static fingerspelling classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (26 for A-Z).
    hidden_dim  : int
        Width of hidden layers.
    dropout     : float
        Dropout probability.
    """
    def __init__(self, num_classes: int = NUM_STATIC_CLASSES,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(DUAL_GEO_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 3
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # Output
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        right_geo:     torch.Tensor,   # (B, 35)
        left_geo:      torch.Tensor,   # (B, 35)
        right_present: torch.Tensor,   # (B,) bool
        left_present:  torch.Tensor,   # (B,) bool
    ) -> torch.Tensor:
        """Returns logits (B, num_classes)."""
        # Zero out absent hand's geo features so they contribute nothing
        rg = right_geo * right_present.float().unsqueeze(1)
        lg = left_geo  * left_present.float().unsqueeze(1)

        pf = torch.stack([right_present.float(), left_present.float()], dim=1)
        x  = torch.cat([rg, lg, pf], dim=1)   # (B, 72)
        return self.net(x)                      # (B, 26)


def build_geo_mlp(num_classes: int = NUM_STATIC_CLASSES,
                  hidden_dim: int = 256, dropout: float = 0.3) -> GeoMLP:
    """Convenience constructor."""
    return GeoMLP(num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
def _self_test():
    print("Running GeoMLP self-tests...")
    model = build_geo_mlp()
    model.eval()

    B = 16
    right_geo     = torch.rand(B, GEO_DIM_PER_HAND)
    left_geo      = torch.rand(B, GEO_DIM_PER_HAND)
    right_present = torch.ones(B, dtype=torch.bool)
    left_present  = torch.ones(B, dtype=torch.bool)

    # Test 1: Output shape
    logits = model(right_geo, left_geo, right_present, left_present)
    assert logits.shape == (B, NUM_STATIC_CLASSES), \
        f"Expected ({B}, {NUM_STATIC_CLASSES}), got {logits.shape}"
    print(f"  ✓ Output shape: {logits.shape}")

    # Test 2: Absent hand → different output
    right_absent = torch.zeros(B, dtype=torch.bool)
    logits_no_r  = model(right_geo, left_geo, right_absent, left_present)
    assert not torch.allclose(logits, logits_no_r)
    print("  ✓ Presence flags affect output")

    # Test 3: Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Parameters: {n_params:,}  (very lightweight)")

    # Test 4: Gradient flow
    model.train()
    logits2 = model(right_geo, left_geo, right_present, left_present)
    logits2.mean().backward()
    print("  ✓ Gradient flows")

    print("\nAll GeoMLP self-tests PASSED ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
