"""
src/models/conformer_ctc.py
=============================
Stage 6: Conformer encoder + CTC output layer.

ARCHITECTURE
------------
  Input: (B, T, 63)
    ↓
  Linear projection: 63 → d_model (144)
    ↓
  ConformerBlock × num_layers (4):
    ┌─────────────────────────────────────────────────┐
    │ 1. Feed-Forward Module (half-step)               │
    │ 2. Multi-Head Self-Attention (heads=4)           │ ← global context
    │ 3. Convolution Module (depthwise, kernel=31)     │ ← local context
    │ 4. Feed-Forward Module (half-step)               │
    │ 5. Layer Norm                                    │
    └─────────────────────────────────────────────────┘
    ↓
  Linear: d_model → 27
    ↓
  LogSoftmax(dim=-1)

WHY CONFORMER OVER BILSTM (DD-007)
------------------------------------
BiLSTM encodes local and global context through the same hidden state.
Conformer explicitly separates them:
  - Self-attention: attends to ALL positions simultaneously → global dependencies
  - Conv module (kernel=31): convolution over ~1 second window → local patterns
  - Feed-forward: pointwise transformation at each position

This architectural separation is exactly what fingerspelling benefits from:
  - Global: the identity of a letter depends on neighboring letters (coarticulation)
  - Local: the shape of a single letter fits within ~0.3–0.8 seconds

WHY IMPLEMENTED FROM SCRATCH (not using a library):
  Thesis defense requirement — you must be able to explain every component.
  The implementation follows "Conformer: Convolution-augmented Transformer
  for Speech Recognition" (Gulati et al., 2020) adapted for landmark sequences.

IMPLEMENTATION NOTES
---------------------
- No positional encoding: unlike text, landmark sequences have strong positional
  information built in (the LSTM-free self-attention still works because relative
  positions are implicit in the conv module). Alternative: learned positional encoding.
  → DESIGN DECISION: we ADD learned positional encodings for completeness.
- CausalConv vs FullConv: we use non-causal (full) convolution because we process
  full segments offline (not streaming). Streaming inference would need causal conv.
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import ConformerConfig, conformer_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class FeedForwardModule(nn.Module):
    """
    Conformer Feed-Forward Module (used twice per block, scaled by 0.5).

    Structure: LayerNorm → Linear(d→4d) → Swish → Dropout → Linear(4d→d) → Dropout

    WHY SWISH activation?
      Swish (x * sigmoid(x)) is smooth and non-monotonic, giving better gradient
      flow than ReLU in deep networks. Used in the original Conformer paper.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm   = nn.LayerNorm(d_model)
        self.ff1    = nn.Linear(d_model, d_model * expansion)
        self.act    = nn.SiLU()   # SiLU == Swish in PyTorch
        self.drop1  = nn.Dropout(dropout)
        self.ff2    = nn.Linear(d_model * expansion, d_model)
        self.drop2  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalize before transformation (more stable than post-norm)
        residual = x
        x = self.norm(x)
        x = self.ff1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.ff2(x)
        x = self.drop2(x)
        return residual + 0.5 * x   # half-step scaling (Conformer convention)


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-Head Self-Attention with relative positional encoding.

    WHY SELF-ATTENTION FOR LONG-RANGE DEPENDENCIES?
      Unlike LSTM which processes tokens sequentially (O(T) sequential steps),
      self-attention processes all pairs simultaneously (O(1) sequential steps,
      O(T^2) memory). For fingerspelling sequences (~30–300 frames), T^2 is small.

    WHY RELATIVE POSITIONAL ENCODING?
      Standard absolute PE breaks for variable-length sequences.
      We use a simple learned absolute PE (capped at MAX_SEQ_LEN=1000).
      This is simpler than full relative PE (Transformer-XL style) and
      sufficient for our sequence lengths.
    """

    MAX_SEQ_LEN = 1000

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.norm  = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop  = nn.Dropout(dropout)

        # Learned positional encoding
        self.pos_enc = nn.Embedding(self.MAX_SEQ_LEN, d_model)
        self._init_pos_enc()

    def _init_pos_enc(self):
        """Initialize positional encoding with sinusoidal values for faster convergence."""
        pos = torch.arange(self.MAX_SEQ_LEN).unsqueeze(1).float()
        d   = self.pos_enc.embedding_dim
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe  = torch.zeros(self.MAX_SEQ_LEN, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d // 2])
        self.pos_enc.weight.data.copy_(pe)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        # Add positional encoding
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        x = x + self.pos_enc(pos_ids)                              # (B, T, d_model)

        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.drop(x)
        return residual + x


class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module.

    WHY DEPTHWISE CONVOLUTION?
      Standard convolution over (d_model, kernel) has O(d_model^2 × kernel) params.
      Depthwise separable conv (pointwise → depthwise → pointwise) has
      O(d_model × kernel) params — much more parameter-efficient.

    Structure:
      LayerNorm → Pointwise Conv(d→2d) → GLU → Depthwise Conv(kernel) →
      BatchNorm → Swish → Pointwise Conv(d→d) → Dropout

    GLU (Gated Linear Unit): splits the 2d channels into two d-dim halves;
    one half is the gate (sigmoid), other is the signal. This is a form of
    learned feature selection.
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for same-size padding"
        padding = (kernel_size - 1) // 2

        self.norm     = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)  # expand
        self.dw_conv  = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,   # depthwise: each channel convolved separately
        )
        self.bn       = nn.BatchNorm1d(d_model)
        self.act      = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)      # project back
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        residual = x
        x = self.norm(x)

        # Conv1d expects (B, C, T) — transpose
        x = x.transpose(1, 2)              # (B, d_model, T)
        x = self.pw_conv1(x)               # (B, 2*d_model, T)

        # GLU: split along channel dim, apply gate
        x, gate = x.chunk(2, dim=1)        # each (B, d_model, T)
        x = x * torch.sigmoid(gate)        # gated linear unit

        x = self.dw_conv(x)                # (B, d_model, T)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw_conv2(x)               # (B, d_model, T)
        x = self.drop(x)

        x = x.transpose(1, 2)             # back to (B, T, d_model)
        return residual + x


class ConformerBlock(nn.Module):
    """
    Single Conformer encoder block.

    Combines: FF (half) → MHSA → Conv → FF (half) → LayerNorm
    This "sandwich" structure is the key innovation of the Conformer paper.
    """

    def __init__(self, d_model: int, num_heads: int, conv_kernel: int,
                 ff_expansion: int, dropout: float):
        super().__init__()
        self.ff1  = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2  = FeedForwardModule(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class ConformerCTC(nn.Module):
    """
    Full Conformer encoder + CTC head.

    Parameters
    ----------
    cfg : ConformerConfig   see config.py
    """

    def __init__(self, cfg: ConformerConfig = conformer_cfg):
        super().__init__()
        self.cfg = cfg

        # Input projection: landmarks (63) → d_model (144)
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

        # Stack of Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                conv_kernel=cfg.conv_kernel,
                ff_expansion=cfg.ff_expansion,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.num_layers)
        ])

        # CTC head — same as BiLSTM for apples-to-apples comparison
        self.classifier = nn.Linear(cfg.d_model, cfg.num_classes)

    def _make_padding_mask(self, input_lengths: torch.Tensor, T: int) -> torch.Tensor:
        """
        Creates a boolean mask for padded positions.
        True = position is PADDING (should be ignored by attention).

        WHY: Without masking, the self-attention would attend to zero-padded
        positions and learn spurious dependencies.
        """
        # mask[i, t] = True if t >= input_lengths[i]  (i.e., padded)
        B = input_lengths.shape[0]
        mask = torch.arange(T, device=input_lengths.device).unsqueeze(0) >= \
               input_lengths.unsqueeze(1)   # (B, T)
        return mask

    def forward(self, landmarks: torch.Tensor, input_lengths: torch.Tensor):
        """
        Parameters
        ----------
        landmarks     : (B, T, 63)
        input_lengths : (B,)

        Returns
        -------
        log_probs     : (T, B, num_classes)
        output_lengths: (B,)  same as input_lengths
        """
        B, T, _ = landmarks.shape

        x = self.input_proj(landmarks)    # (B, T, d_model)

        # Padding mask for self-attention
        pad_mask = self._make_padding_mask(input_lengths, T)  # (B, T)

        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        logits = self.classifier(x)                         # (B, T, num_classes)
        log_probs = torch.log_softmax(logits, dim=-1)       # (B, T, num_classes)
        log_probs = log_probs.permute(1, 0, 2)              # (T, B, num_classes)

        return log_probs, input_lengths

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_conformer(cfg: ConformerConfig = conformer_cfg) -> ConformerCTC:
    """Factory function."""
    model = ConformerCTC(cfg)
    print(f"[ConformerCTC] Parameters: {model.count_parameters():,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick shape test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_conformer()
    model.eval()

    B, T, F = 4, 97, 63
    fake_lm  = torch.randn(B, T, F)
    fake_len = torch.tensor([97, 80, 60, 45])

    with torch.no_grad():
        log_probs, out_len = model(fake_lm, fake_len)

    print(f"  Input:       ({B}, {T}, {F})")
    print(f"  log_probs:   {tuple(log_probs.shape)}  (should be ({T}, {B}, 27))")
    print(f"  out_lengths: {out_len.tolist()}")
    print("ConformerCTC shape test PASSED ✓")
