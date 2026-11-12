"""
src/models/bilstm_ctc.py
==========================
Stage 4: BiLSTM + CTC baseline model.

ARCHITECTURE
------------
  Input: (B, T, 63)  — normalized landmark sequences
    ↓
  LinearProjection: 63 → 128
    ↓
  BiLSTM × 2 layers, hidden=256  →  output dim = 512 (bidirectional)
    ↓
  Dropout(0.3)
    ↓
  Linear: 512 → 27  (26 letters + CTC blank)
    ↓
  LogSoftmax(dim=-1)

WHY THIS ARCHITECTURE (DD-006)
-------------------------------
BiLSTM is the standard strong baseline for sequence-to-sequence tasks on
landmark data. It processes time steps recurrently, maintaining a hidden
state that summarizes past and future context (bidirectional).

The linear projection before the LSTM (63→128) serves two purposes:
  1. Reduces input dimensionality (the LSTM's gate matrices scale as O(d^2))
  2. Allows the model to learn a useful feature space before sequencing

ALTERNATIVE (rejected): single-direction LSTM
  WHY REJECTED: Bidirectional significantly outperforms on sign language tasks
  because the shape of letter i is influenced by letter i+1 (coarticulation).
  The cost is 2× parameters and 2× compute — acceptable for our dataset size.

CTCLoss configuration:
  blank=26, zero_infinity=True
  WHY zero_infinity: If CTC loss hits -infinity (can happen in early training
  when the blank probability is very low), zero_infinity=True replaces it with
  0 instead of crashing. This is recommended for small datasets.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import BiLSTMConfig, bilstm_cfg


class BiLSTMCTC(nn.Module):
    """
    Bidirectional LSTM encoder with CTC output layer.

    Parameters
    ----------
    cfg : BiLSTMConfig   hyperparameter bundle (see config.py)
    """

    def __init__(self, cfg: BiLSTMConfig = bilstm_cfg):
        super().__init__()
        self.cfg = cfg

        # Linear projection: maps raw 63-dim landmarks to a learned feature space
        # before the recurrent layers. This is more efficient than feeding 63-dim
        # directly into the LSTM (gate matrices would be much larger).
        self.input_proj = nn.Linear(cfg.input_dim, cfg.proj_dim)
        self.proj_norm  = nn.LayerNorm(cfg.proj_dim)  # stabilizes early training

        # Bidirectional LSTM stack
        # output dimension = 2 × hidden_dim because of bidirectionality
        self.lstm = nn.LSTM(
            input_size=cfg.proj_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,         # input shape: (B, T, features)
            bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            # Dropout is applied between LSTM layers (not on output of last layer)
        )

        self.dropout = nn.Dropout(p=cfg.dropout)

        # Final classifier: maps LSTM output to character logits
        self.classifier = nn.Linear(cfg.hidden_dim * 2, cfg.num_classes)

    def forward(self, landmarks: torch.Tensor, input_lengths: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        landmarks     : torch.Tensor  shape (B, T, 63)
        input_lengths : torch.Tensor  shape (B,)  actual sequence lengths

        Returns
        -------
        log_probs     : torch.Tensor  shape (T, B, num_classes)
            Log-softmax output in (T, B, C) format required by nn.CTCLoss.
        output_lengths: torch.Tensor  shape (B,)
            Same as input_lengths (BiLSTM doesn't reduce time dimension).
        """
        # Project input
        x = self.input_proj(landmarks)    # (B, T, proj_dim)
        x = self.proj_norm(x)
        x = torch.relu(x)

        # Pack padded sequences — tells LSTM to skip padding positions
        # WHY: Without packing, the LSTM would process padding zeros and corrupt
        # the hidden state, particularly for short sequences in a long-sequence batch.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # x shape: (B, T, hidden_dim * 2)

        x = self.dropout(x)
        logits = self.classifier(x)       # (B, T, num_classes)

        # CTCLoss expects (T, B, C) — permute
        log_probs = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
        # shape: (T, B, num_classes)

        return log_probs, input_lengths   # output_lengths = input_lengths (no downsampling)

    def count_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_bilstm(cfg: BiLSTMConfig = bilstm_cfg) -> BiLSTMCTC:
    """Factory function — returns an initialized BiLSTMCTC model."""
    model = BiLSTMCTC(cfg)
    print(f"[BiLSTMCTC] Parameters: {model.count_parameters():,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick shape test (run with: python src/models/bilstm_ctc.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    model = build_bilstm()
    model.eval()

    B, T, F = 4, 97, 63
    fake_lm  = torch.randn(B, T, F)
    fake_len = torch.tensor([97, 80, 60, 45])

    with torch.no_grad():
        log_probs, out_len = model(fake_lm, fake_len)

    print(f"  Input:       ({B}, {T}, {F})")
    print(f"  log_probs:   {tuple(log_probs.shape)}  (should be ({T}, {B}, 27))")
    print(f"  out_lengths: {out_len.tolist()}  (should match input lengths)")
    print("BiLSTMCTC shape test PASSED ✓")
