"""
src/training/train.py
=======================
Shared training loop for BiLSTM (Stage 4) and Conformer (Stage 6).

WHY ONE TRAINING SCRIPT FOR BOTH MODELS?
  Identical training procedure is the only way to ensure an apples-to-apples
  comparison between BiLSTM and Conformer. The same:
    - Data splits and batching
    - Loss function and hyperparameter schedule
    - Checkpoint / early-stopping criteria
    - Evaluation metric
  means any CER difference is due to architecture, not setup.

TRAINING LOOP DETAILS
---------------------
  1. CTC loss with blank=26, zero_infinity=True
  2. Adam optimizer + ReduceLROnPlateau on val CER
  3. Early stopping: stop if val CER doesn't improve for `patience` epochs
  4. Checkpoint: save model when val CER improves (best model)
  5. Logging: TensorBoard + CSV (one row per epoch)

USAGE
-----
  # Train BiLSTM baseline (Stage 4):
  python src/training/train.py --model bilstm

  # Train Conformer (Stage 6):
  python src/training/train.py --model conformer

  # Quick sanity run on 100 samples to verify pipeline:
  python src/training/train.py --model bilstm --subset 100 --epochs 5

  # Resume from checkpoint:
  python src/training/train.py --model bilstm --resume checkpoints/bilstm_best.pt

  # Evaluate only (no training):
  python src/training/train.py --model bilstm --eval-only --resume checkpoints/bilstm_best.pt
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    BiLSTMConfig, ConformerConfig, bilstm_cfg, conformer_cfg,
    BLANK_IDX, CKPT_DIR, LOG_DIR, RESULTS_DIR
)
from src.data.dataset_loader import get_dataloader, ISLDataset
from src.models.bilstm_ctc import build_bilstm
from src.models.conformer_ctc import build_conformer
from src.evaluation.decoder import ctc_greedy_decode
from src.evaluation.metrics import character_error_rate, evaluate_model
from src.evaluation.confusion_analysis import run_confusion_analysis


def get_device() -> str:
    """
    Returns 'cuda' if a GPU is available, else 'cpu'.
    Prints the device so it's visible in training logs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[train] Using GPU: {gpu_name}")
    else:
        print("[train] No GPU found — using CPU. Training will be slow.")
        print("        Expected time: BiLSTM ~30-60 min, Conformer ~2-4 hours")
    return device


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    ctc_loss: nn.CTCLoss,
    device: str,
    subset_limit: int = None,
) -> float:
    """
    Runs one training epoch and returns the mean loss.

    Parameters
    ----------
    model        : nn.Module
    loader       : DataLoader
    optimizer    : Adam
    ctc_loss     : nn.CTCLoss
    device       : str
    subset_limit : int or None  — stop after this many batches (for quick sanity runs)

    Returns
    -------
    float  mean loss over the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        if subset_limit and batch_idx >= subset_limit:
            break

        landmarks     = batch["landmarks"].to(device)      # (B, T, 63)
        labels        = batch["labels"].to(device)         # (B, L)
        input_lengths = batch["input_lengths"].to(device)  # (B,)
        label_lengths = batch["label_lengths"].to(device)  # (B,)

        optimizer.zero_grad()
        log_probs, out_lengths = model(landmarks, input_lengths)
        # log_probs: (T, B, C)

        # CTCLoss requires labels as a flat 1D tensor of class indices
        # (not padded — only the valid label chars)
        flat_labels = []
        for i in range(labels.shape[0]):
            L_i = label_lengths[i].item()
            flat_labels.append(labels[i, :L_i])
        flat_labels = torch.cat(flat_labels)  # (sum of L_i,)

        loss = ctc_loss(log_probs, flat_labels, out_lengths, label_lengths)

        loss.backward()
        # Gradient clipping — important for LSTM to prevent exploding gradients
        # Conformer is less prone but we apply it uniformly for consistency
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def validate(
    model: nn.Module,
    loader,
    ctc_loss: nn.CTCLoss,
    device: str,
) -> tuple:
    """
    Runs validation and returns (val_loss, val_CER).

    WHY BOTH LOSS AND CER FOR VALIDATION?
      Loss tells us if the model is learning at all.
      CER is the actual task metric (lower = better recognizer).
      They sometimes disagree — e.g. loss decreases but CER plateaus
      (the model is overfitting to common patterns).
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_hyps   = []
    all_refs   = []

    with torch.no_grad():
        for batch in loader:
            landmarks     = batch["landmarks"].to(device)
            labels        = batch["labels"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            log_probs, out_lengths = model(landmarks, input_lengths)

            # Loss
            flat_labels = torch.cat([
                labels[i, :label_lengths[i].item()]
                for i in range(labels.shape[0])
            ])
            loss = ctc_loss(log_probs, flat_labels, out_lengths, label_lengths)
            total_loss += loss.item()
            n_batches  += 1

            # Decode for CER
            hypotheses = ctc_greedy_decode(log_probs, out_lengths)
            all_hyps.extend(hypotheses)

            from src.training.config import IDX_TO_CHAR
            for i in range(labels.shape[0]):
                L_i = label_lengths[i].item()
                ref = "".join(IDX_TO_CHAR.get(idx.item(), "?")
                              for idx in labels[i, :L_i])
                all_refs.append(ref)

    val_loss = total_loss / n_batches if n_batches > 0 else float("inf")
    val_cer  = character_error_rate(
        [h.replace(" ", "") for h in all_hyps],
        [r.replace(" ", "") for r in all_refs],
    )
    return val_loss, val_cer, list(zip(all_hyps, all_refs))


def train(
    model_name: str,
    subset: int = None,
    max_epochs: int = None,
    resume: str = None,
    eval_only: bool = False,
):
    """
    Main training function.

    Parameters
    ----------
    model_name  : 'bilstm' or 'conformer'
    subset      : if set, use only this many batches per epoch (quick sanity run)
    max_epochs  : override config's max_epochs
    resume      : path to checkpoint to resume from
    eval_only   : skip training, run evaluation only
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Build model and optimizer
    if model_name == "bilstm":
        cfg   = bilstm_cfg
        model = build_bilstm(cfg).to(device)
    elif model_name == "conformer":
        cfg   = conformer_cfg
        model = build_conformer(cfg).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer  = Adam(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay)
    scheduler  = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_factor,
                                   patience=cfg.lr_patience, verbose=True)
    ctc_loss   = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True, reduction="mean")

    epochs     = max_epochs or cfg.max_epochs
    ckpt_path  = CKPT_DIR / f"{model_name}_best.pt"
    log_path   = LOG_DIR   / f"{model_name}_training_log.csv"

    # Resume from checkpoint
    start_epoch   = 0
    best_val_cer  = float("inf")
    no_improve    = 0

    if resume and Path(resume).exists():
        print(f"[train] Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_cer  = ckpt.get("best_val_cer", float("inf"))
        print(f"        Epoch {start_epoch}, best val CER = {best_val_cer:.4f}")

    # DataLoaders
    print("\n[train] Loading data...")
    train_loader = get_dataloader("train", shuffle=True)
    val_loader   = get_dataloader("val",   shuffle=False)
    test_loader  = get_dataloader("test",  shuffle=False)

    if eval_only:
        print("[train] Eval-only mode — skipping training")
        val_loss, val_cer, pairs = validate(model, val_loader, ctc_loss, device)
        print(f"  Val  Loss={val_loss:.4f}  CER={val_cer:.4f} ({100*val_cer:.1f}%)")
        test_loss, test_cer, test_pairs = validate(model, test_loader, ctc_loss, device)
        print(f"  Test Loss={test_loss:.4f}  CER={test_cer:.4f} ({100*test_cer:.1f}%)")
        run_confusion_analysis(test_pairs, str(RESULTS_DIR / f"{model_name}_test"))
        return

    # TensorBoard writer (optional — skip gracefully if not installed)
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(LOG_DIR / model_name))
        print(f"[train] TensorBoard logs at {LOG_DIR / model_name}")
    except ImportError:
        print("[train] tensorboard not found — CSV logging only")

    # CSV log header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_cer",
                         "best_val_cer", "lr", "elapsed_sec"])

    print(f"\n[train] Starting training: {model_name.upper()} for {epochs} epochs\n")
    print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} "
          f"{'Val CER':<12} {'LR':<12} {'Time'}")
    print("  " + "-" * 75)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, ctc_loss, device,
            subset_limit=subset,
        )

        # Validate
        val_loss, val_cer, val_pairs = validate(model, val_loader, ctc_loss, device)

        # LR schedule
        scheduler.step(val_cer)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        improved = val_cer < best_val_cer
        marker   = " ★" if improved else ""

        print(f"  {epoch+1:<8} {train_loss:<14.4f} {val_loss:<14.4f} "
              f"{val_cer:<12.4f} {current_lr:<12.2e} {elapsed:.0f}s{marker}")

        # Log to TensorBoard
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val",   val_loss,   epoch)
            tb_writer.add_scalar("CER/val",    val_cer,    epoch)
            tb_writer.add_scalar("LR",         current_lr, epoch)

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_cer:.6f}", f"{best_val_cer:.6f}", current_lr, f"{elapsed:.1f}"
            ])

        # Checkpoint on improvement
        if improved:
            best_val_cer = val_cer
            no_improve   = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_cer":    best_val_cer,
                "model_name":      model_name,
                "config":          cfg,
            }, ckpt_path)
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= cfg.patience:
            print(f"\n[train] Early stopping after {no_improve} epochs without improvement.")
            break

        # Sanity prediction print every 10 epochs (for monitoring)
        if (epoch + 1) % 10 == 0 and val_pairs:
            print(f"\n  Sample val predictions (epoch {epoch+1}):")
            for hyp, ref in val_pairs[:3]:
                print(f"    pred='{hyp}'  ref='{ref}'")
            print()

    # Final evaluation on test set using best checkpoint
    print(f"\n[train] Loading best checkpoint from {ckpt_path}")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    print("[train] Final evaluation on TEST set...")
    test_loss, test_cer, test_pairs = validate(model, test_loader, ctc_loss, device)
    from src.evaluation.metrics import word_error_rate
    test_wer = word_error_rate(
        [h.replace(" ", "") for h, _ in test_pairs],
        [r.replace(" ", "") for _, r in test_pairs],
    )

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Best Val CER : {best_val_cer:.4f} ({100*best_val_cer:.1f}%)")
    print(f"  Test CER     : {test_cer:.4f} ({100*test_cer:.1f}%)")
    print(f"  Test WER     : {test_wer:.4f} ({100*test_wer:.1f}%)")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"{'='*60}\n")

    # Confusion analysis on test predictions
    run_confusion_analysis(test_pairs, str(RESULTS_DIR / f"{model_name}_test"))

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL Fingerspelling model")
    parser.add_argument("--model",    type=str, required=True, choices=["bilstm", "conformer"])
    parser.add_argument("--subset",   type=int, default=None,
                        help="Max batches per epoch (for quick sanity check)")
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--resume",   type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    train(
        model_name=args.model,
        subset=args.subset,
        max_epochs=args.epochs,
        resume=args.resume,
        eval_only=args.eval_only,
    )
