"""
src/static/train_static.py
============================
Training script for the DH-GAT static ISL fingerspelling classifier.

TRAINING SETUP:
  - Loss      : CrossEntropyLoss with label smoothing (0.1)
  - Optimizer : AdamW (better generalization than Adam via weight decay)
  - LR Schedule: CosineAnnealingLR (smooth decay, no plateau needed)
  - Early stopping: patience=15 on validation accuracy

OUTPUTS:
  - Best checkpoint    : checkpoints/dh_gat_static_best.pt
  - Per-class accuracy : results/static_per_class_accuracy.png
  - Confusion matrix   : results/static_confusion_normalized.png
  - Test pairs CSV     : results/static_test_pairs.csv

Run:
  python src/static/train_static.py
  python src/static/train_static.py --eval-only --resume checkpoints/dh_gat_static_best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    StaticConfig, STATIC_LETTERS, IDX_TO_LETTER, NUM_STATIC_CLASSES,
    RESULTS_DIR, ROOT
)
from src.models.dh_gat import build_dhgat
from src.static.static_dataset import get_static_dataloader, get_finetune_dataloader

CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        logits = model(batch["right_node"], batch["left_node"],
                       batch["right_present"], batch["left_present"],
                       batch["right_geo"], batch["left_geo"])
        loss = criterion(logits, batch["label"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()
        total   += batch["label"].size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float, list]:
    """Returns (loss, accuracy, pairs) where pairs = [(pred_letter, true_letter)]"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pairs = []
    for batch in loader:
        batch  = _batch_to_device(batch, device)
        logits = model(batch["right_node"], batch["left_node"],
                       batch["right_present"], batch["left_present"],
                       batch["right_geo"], batch["left_geo"])
        loss   = criterion(logits, batch["label"])

        total_loss += loss.item() * batch["label"].size(0)
        preds  = logits.argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()
        total   += batch["label"].size(0)

        for p, t in zip(preds.cpu().tolist(), batch["label"].cpu().tolist()):
            pairs.append((IDX_TO_LETTER[p], IDX_TO_LETTER[t]))

    return total_loss / max(total, 1), correct / max(total, 1), pairs


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────────────────────────────────────
def compute_per_class_accuracy(pairs: list) -> dict[str, float]:
    correct_per = {lt: 0 for lt in STATIC_LETTERS}
    total_per   = {lt: 0 for lt in STATIC_LETTERS}
    for pred, ref in pairs:
        total_per[ref] += 1
        if pred == ref:
            correct_per[ref] += 1
    return {lt: correct_per[lt] / max(total_per[lt], 1) for lt in STATIC_LETTERS
            if total_per[lt] > 0}


def save_per_class_bar(per_class: dict, out_path: Path):
    letters = sorted(per_class.keys())
    accs    = [per_class[lt] * 100 for lt in letters]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors  = ["#2ecc71" if a >= 90 else "#e67e22" if a >= 70 else "#e74c3c" for a in accs]
    bars    = ax.bar(letters, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.8, label="90% target")
    ax.set_xlabel("ISL Letter", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("DH-GAT Per-Class Accuracy — ISL Static Fingerspelling", fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.0f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[results] Per-class accuracy saved to {out_path}")


def save_confusion_matrix(pairs: list, out_path: Path):
    N   = NUM_STATIC_CLASSES
    mat = np.zeros((N, N), dtype=int)
    for pred, ref in pairs:
        pi = STATIC_LETTERS.index(pred) if pred in STATIC_LETTERS else -1
        ri = STATIC_LETTERS.index(ref)  if ref  in STATIC_LETTERS else -1
        if pi >= 0 and ri >= 0:
            mat[ri, pi] += 1

    # Row-normalize (each row = reference class, shows recall per class)
    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    norm_mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm_mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels(STATIC_LETTERS, fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(STATIC_LETTERS, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Reference (True Label)", fontsize=11)
    ax.set_title("DH-GAT Confusion Matrix — ISL Static Fingerspelling (Normalized)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.03)
    for i in range(N):
        for j in range(N):
            val = norm_mat[i, j]
            if val > 0.05:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.6 else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[results] Confusion matrix saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(resume: str = None, eval_only: bool = False):
    cfg    = StaticConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_static] Device: {device}")

    model     = build_dhgat(cfg).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DHGAT] Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=1e-5)

    ckpt_path = CHECKPOINTS_DIR / "dh_gat_static_best.pt"
    start_epoch  = 0
    best_val_acc = 0.0

    if resume and Path(resume).exists():
        print(f"[train_static] Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if not eval_only:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt.get("scheduler_state", scheduler.state_dict()))
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"        Epoch {start_epoch}, best val acc = {best_val_acc:.4f}")

    print("\n[train_static] Loading data...")
    train_loader = get_static_dataloader("train", batch_size=cfg.batch_size)
    val_loader   = get_static_dataloader("val",   batch_size=cfg.batch_size)
    test_loader  = get_static_dataloader("test",  batch_size=cfg.batch_size)

    if eval_only:
        print("[train_static] Eval-only mode — skipping training")
        _, val_acc,  _ = evaluate(model, val_loader,  criterion, device)
        _, test_acc, test_pairs = evaluate(model, test_loader, criterion, device)
        print(f"  Val  Accuracy: {100*val_acc:.2f}%")
        print(f"  Test Accuracy: {100*test_acc:.2f}%")
        _save_results(test_pairs)
        return

    print(f"\n[train_static] Training DH-GAT for {cfg.max_epochs} epochs\n")
    print(f"  {'Epoch':6s}  {'Train Loss':11s}  {'Train Acc':10s}  "
          f"{'Val Loss':9s}  {'Val Acc':8s}  {'LR':10s}  {'Time':6s}")
    print(f"  {'-'*75}")

    no_improve = 0
    for epoch in range(start_epoch, cfg.max_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = val_acc > best_val_acc
        marker  = " *" if is_best else ""
        print(f"  {epoch+1:6d}  {train_loss:11.4f}  {100*train_acc:9.2f}%  "
              f"{val_loss:9.4f}  {100*val_acc:7.2f}%  "
              f"{optimizer.param_groups[0]['lr']:.2e}  {elapsed:.0f}s{marker}")

        if is_best:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc":   best_val_acc,
                "config":         cfg,
            }, ckpt_path)
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            print(f"\n[train_static] Early stopping after {no_improve} epochs without improvement.")
            break

        # Print sample predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            _, _, val_pairs = evaluate(model, val_loader, criterion, device)
            print(f"\n  Sample val predictions (epoch {epoch+1}):")
            for pred, ref in val_pairs[:5]:
                mark = "OK" if pred == ref else "XX"
                print(f"    {mark}  pred='{pred}'  ref='{ref}'")
            print()

    # Final evaluation
    print(f"\n[train_static] Loading best checkpoint (val acc={100*best_val_acc:.2f}%)")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    _, test_acc, test_pairs = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: DH-GAT Static ISL Classifier")
    print(f"{'='*60}")
    print(f"  Best Val Accuracy  : {100*best_val_acc:.2f}%")
    print(f"  Test Accuracy      : {100*test_acc:.2f}%")
    print(f"  Checkpoint         : {ckpt_path}")
    print(f"{'='*60}\n")

    _save_results(test_pairs)


def _save_results(test_pairs: list):
    per_class = compute_per_class_accuracy(test_pairs)

    print("\n  Per-class accuracy:")
    for lt, acc in sorted(per_class.items(), key=lambda x: x[1]):
        bar = "#" * int(acc * 20)
        print(f"    '{lt}'  {100*acc:5.1f}%  {bar}")

    save_per_class_bar(per_class, RESULTS_DIR / "static_per_class_accuracy.png")
    save_confusion_matrix(test_pairs, RESULTS_DIR / "static_confusion_normalized.png")

    pairs_path = RESULTS_DIR / "static_test_pairs.csv"
    import csv
    with open(pairs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["predicted", "reference"])
        w.writeheader()
        for pred, ref in test_pairs:
            w.writerow({"predicted": pred, "reference": ref})
    print(f"\n  Test pairs saved to {pairs_path}")


def fine_tune():
    """
    Fine-tune the trained DH-GAT on user-collected webcam data.
    Uses a very low learning rate to adapt without forgetting original knowledge.
    Only runs for up to 30 epochs with early stopping.
    """
    ckpt_path = CHECKPOINTS_DIR / "dh_gat_static_best.pt"
    if not ckpt_path.exists():
        print("[fine_tune] ERROR: No checkpoint found. Run train first.")
        return

    webcam_loader = get_finetune_dataloader(batch_size=16)
    if webcam_loader is None:
        print("[fine_tune] ERROR: No webcam data found.")
        print("  Run: python src/static/collect_webcam_data.py")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = StaticConfig()

    model = build_dhgat(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    print(f"[fine_tune] Loaded checkpoint (val acc={100*best_val_acc:.2f}%)")

    # Very low LR for fine-tuning — we don't want to forget what was learned
    ft_lr   = cfg.learning_rate * 0.05   # 5% of original LR
    optimizer = AdamW(model.parameters(), lr=ft_lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # less smoothing for fine-tune
    val_loader = get_static_dataloader("val", batch_size=cfg.batch_size)

    print(f"\n[fine_tune] Fine-tuning on {len(webcam_loader.dataset)} webcam samples")
    print(f"  LR={ft_lr:.2e}  max_epochs=30  patience=8")
    print(f"  {'Epoch':6s}  {'FT Loss':9s}  {'Val Acc':8s}")
    print(f"  {'-'*35}")

    best_ft_acc = best_val_acc
    no_improve  = 0
    ft_ckpt_path = CHECKPOINTS_DIR / "dh_gat_static_finetuned.pt"

    for epoch in range(30):
        model.train()
        total_loss = 0
        n_batches  = 0
        for batch in webcam_loader:
            batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch["right_node"], batch["left_node"],
                           batch["right_present"], batch["left_present"],
                           batch["right_geo"], batch["left_geo"])
            loss   = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        _, val_acc, _ = evaluate(model, val_loader, criterion, device)
        avg_loss = total_loss / max(n_batches, 1)
        is_best  = val_acc > best_ft_acc
        marker   = " ★" if is_best else ""
        print(f"  {epoch+1:6d}  {avg_loss:9.4f}  {100*val_acc:7.2f}%{marker}")

        if is_best:
            best_ft_acc = val_acc
            no_improve  = 0
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc":   best_ft_acc,
                "config":         cfg,
            }, ft_ckpt_path)
        else:
            no_improve += 1
            if no_improve >= 8:
                print(f"[fine_tune] Early stopping.")
                break

    print(f"\n[fine_tune] Done! Best val acc after fine-tuning: {100*best_ft_acc:.2f}%")
    print(f"  Checkpoint: {ft_ckpt_path}")
    print(f"\n  To use the fine-tuned model in webcam demo, run:")
    print(f"    python src/static/webcam_static.py --checkpoint checkpoints/dh_gat_static_finetuned.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DH-GAT static ISL classifier")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--fine-tune",  action="store_true",
                        help="Fine-tune on user-collected webcam data")
    parser.add_argument("--resume",     type=str, default=None)
    args = parser.parse_args()

    if args.fine_tune:
        fine_tune()
    else:
        train(resume=args.resume, eval_only=args.eval_only)
