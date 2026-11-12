"""
src/static/train_comparison.py
================================
Unified training script that trains and compares all 4 model architectures:

  1. DH-GAT  — Enriched Dual-Hand Graph Attention Network (graph-based)
  2. GeoMLP  — Pure geometric feature MLP (no raw coords, no graph)
  3. HandTransformer — Lightweight Transformer on landmark tokens
  4. RandomForest    — Classical ML baseline on geometric features

All models use:
  - Same 60/20/20 stratified splits
  - Same enhanced augmentation pipeline
  - Same evaluation metrics (accuracy, per-class, confusion matrix)

Usage:
  python src/static/train_comparison.py
  python src/static/train_comparison.py --models dhgat geo_mlp transformer
  python src/static/train_comparison.py --models random_forest --no-train
"""

import argparse
import csv
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
from src.models.geo_mlp import build_geo_mlp, GEO_DIM_PER_HAND
from src.models.hand_transformer import build_hand_transformer
from src.static.static_dataset import get_static_dataloader
from src.static.train_static import (
    train_one_epoch, evaluate,
    compute_per_class_accuracy, save_per_class_bar, save_confusion_matrix
)

CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_MODEL_NAMES = ["dhgat", "geo_mlp", "transformer", "random_forest"]


# ─────────────────────────────────────────────────────────────────────────────
# GeoMLP adapter: wraps GeoMLP to accept the same batch dict as DH-GAT
# ─────────────────────────────────────────────────────────────────────────────
class GeoMLPAdapter(nn.Module):
    """Wraps GeoMLP so it takes the standard batch dict interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, right_node, left_node, right_present, left_present,
                right_geo=None, left_geo=None):
        # GeoMLP uses the full 35-dim geo vector, not just the 10-dim inter-fingertip
        # We need right_geo_full/left_geo_full; this adapter is called from the
        # dedicated training loop below that passes them directly.
        return self.model(right_geo, left_geo, right_present, left_present)


# ─────────────────────────────────────────────────────────────────────────────
# GeoMLP training loop (slightly different — uses right_geo_full)
# ─────────────────────────────────────────────────────────────────────────────
def _train_geo_mlp_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model(batch["right_geo_full"], batch["left_geo_full"],
                       batch["right_present"], batch["left_present"])
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
def _eval_geo_mlp(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pairs = []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model(batch["right_geo_full"], batch["left_geo_full"],
                       batch["right_present"], batch["left_present"])
        loss = criterion(logits, batch["label"])
        total_loss += loss.item() * batch["label"].size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()
        total   += batch["label"].size(0)
        for p, t in zip(preds.cpu().tolist(), batch["label"].cpu().tolist()):
            pairs.append((IDX_TO_LETTER[p], IDX_TO_LETTER[t]))
    return total_loss / max(total, 1), correct / max(total, 1), pairs


# ─────────────────────────────────────────────────────────────────────────────
# Neural model training (generic)
# ─────────────────────────────────────────────────────────────────────────────
def train_neural_model(model_name: str, cfg: StaticConfig,
                       train_loader, val_loader, test_loader,
                       device: torch.device) -> dict:
    """Train one neural model and return result dict."""
    print(f"\n{'='*65}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*65}")

    if model_name == "dhgat":
        model = build_dhgat(cfg).to(device)
        train_fn = train_one_epoch
        eval_fn  = evaluate
    elif model_name == "geo_mlp":
        model = build_geo_mlp(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
        train_fn = _train_geo_mlp_epoch
        eval_fn  = _eval_geo_mlp
    elif model_name == "transformer":
        model = build_hand_transformer(cfg).to(device)
        train_fn = train_one_epoch
        eval_fn  = evaluate
    else:
        raise ValueError(f"Unknown neural model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=1e-5)

    ckpt_path    = CHECKPOINTS_DIR / f"{model_name}_best.pt"
    best_val_acc = 0.0
    no_improve   = 0

    print(f"\n  {'Epoch':6s}  {'TrLoss':8s}  {'TrAcc':7s}  {'VaAcc':7s}  {'LR':9s}  {'Time':5s}")
    print(f"  {'-'*55}")

    for epoch in range(cfg.max_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_fn(model, train_loader, optimizer, criterion, device)
        _, va_acc, _    = eval_fn(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = va_acc > best_val_acc
        marker  = " *" if is_best else ""
        print(f"  {epoch+1:6d}  {tr_loss:8.4f}  {100*tr_acc:6.2f}%  "
              f"{100*va_acc:6.2f}%  {optimizer.param_groups[0]['lr']:.2e}  "
              f"{elapsed:.0f}s{marker}")

        if is_best:
            best_val_acc = va_acc
            no_improve   = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "best_val_acc": best_val_acc, "config": cfg,
                "model_name": model_name,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"\n  Early stopping (patience={cfg.patience})")
                break

    # Load best and test
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    _, test_acc, test_pairs = eval_fn(model, test_loader, criterion, device)

    # Inference speed (ms per sample)
    model.eval()
    sample_batch = next(iter(test_loader))
    sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample_batch.items()}
    N_RUNS = 50
    t_start = time.time()
    with torch.no_grad():
        for _ in range(N_RUNS):
            if model_name == "geo_mlp":
                model(sample_batch["right_geo_full"][:1], sample_batch["left_geo_full"][:1],
                      sample_batch["right_present"][:1], sample_batch["left_present"][:1])
            else:
                model(sample_batch["right_node"][:1], sample_batch["left_node"][:1],
                      sample_batch["right_present"][:1], sample_batch["left_present"][:1],
                      sample_batch["right_geo"][:1], sample_batch["left_geo"][:1])
    ms_per_sample = (time.time() - t_start) / N_RUNS * 1000

    print(f"\n  {model_name.upper()} Results:")
    print(f"    Best Val Acc : {100*best_val_acc:.2f}%")
    print(f"    Test Acc     : {100*test_acc:.2f}%")
    print(f"    Inference    : {ms_per_sample:.2f} ms/sample (CPU)")

    # Save confusion matrix and per-class accuracy
    per_class = compute_per_class_accuracy(test_pairs)
    save_confusion_matrix(test_pairs, RESULTS_DIR / f"{model_name}_confusion.png")
    save_per_class_bar(per_class, RESULTS_DIR / f"{model_name}_per_class.png")

    return {
        "model": model_name,
        "params": n_params,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "ms_per_sample": ms_per_sample,
        "test_pairs": test_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest baseline
# ─────────────────────────────────────────────────────────────────────────────
def train_random_forest(train_loader, val_loader, test_loader) -> dict:
    """Train a Random Forest on geometric features (no neural network)."""
    print(f"\n{'='*65}")
    print(f"  Training: RANDOM FOREST (classical ML baseline)")
    print(f"{'='*65}")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("  [SKIP] scikit-learn not installed. Run: pip install scikit-learn")
        return {"model": "random_forest", "test_acc": 0.0, "skipped": True}

    def _collect(loader):
        X, y = [], []
        for batch in loader:
            r = batch["right_geo_full"].numpy()    # (B, 35)
            l = batch["left_geo_full"].numpy()     # (B, 35)
            rp = batch["right_present"].numpy().astype(np.float32).reshape(-1, 1)
            lp = batch["left_present"].numpy().astype(np.float32).reshape(-1, 1)
            feat = np.concatenate([r * rp, l * lp, rp, lp], axis=1)   # (B, 72)
            X.append(feat)
            y.append(batch["label"].numpy())
        return np.concatenate(X), np.concatenate(y)

    print("  Collecting features from train set...")
    X_train, y_train = _collect(train_loader)
    X_val,   y_val   = _collect(val_loader)
    X_test,  y_test  = _collect(test_loader)

    print(f"  Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        n_jobs=-1, random_state=42, class_weight="balanced"
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_preds  = clf.predict(X_val)
    test_preds = clf.predict(X_test)
    val_acc  = accuracy_score(y_val,  val_preds)
    test_acc = accuracy_score(y_test, test_preds)

    # Inference time
    t_inf = time.time()
    for _ in range(50):
        clf.predict(X_test[:1])
    ms_per_sample = (time.time() - t_inf) / 50 * 1000

    print(f"  Training time  : {elapsed:.1f}s")
    print(f"  Val  Accuracy  : {100*val_acc:.2f}%")
    print(f"  Test Accuracy  : {100*test_acc:.2f}%")
    print(f"  Inference      : {ms_per_sample:.3f} ms/sample")

    test_pairs = [(IDX_TO_LETTER[p], IDX_TO_LETTER[t])
                  for p, t in zip(test_preds.tolist(), y_test.tolist())]
    per_class = compute_per_class_accuracy(test_pairs)
    save_confusion_matrix(test_pairs, RESULTS_DIR / "random_forest_confusion.png")
    save_per_class_bar(per_class, RESULTS_DIR / "random_forest_per_class.png")

    # Save model
    import pickle
    rf_path = CHECKPOINTS_DIR / "random_forest.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Saved to {rf_path}")

    return {
        "model": "random_forest",
        "params": "N/A",
        "best_val_acc": val_acc,
        "test_acc": test_acc,
        "ms_per_sample": ms_per_sample,
        "test_pairs": test_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison summary
# ─────────────────────────────────────────────────────────────────────────────
def _print_comparison_table(results: list):
    print(f"\n\n{'='*75}")
    print(f"  MODEL COMPARISON RESULTS — ISL Static Fingerspelling")
    print(f"{'='*75}")
    print(f"  {'Model':<18}  {'Params':>10}  {'Val Acc':>8}  {'Test Acc':>9}  {'ms/sample':>10}")
    print(f"  {'-'*65}")
    for r in sorted(results, key=lambda x: x.get("test_acc", 0), reverse=True):
        params = f"{r.get('params','N/A'):,}" if isinstance(r.get('params'), int) else "N/A"
        print(f"  {r['model']:<18}  {params:>10}  "
              f"{100*r.get('best_val_acc',0):>7.2f}%  "
              f"{100*r.get('test_acc',0):>8.2f}%  "
              f"{r.get('ms_per_sample',0):>9.2f}ms")
    print(f"{'='*75}")

    # Save CSV
    csv_path = RESULTS_DIR / "model_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "params", "best_val_acc",
                                           "test_acc", "ms_per_sample"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"\n  Comparison table saved to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(model_names: list):
    cfg    = StaticConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_comparison] Device: {device}")

    print("\n[train_comparison] Loading datasets...")
    train_loader = get_static_dataloader("train", batch_size=cfg.batch_size)
    val_loader   = get_static_dataloader("val",   batch_size=cfg.batch_size)
    test_loader  = get_static_dataloader("test",  batch_size=cfg.batch_size)

    results = []
    for name in model_names:
        if name == "random_forest":
            r = train_random_forest(train_loader, val_loader, test_loader)
        else:
            r = train_neural_model(name, cfg, train_loader, val_loader, test_loader, device)
        results.append(r)

    _print_comparison_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ISL static fingerspelling models")
    parser.add_argument(
        "--models", nargs="+",
        choices=ALL_MODEL_NAMES,
        default=ALL_MODEL_NAMES,
        help="Which models to train and compare (default: all)"
    )
    args = parser.parse_args()
    main(args.models)
