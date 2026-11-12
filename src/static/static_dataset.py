"""
src/static/static_dataset.py
=============================
PyTorch Dataset for static ISL fingerspelling recognition.

Loads pre-extracted landmark .npz files from cache/static/ and serves them
as (right_lm, left_lm, right_present, left_present, label) tuples.

KEY DESIGN DECISIONS:
1. MERGED DATASET: Both Static_ISL (~1200 imgs/class) and Kaggle (~25 imgs/class)
   are merged. Kaggle images have varied backgrounds which MediaPipe already
   handled during extraction — landmarks are inherently background-invariant.

2. STRATIFIED SPLIT: 60% train / 20% val / 20% test, stratified by class.
   The split is deterministic (seeded) and cached to a CSV for reproducibility.

3. ON-THE-FLY AUGMENTATION (training only):
   - Gaussian noise: simulates MediaPipe jitter
   - Random scale: simulates different hand distances from camera
   - Random rotation (Z-axis): simulates wrist tilt
   - Landmark dropout: simulates partial occlusion

4. MISSING HAND HANDLING: If a hand was not detected, its landmark array is
   all-zeros and present=False. The model uses presence flags to ignore these.

Run self-test:
  python src/static/static_dataset.py --self-test
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    STATIC_CACHE_DIR, STATIC_LETTERS, LETTER_TO_IDX,
    NUM_LANDMARKS, LANDMARK_DIM, NUM_STATIC_CLASSES, ROOT
)
from src.preprocessing.feature_engineer import compute_node_features, compute_global_distances, compute_geo_vector

SPLIT_CSV       = STATIC_CACHE_DIR / "split.csv"
WEBCAM_DATA_DIR = ROOT / "cache" / "webcam_data"


# ─────────────────────────────────────────────────────────────────────────────
# Split management
# ─────────────────────────────────────────────────────────────────────────────
def _build_or_load_split() -> dict[str, list]:
    """
    Discover all .npz files in cache/static/ and assign each to a split.
    Saves split assignments to split.csv for reproducibility.
    Returns dict: {"train": [...], "val": [...], "test": [...]}
    """
    if SPLIT_CSV.exists():
        split = {"train": [], "val": [], "test": []}
        with open(SPLIT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                split[row["split"]].append((Path(row["path"]), row["label"]))
        return split

    # Collect all files grouped by letter
    rng = np.random.default_rng(seed=42)
    split = {"train": [], "val": [], "test": []}

    for letter in STATIC_LETTERS:
        files = sorted((STATIC_CACHE_DIR / letter).glob("*.npz"))
        if not files:
            continue
        files = np.array(files)
        rng.shuffle(files)
        n = len(files)
        n_val  = max(1, int(0.20 * n))
        n_test = max(1, int(0.20 * n))
        n_train = n - n_val - n_test

        split["train"].extend([(f, letter) for f in files[:n_train]])
        split["val"].extend(  [(f, letter) for f in files[n_train:n_train+n_val]])
        split["test"].extend( [(f, letter) for f in files[n_train+n_val:]])

    # Save to CSV
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "path", "label"])
        w.writeheader()
        for s, items in split.items():
            for path, label in items:
                w.writerow({"split": s, "path": str(path), "label": label})

    return split


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _augment_hand(lm: np.ndarray, present: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Apply random augmentations to a single hand's (21, 3) landmarks.
    Only modifies the array if the hand is present.

    Augmentations are designed to bridge the domain gap between curated
    dataset images and real-time webcam captures.
    """
    if not present or lm is None:
        return lm

    lm = lm.copy()

    # 1. Finger-specific Gaussian noise
    #    Fingertips (4,8,12,16,20) are noisier than palm joints in MediaPipe.
    FINGERTIPS = [4, 8, 12, 16, 20]
    PALM_JOINTS = [0, 1, 5, 9, 13, 17]
    noise = rng.normal(0, 0.015, lm.shape).astype(np.float32)
    # Higher noise for fingertips
    for tip_idx in FINGERTIPS:
        noise[tip_idx] *= 1.5
    # Lower noise for stable palm joints
    for palm_idx in PALM_JOINTS:
        noise[palm_idx] *= 0.4
    lm += noise

    # 2. Random scale (±10%)
    scale = rng.uniform(0.90, 1.10)
    lm *= scale

    # 3. Z-axis rotation (simulate wrist tilt, ±25° — wider than before)
    if rng.random() < 0.5:
        angle = rng.uniform(-25, 25) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        lm[:, :2] = lm[:, :2] @ rot.T

    # 4. X-axis rotation (simulate hand viewed from slightly above/below, ±20°)
    if rng.random() < 0.5:
        angle = rng.uniform(-20, 20) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[1, 0, 0],
                        [0, cos_a, -sin_a],
                        [0, sin_a,  cos_a]], dtype=np.float32)
        lm = lm @ rot.T

    # 5. Y-axis rotation (simulate hand viewed from slight side angle, ±20°)
    if rng.random() < 0.5:
        angle = rng.uniform(-20, 20) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[ cos_a, 0, sin_a],
                        [     0, 1,     0],
                        [-sin_a, 0, cos_a]], dtype=np.float32)
        lm = lm @ rot.T

    # 6. Z-coordinate corruption: simulate MediaPipe depth estimation variance
    #    between dataset images and webcam (add correlated offset to all z)
    if rng.random() < 0.35:
        z_offset = rng.uniform(-0.12, 0.12)
        lm[:, 2] += z_offset

    # 7. Landmark dropout (zero out ~5% of joints randomly, simulates occlusion)
    if rng.random() < 0.3:
        drop_idxs = rng.choice(lm.shape[0], size=max(1, lm.shape[0] // 20), replace=False)
        lm[drop_idxs] = 0.0

    return lm


def _augment_hand_swap(right_lm: np.ndarray, left_lm: np.ndarray,
                       right_present: bool, left_present: bool,
                       rng: np.random.Generator):
    """
    Hand swap augmentation (15% probability): swap right↔left hands and
    mirror x-coordinates. Simulates left-handed signers in the training data.

    Returns augmented (right_lm, left_lm, right_present, left_present).
    """
    if rng.random() >= 0.15:
        return right_lm, left_lm, right_present, left_present

    # Mirror x-coordinates (flip hand laterally)
    r_mirrored = right_lm.copy()
    r_mirrored[:, 0] *= -1
    l_mirrored = left_lm.copy()
    l_mirrored[:, 0] *= -1

    # Swap: what was right becomes left and vice versa
    return l_mirrored, r_mirrored, left_present, right_present


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────
class StaticISLDataset(Dataset):
    """
    Dataset for static ISL fingerspelling recognition.

    Parameters
    ----------
    split     : 'train' | 'val' | 'test'
    augment   : whether to apply on-the-fly augmentation (True for train only)
    """
    def __init__(self, split: str = "train", augment: bool = None):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split   = split
        self.augment = augment if augment is not None else (split == "train")
        self.rng     = np.random.default_rng(seed=None)  # non-deterministic per epoch

        all_splits = _build_or_load_split()
        items = all_splits[split]

        n_per_class = {}
        for _, label in items:
            n_per_class[label] = n_per_class.get(label, 0) + 1
        print(f"[StaticISLDataset] split={split:5s}  total={len(items):4d}  "
              f"classes={len(n_per_class):2d}  augment={self.augment}  (preloading...)", flush=True)

        # Preload all data into RAM to avoid per-sample disk I/O during training.
        # Total size: ~45K samples × 2 × (21×3) float32 ≈ 22 MB — very manageable.
        n = len(items)
        self.right_lm      = np.zeros((n, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        self.left_lm       = np.zeros((n, NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        self.right_present = np.zeros(n, dtype=bool)
        self.left_present  = np.zeros(n, dtype=bool)
        self.labels        = np.zeros(n, dtype=np.int64)

        for i, (path, label) in enumerate(items):
            data = np.load(path, allow_pickle=True)
            self.right_lm[i]      = data["right_lm"].astype(np.float32)
            self.left_lm[i]       = data["left_lm"].astype(np.float32)
            self.right_present[i] = bool(data["right_present"])
            self.left_present[i]  = bool(data["left_present"])
            self.labels[i]        = LETTER_TO_IDX[label]

        print(f"[StaticISLDataset] split={split:5s}  preload done", flush=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        right_lm      = self.right_lm[idx].copy()
        left_lm       = self.left_lm[idx].copy()
        right_present = bool(self.right_present[idx])
        left_present  = bool(self.left_present[idx])

        if self.augment:
            right_lm, left_lm, right_present, left_present = _augment_hand_swap(
                right_lm, left_lm, right_present, left_present, self.rng
            )
            right_lm = _augment_hand(right_lm, right_present, self.rng)
            left_lm  = _augment_hand(left_lm,  left_present,  self.rng)

        # Compute enriched node features and global geometric features
        right_node = compute_node_features(right_lm)       # (21, 6)
        left_node  = compute_node_features(left_lm)        # (21, 6)
        right_geo  = compute_global_distances(right_lm)    # (10,)
        left_geo   = compute_global_distances(left_lm)     # (10,)
        right_geo_full = compute_geo_vector(right_lm)      # (35,) for GeoMLP
        left_geo_full  = compute_geo_vector(left_lm)       # (35,) for GeoMLP

        return {
            # Raw landmarks (kept for backward compat / visualization)
            "right_lm":      torch.from_numpy(right_lm),
            "left_lm":       torch.from_numpy(left_lm),
            # Enriched node features for DH-GAT and HandTransformer
            "right_node":    torch.from_numpy(right_node),
            "left_node":     torch.from_numpy(left_node),
            # Global geometric features for DH-GAT fusion and GeoMLP
            "right_geo":     torch.from_numpy(right_geo),
            "left_geo":      torch.from_numpy(left_geo),
            # Full geometric vector for GeoMLP model
            "right_geo_full": torch.from_numpy(right_geo_full),
            "left_geo_full":  torch.from_numpy(left_geo_full),
            # Presence flags and label
            "right_present": torch.tensor(right_present, dtype=torch.bool),
            "left_present":  torch.tensor(left_present,  dtype=torch.bool),
            "label":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_static_dataloader(split: str, batch_size: int = 64,
                          shuffle: bool = None) -> DataLoader:
    """
    Convenience function that returns a DataLoader for the given split.
    Augmentation is automatically enabled only for train.
    """
    if shuffle is None:
        shuffle = (split == "train")
    dataset = StaticISLDataset(split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,          # Windows safe
        pin_memory=False,
        drop_last=(split == "train"),
    )


def get_finetune_dataloader(batch_size: int = 32) -> DataLoader:
    """
    Returns a DataLoader that ONLY contains user-collected webcam data.
    Used during fine-tuning to adapt the model to real webcam domain.
    All webcam samples go to training (no val/test needed — only a few dozen samples).
    """
    items = []
    for letter in STATIC_LETTERS:
        folder = WEBCAM_DATA_DIR / letter
        if folder.exists():
            for f in sorted(folder.glob("*.npz")):
                items.append((f, letter))

    if not items:
        return None

    class WebcamDataset(Dataset):
        def __init__(self):
            self.items = items
            self.rng   = np.random.default_rng()
            print(f"[WebcamDataset] Loaded {len(items)} real webcam samples")

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            data = np.load(path, allow_pickle=True)
            right_lm      = data["right_lm"].astype(np.float32)
            left_lm       = data["left_lm"].astype(np.float32)
            right_present = bool(data["right_present"])
            left_present  = bool(data["left_present"])
            # Apply augmentation (more aggressive for small dataset)
            right_lm, left_lm, right_present, left_present = _augment_hand_swap(
                right_lm, left_lm, right_present, left_present, self.rng
            )
            right_lm = _augment_hand(right_lm, right_present, self.rng)
            left_lm  = _augment_hand(left_lm,  left_present,  self.rng)
            right_node = compute_node_features(right_lm)
            left_node  = compute_node_features(left_lm)
            right_geo  = compute_global_distances(right_lm)
            left_geo   = compute_global_distances(left_lm)
            right_geo_full = compute_geo_vector(right_lm)
            left_geo_full  = compute_geo_vector(left_lm)
            return {
                "right_lm":       torch.from_numpy(right_lm),
                "left_lm":        torch.from_numpy(left_lm),
                "right_node":     torch.from_numpy(right_node),
                "left_node":      torch.from_numpy(left_node),
                "right_geo":      torch.from_numpy(right_geo),
                "left_geo":       torch.from_numpy(left_geo),
                "right_geo_full": torch.from_numpy(right_geo_full),
                "left_geo_full":  torch.from_numpy(left_geo_full),
                "right_present":  torch.tensor(right_present, dtype=torch.bool),
                "left_present":   torch.tensor(left_present,  dtype=torch.bool),
                "label":          torch.tensor(LETTER_TO_IDX[label], dtype=torch.long),
            }

    return DataLoader(WebcamDataset(), batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=False, drop_last=False)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
def _self_test():
    print("Running StaticISLDataset self-tests...")

    # Check cache exists
    if not STATIC_CACHE_DIR.exists():
        print("  [SKIP] cache/static/ not found — run extract_static_landmarks.py first")
        return

    npz_files = list(STATIC_CACHE_DIR.rglob("*.npz"))
    if not npz_files:
        print("  [SKIP] No .npz files found — run extract_static_landmarks.py first")
        return

    print(f"  Found {len(npz_files)} cached landmark files")

    ds = StaticISLDataset(split="train")
    assert len(ds) > 0, "Train split is empty!"

    sample = ds[0]
    assert sample["right_lm"].shape == (21, 3), f"Bad right_lm shape: {sample['right_lm'].shape}"
    assert sample["left_lm"].shape  == (21, 3), f"Bad left_lm shape: {sample['left_lm'].shape}"
    assert 0 <= sample["label"].item() < NUM_STATIC_CLASSES
    print(f"  ✓ Sample item shapes correct: right_lm={sample['right_lm'].shape}, label={sample['label'].item()}")

    loader = get_static_dataloader("train", batch_size=16)
    batch  = next(iter(loader))
    assert batch["right_lm"].shape == (16, 21, 3)
    assert batch["label"].shape    == (16,)
    print(f"  ✓ Dataloader batch shapes: right_lm={batch['right_lm'].shape}, label={batch['label'].shape}")

    print("\nAll StaticISLDataset self-tests PASSED ✓\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
