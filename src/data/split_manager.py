"""
src/data/split_manager.py
==========================
Manages the train / val / test split for the ISL Fingerspelling dataset.

DESIGN DECISION (DD-002)
-------------------------
The dataset authors provide a train/test split in split_info.csv but NO
validation set. This module:
  1. Reads the provided split
  2. Identifies unique YouTube source-videos in the train set
  3. Randomly holds out VAL_SPLIT_RATIO (10%) of source-videos as validation
  4. Writes the final three-way split to cache/final_split.csv

WHY VIDEO-LEVEL (not segment-level) VAL CARVING?
  Segments from the same video are temporally correlated and likely show the
  same signer. If we split at segment level, we might put seg000 in train and
  seg001 in val — same signer, same motion style, which defeats the purpose
  of validation as a held-out measure.

REPRODUCIBILITY
  We fix RANDOM_SEED=42. The final_split.csv is written once and reused
  in all subsequent runs, so val composition never changes between experiments.

USAGE
-----
  python src/data/split_manager.py          # generates final_split.csv
  python src/data/split_manager.py --info   # print split sizes only
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import sys

# Add project root to path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    SPLIT_CSV, CACHE_DIR, FINAL_SPLIT_CSV,
    VAL_SPLIT_RATIO, RANDOM_SEED
)


def build_final_split(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Builds the definitive three-way split and caches it.

    Parameters
    ----------
    force_rebuild : bool
        If True, recompute even if final_split.csv already exists.
        Default False — reuse the cached split for reproducibility.

    Returns
    -------
    pd.DataFrame with columns [uid, split]  where split ∈ {train, val, test}
    """
    if FINAL_SPLIT_CSV.exists() and not force_rebuild:
        print(f"[split_manager] Loading cached split from {FINAL_SPLIT_CSV}")
        return pd.read_csv(FINAL_SPLIT_CSV)

    print(f"[split_manager] Building final split from {SPLIT_CSV}")
    df = pd.read_csv(SPLIT_CSV)

    # Extract source video ID from uid format: {youtube_id}_seg{NNN}
    df["video_id"] = df["uid"].str.extract(r"^(.+)_seg\d+$")

    # Separate provided train and test
    train_mask = df["split"] == "train"
    test_df    = df[~train_mask].copy()
    train_full = df[train_mask].copy()

    # Get unique source-videos in the train portion
    train_video_ids = sorted(train_full["video_id"].unique())

    # Randomly assign VAL_SPLIT_RATIO of source-videos to val
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(train_video_ids)

    n_val_videos = max(1, int(len(train_video_ids) * VAL_SPLIT_RATIO))
    val_video_ids = set(train_video_ids[:n_val_videos])
    train_video_ids_final = set(train_video_ids[n_val_videos:])

    # Assign new split labels
    def assign_split(row):
        if row["split"] == "test":
            return "test"
        elif row["video_id"] in val_video_ids:
            return "val"
        else:
            return "train"

    df["split"] = df.apply(assign_split, axis=1)

    # Save — drop the helper column
    result = df[["uid", "split"]].copy()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(FINAL_SPLIT_CSV, index=False)
    print(f"[split_manager] Saved to {FINAL_SPLIT_CSV}")

    return result


def print_split_info(df: pd.DataFrame) -> None:
    """Prints a human-readable summary of the final split."""
    print("\n" + "=" * 60)
    print("  Final Train / Val / Test Split")
    print("=" * 60)
    counts = df["split"].value_counts()
    total  = len(df)
    for split_name in ["train", "val", "test"]:
        n   = counts.get(split_name, 0)
        pct = 100 * n / total
        print(f"  {split_name:<8}  {n:>5} segments  ({pct:.1f}%)")
    print(f"  {'TOTAL':<8}  {total:>5} segments")

    # Verify video-source independence
    df["video_id"] = df["uid"].str.extract(r"^(.+)_seg\d+$")
    leakers = df.groupby("video_id")["split"].nunique()
    leakers = leakers[leakers > 1]
    if len(leakers) > 0:
        print(f"\n  WARNING: {len(leakers)} source-videos appear in multiple splits!")
    else:
        print(f"\n  OK: Split is fully video-source-independent.")
    print("=" * 60 + "\n")


def get_uids(split: str) -> list:
    """
    Returns the list of UIDs for a given split ('train', 'val', or 'test').
    Builds the split if not already cached.
    """
    df = build_final_split()
    return df[df["split"] == split]["uid"].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/inspect the ISL dataset split")
    parser.add_argument("--info",    action="store_true", help="Print split info only")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild even if cached")
    args = parser.parse_args()

    df = build_final_split(force_rebuild=args.rebuild)
    print_split_info(df)
