"""
src/data/dataset_loader.py
============================
Stage 3: PyTorch Dataset and DataLoader for ISL Fingerspelling.

DESIGN DECISIONS
-----------------
1. LAZY LOADING: Landmark arrays are loaded from .npy cache at __getitem__ time
   using mmap_mode='r'. This means the full dataset never lives in RAM at once —
   critical for 1308 × ~97 frames × 63 features.

2. LABEL ENCODING: Labels are encoded as integer sequences using CHAR_TO_IDX.
   Special characters are silently stripped (DD-001). A segment with no valid
   characters after stripping is excluded from the dataset.

3. DROPPED SEGMENTS: Segments flagged in extraction_stats.csv (>50% missing
   hand detections) are excluded from all splits.

4. VARIABLE LENGTH BATCHING: Sequences vary from ~20 to ~315 frames.
   The collate_fn pads to the longest sequence in each batch and returns:
     - padded_landmarks: (B, T_max, 63)
     - labels:           (B, L_max) padded with -1
     - input_lengths:    (B,) actual frame counts (for CTC)
     - label_lengths:    (B,) actual label lengths (for CTC)
   WHY -1 for label padding: CTC only reads up to label_lengths[i] characters
   for sample i, so the padding value is never used in loss computation.

SELF-TEST
---------
  python src/data/dataset_loader.py --self-test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    ANN_CSV, CACHE_DIR, EXTRACTION_STATS_CSV,
    CHAR_TO_IDX, VOCAB,
    FEATURE_DIM, BATCH_SIZE, NUM_WORKERS,
    DROP_SEGMENT_THRESHOLD,
)
from src.data.split_manager import build_final_split
from src.preprocessing.normalizer import normalize_sequence
from src.extraction.landmark_extractor import load_landmarks


def encode_label(text: str) -> list:
    """
    Converts a transcription string to a list of integer class indices.

    Only characters in VOCAB (a–z) are kept; everything else is stripped.
    WHY: Special chars (hyphens, digits) are annotation artifacts, not signs
    (DD-001 in DESIGN_DECISIONS.md).

    Returns empty list if no valid characters remain.
    """
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def load_dropped_uids() -> set:
    """
    Loads the set of UIDs flagged as dropped during extraction.
    If extraction_stats.csv doesn't exist yet, returns empty set.
    """
    if not EXTRACTION_STATS_CSV.exists():
        return set()
    stats = pd.read_csv(EXTRACTION_STATS_CSV)
    return set(stats[stats["dropped"] == True]["uid"].tolist())


class ISLDataset(Dataset):
    """
    PyTorch Dataset for the ISL Continuous Fingerspelling task.

    Each item is one video segment → its landmark sequence + character label.
    The dataset automatically filters:
      - Segments with no valid landmark cache (not yet extracted)
      - Segments with empty labels (all chars were stripped)
      - Segments flagged as dropped in extraction_stats.csv

    Parameters
    ----------
    split  : str    'train', 'val', or 'test'
    normalize : bool  apply wrist-relative normalization (always True for training)
    """

    def __init__(self, split: str, normalize: bool = True):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split     = split
        self.normalize = normalize

        # Load annotations and merge with final split
        ann_df   = pd.read_csv(ANN_CSV)
        split_df = build_final_split()
        df = ann_df.merge(split_df, on="uid", how="inner")
        df = df[df["split"] == split].reset_index(drop=True)

        # Load set of UIDs dropped due to missing landmarks
        dropped = load_dropped_uids()

        # Filter: keep only segments that have a cached .npy AND valid label
        valid_rows = []
        skipped_no_cache  = 0
        skipped_no_label  = 0
        skipped_dropped   = 0

        for _, row in df.iterrows():
            uid  = row["uid"]
            text = str(row["text"]).strip() if pd.notna(row["text"]) else ""

            # Check dropped
            if uid in dropped:
                skipped_dropped += 1
                continue

            # Check cache exists
            npy_path = CACHE_DIR / f"{uid}.npy"
            if not npy_path.exists():
                skipped_no_cache += 1
                continue

            # Check label
            label = encode_label(text)
            if len(label) == 0:
                skipped_no_label += 1
                continue

            valid_rows.append({"uid": uid, "label": label})

        self.samples = valid_rows

        print(f"[ISLDataset] split={split}  "
              f"valid={len(self.samples)}  "
              f"no_cache={skipped_no_cache}  "
              f"no_label={skipped_no_label}  "
              f"dropped={skipped_dropped}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample as a dict.

        Returns
        -------
        dict with:
          'landmarks'  : torch.Tensor  shape (T, 63)
          'label'      : torch.Tensor  shape (L,)  dtype=long
          'uid'        : str
        """
        sample = self.samples[idx]
        uid    = sample["uid"]
        label  = sample["label"]

        landmarks, mask = load_landmarks(uid)

        if self.normalize:
            landmarks = normalize_sequence(landmarks)

        return {
            "landmarks": torch.from_numpy(landmarks),          # (T, 63)
            "label":     torch.tensor(label, dtype=torch.long), # (L,)
            "uid":       uid,
        }


def collate_fn(batch: list) -> dict:
    """
    Custom collate for variable-length sequences.

    Pads landmark sequences to the longest in the batch.
    Pads label sequences to the longest in the batch with -1.

    WHY -1 for label padding?
      PyTorch CTCLoss ignores indices beyond label_lengths[i],
      so the padding value doesn't affect loss. Using -1 makes
      it obviously invalid if accidentally accessed.

    Returns
    -------
    dict:
      'landmarks'     : (B, T_max, 63)
      'labels'        : (B, L_max) padded with -1
      'input_lengths' : (B,) int32  actual frame counts
      'label_lengths' : (B,) int32  actual label lengths
      'uids'          : list of str
    """
    landmarks_list  = [item["landmarks"] for item in batch]  # list of (T_i, 63)
    labels_list     = [item["label"]     for item in batch]  # list of (L_i,)
    uids            = [item["uid"]       for item in batch]

    B = len(batch)
    T_max = max(lm.shape[0] for lm in landmarks_list)
    L_max = max(lb.shape[0] for lb in labels_list)

    # Pad landmarks with zeros (same as missing frames)
    padded_lm = torch.zeros(B, T_max, landmarks_list[0].shape[1])
    for i, lm in enumerate(landmarks_list):
        T_i = lm.shape[0]
        padded_lm[i, :T_i, :] = lm

    # Pad labels with -1
    padded_labels = torch.full((B, L_max), fill_value=-1, dtype=torch.long)
    for i, lb in enumerate(labels_list):
        L_i = lb.shape[0]
        padded_labels[i, :L_i] = lb

    input_lengths = torch.tensor([lm.shape[0] for lm in landmarks_list], dtype=torch.long)
    label_lengths = torch.tensor([lb.shape[0] for lb in labels_list],    dtype=torch.long)

    return {
        "landmarks":     padded_lm,       # (B, T_max, 63)
        "labels":        padded_labels,   # (B, L_max)
        "input_lengths": input_lengths,   # (B,)
        "label_lengths": label_lengths,   # (B,)
        "uids":          uids,
    }


def get_dataloader(split: str, shuffle: bool = None, normalize: bool = True) -> DataLoader:
    """
    Convenience function — returns a DataLoader for a given split.

    Parameters
    ----------
    split     : str   'train', 'val', or 'test'
    shuffle   : bool  default True for train, False otherwise
    normalize : bool

    Returns
    -------
    torch.utils.data.DataLoader
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = ISLDataset(split=split, normalize=normalize)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
def self_test():
    """
    Verify DataLoader works end-to-end.
    Tests shape contracts that the training loop depends on.
    """
    print("Running DataLoader self-test...")
    print("(This requires landmark cache to exist — run landmark_extractor.py first)\n")

    for split in ["train", "val", "test"]:
        try:
            dl = get_dataloader(split, shuffle=False)
            if len(dl.dataset) == 0:
                print(f"  [{split}] SKIP — no cached samples yet")
                continue

            batch = next(iter(dl))
            B = len(batch["uids"])

            lm  = batch["landmarks"]
            lab = batch["labels"]
            il  = batch["input_lengths"]
            ll  = batch["label_lengths"]

            assert lm.ndim  == 3, f"landmarks should be 3D, got {lm.ndim}D"
            assert lm.shape[0]  == B
            assert lm.shape[2]  == 63
            assert lab.shape[0] == B
            assert il.shape[0]  == B
            assert ll.shape[0]  == B
            assert il.max().item() == lm.shape[1], "T_max mismatch"

            # CTC constraint: input_length >= label_length for all samples
            ctc_ok = (il >= ll).all().item()
            assert ctc_ok, "CTC constraint violated: some input_length < label_length"

            print(f"  [{split}] PASSED  samples={len(dl.dataset)}  "
                  f"batch_shape={tuple(lm.shape)}  "
                  f"label_shape={tuple(lab.shape)}  "
                  f"CTC_constraint=OK")
        except Exception as e:
            print(f"  [{split}] ERROR — {e}")

    print("\nDataLoader self-test complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
