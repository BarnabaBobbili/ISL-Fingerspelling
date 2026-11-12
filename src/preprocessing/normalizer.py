"""
src/preprocessing/normalizer.py
=================================
Stage 3: Landmark normalization — translation + scale invariance.

DESIGN DECISION (DD-005)
-------------------------
Two normalization steps, applied in order:

1. TRANSLATION INVARIANCE
   Subtract the wrist landmark (MediaPipe index 0) from all 21 landmarks.
   After this step, the wrist is always at (0, 0, 0).
   WHY: Removes the dependence on where the signer holds their hand in the camera frame.

2. SCALE INVARIANCE
   Divide all coordinates by the Euclidean distance from wrist (index 0)
   to middle-finger MCP joint (index 9).
   WHY: Makes the representation independent of signer hand size and distance from camera.

After normalization the wrist is always (0,0,0), and the wrist-to-MCP-9 distance is always 1.0.
This gives the model a translation- and scale-invariant view of hand shape.

IMPORTANT: Normalization is done AT LOAD TIME (inside the Dataset class), not during extraction.
This way the raw cached .npy files are always in MediaPipe's original coordinate space,
and we can experiment with different normalization strategies without re-running extraction.

SELF-TEST
---------
  python src/preprocessing/normalizer.py --self-test
  python src/preprocessing/normalizer.py --visualize --uid KxqgVJSEiW0_seg000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import WRIST_IDX, SCALE_IDX, NUM_LANDMARKS, FEATURE_DIM, RESULTS_DIR


def normalize_sequence(landmarks: np.ndarray) -> np.ndarray:
    """
    Applies wrist-relative translation + scale normalization to a sequence.

    WHY THIS FUNCTION IS PURE (no side effects, no global state):
      It can be called safely inside a Dataset __getitem__ without any locking,
      making it compatible with multi-process DataLoader (num_workers > 0).

    Parameters
    ----------
    landmarks : np.ndarray
        Shape (T, 63) where T is number of frames.
        Each row is [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].

    Returns
    -------
    np.ndarray  shape (T, 63), normalized.
    All-zero rows (missing frames) remain all-zero — the zero vector
    is the correct representation of "no hand detected" after normalization
    (the wrist would be at origin, which is already zero).
    """
    if landmarks.ndim != 2 or landmarks.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected shape (T, {FEATURE_DIM}), got {landmarks.shape}"
        )

    T = landmarks.shape[0]
    # Reshape to (T, 21, 3) for easier indexing
    lm = landmarks.reshape(T, NUM_LANDMARKS, 3).copy()

    # Step 1: Translation — subtract wrist (index 0) from all landmarks
    # Shape of lm[:, WRIST_IDX, :] = (T, 3); broadcasting subtracts per frame
    wrist = lm[:, WRIST_IDX, :]           # (T, 3)
    lm   -= wrist[:, np.newaxis, :]       # (T, 21, 3) - (T, 1, 3)

    # Step 2: Scale — divide by wrist→MCP9 distance per frame
    # After step 1, wrist is at (0,0,0), so distance = ||lm[:, SCALE_IDX, :]||
    scale = np.linalg.norm(lm[:, SCALE_IDX, :], axis=1)  # (T,)

    # Guard against zero scale (missing frames or degenerate pose)
    # Use 1.0 as fallback to avoid division-by-zero; zero rows stay zero
    safe_scale = np.where(scale > 1e-6, scale, 1.0)
    lm /= safe_scale[:, np.newaxis, np.newaxis]   # (T, 21, 3) / (T, 1, 1)

    return lm.reshape(T, FEATURE_DIM).astype(np.float32)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Single-frame version for use in real-time inference (Stage 10).

    Parameters
    ----------
    frame : np.ndarray  shape (63,)

    Returns
    -------
    np.ndarray  shape (63,), normalized
    """
    return normalize_sequence(frame[np.newaxis, :])[0]


def normalize_hand_static(lm: np.ndarray) -> np.ndarray:
    """
    Canonical wrist-relative + scale normalization for a single static hand.
    Use this everywhere (extraction, webcam inference, eval scripts) to ensure
    training and inference use EXACTLY the same normalization.

    Previously duplicated as _normalize_hand() in multiple scripts — this is
    the single source of truth.

    Parameters
    ----------
    lm : (21, 3) float32 — raw MediaPipe hand landmarks

    Returns
    -------
    (21, 3) float32 — wrist at origin, wrist→MCP9 distance = 1.0
    """
    lm = lm.copy().astype(np.float32)
    wrist = lm[WRIST_IDX]          # (3,)
    lm -= wrist                    # translate so wrist = origin
    scale = np.linalg.norm(lm[SCALE_IDX])
    if scale > 1e-6:
        lm /= scale                # normalize to wrist→MCP distance = 1
    return lm


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

def self_test():
    """
    Unit tests for normalize_sequence.
    Each test checks a known property of the normalization.
    Run with: python src/preprocessing/normalizer.py --self-test
    """
    print("Running normalizer self-tests...")

    # Test 1: After normalization, wrist should be at (0,0,0) for all frames
    T = 10
    fake = np.random.rand(T, FEATURE_DIM).astype(np.float32)
    normed = normalize_sequence(fake)
    normed_reshaped = normed.reshape(T, NUM_LANDMARKS, 3)
    wrist_coords = normed_reshaped[:, WRIST_IDX, :]
    assert np.allclose(wrist_coords, 0.0, atol=1e-5), \
        f"FAIL Test 1: wrist should be (0,0,0), got max={np.abs(wrist_coords).max():.6f}"
    print("  Test 1 PASSED: Wrist is (0,0,0) for all frames")

    # Test 2: After normalization, wrist→MCP9 distance should be 1.0 for all frames
    # (for frames where both landmarks are non-zero)
    mcp9_coords = normed_reshaped[:, SCALE_IDX, :]
    distances = np.linalg.norm(mcp9_coords, axis=1)   # (T,)
    assert np.allclose(distances, 1.0, atol=1e-4), \
        f"FAIL Test 2: wrist→MCP9 dist should be 1.0, got {distances}"
    print("  Test 2 PASSED: Wrist→MCP9 distance = 1.0 for all frames")

    # Test 3: All-zero input (missing frame) stays all-zero after normalization
    zero_seq = np.zeros((5, FEATURE_DIM), dtype=np.float32)
    normed_zero = normalize_sequence(zero_seq)
    assert np.allclose(normed_zero, 0.0), "FAIL Test 3: zero input should stay zero"
    print("  Test 3 PASSED: Zero (missing) frames remain zero after normalization")

    # Test 4: Output shape matches input shape
    seq = np.random.rand(7, FEATURE_DIM).astype(np.float32)
    out = normalize_sequence(seq)
    assert out.shape == (7, FEATURE_DIM), f"FAIL Test 4: shape mismatch {out.shape}"
    print("  Test 4 PASSED: Output shape preserved")

    # Test 5: Translation invariance — shift entire hand, result should be same
    shift = np.array([0.1, 0.2, 0.05] * NUM_LANDMARKS, dtype=np.float32)
    seq_shifted = fake + shift[np.newaxis, :]
    normed_shifted = normalize_sequence(seq_shifted)
    assert np.allclose(normed, normed_shifted, atol=1e-4), \
        "FAIL Test 5: normalization not translation-invariant"
    print("  Test 5 PASSED: Normalization is translation-invariant")

    print("\nAll normalizer self-tests PASSED ✓\n")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION (Stage 3 checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_normalization(uid: str):
    """
    Plots raw vs. normalized x,y trajectories for selected landmarks.

    WHAT TO LOOK FOR:
      - Raw: landmarks drift across the frame as the signer moves
      - Normalized: landmarks stay centered near (0,0); only finger shape changes
      - No sudden jumps or discontinuities (which would indicate a normalization bug)
    """
    from src.extraction.landmark_extractor import load_landmarks
    raw, mask = load_landmarks(uid)

    # Only use valid frames for visualization
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        print(f"No valid frames in {uid}")
        return

    raw_valid  = raw[valid_idx]
    norm_valid = normalize_sequence(raw_valid)

    T = len(valid_idx)
    # Plot x, y of index fingertip (landmark 8) over time
    raw_reshaped  = raw_valid.reshape(T, NUM_LANDMARKS, 3)
    norm_reshaped = norm_valid.reshape(T, NUM_LANDMARKS, 3)

    landmarks_to_plot = {
        "Wrist (0)": 0,
        "Index Tip (8)": 8,
        "Middle Tip (12)": 12,
        "Ring Tip (16)": 16,
        "Pinky Tip (20)": 20,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Normalized vs Raw Landmarks — {uid}", fontsize=13)

    colors = ["blue", "red", "green", "orange", "purple"]
    for ax_row, coord_idx, coord_name in [(0, 0, "X"), (1, 1, "Y")]:
        for ax_col, (raw_flag, label) in enumerate([(True, "RAW"), (False, "NORMALIZED")]):
            ax = axes[ax_row, ax_col]
            data = raw_reshaped if raw_flag else norm_reshaped
            for (lm_name, lm_idx), color in zip(landmarks_to_plot.items(), colors):
                ax.plot(data[:, lm_idx, coord_idx], label=lm_name,
                        color=color, linewidth=1.0, alpha=0.8)
            ax.set_title(f"{coord_name} coordinate — {label}")
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{coord_name} ({'normalized' if not raw_flag else 'image space'})")
            if ax_row == 0 and ax_col == 0:
                ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"stage3_normalization_{uid}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalizer self-test and visualization")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--uid", type=str, default=None)
    args = parser.parse_args()

    if args.self_test:
        self_test()

    if args.visualize:
        if not args.uid:
            import pandas as pd
            from src.training.config import ANN_CSV
            uid = pd.read_csv(ANN_CSV)["uid"].iloc[0]
            print(f"No --uid specified, using: {uid}")
        else:
            uid = args.uid
        visualize_normalization(uid)

    if not args.self_test and not args.visualize:
        parser.print_help()
