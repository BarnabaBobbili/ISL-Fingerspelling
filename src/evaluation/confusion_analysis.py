"""
src/evaluation/confusion_analysis.py
=======================================
Stage 5: Character-level confusion matrix and error analysis.

PURPOSE
-------
After evaluation, this script:
  1. Builds a character confusion matrix from (hypothesis, reference) pairs
  2. Plots it as a heatmap — this is a REQUIRED THESIS FIGURE
  3. Reports the most confused character pairs with frequency
  4. Computes per-character error rates (which letters the model struggles with)

WHY CONFUSION ANALYSIS IS REQUIRED
-------------------------------------
A committee member WILL ask: "Which signs are most confused and why?"
The expected answer is: "Visually similar handshapes in ISL, such as M/N or
S/A, are confused more often because their landmark patterns are close in
Euclidean space." The confusion matrix lets you point to specific pairs.

USAGE
-----
  python src/evaluation/confusion_analysis.py \
      --pairs-file results/bilstm_test_pairs.csv \
      --out-prefix results/stage5_bilstm
"""

import argparse
import sys
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import VOCAB, RESULTS_DIR


def build_char_confusion_matrix(
    pairs: List[Tuple[str, str]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Builds a character confusion matrix from (hypothesis, reference) pairs.

    Uses dynamic-programming character alignment to handle insertions/deletions.
    Each (predicted_char, true_char) pair increments the matrix cell.

    WHY ALIGNMENT FIRST?
      Simply counting character co-occurrences without alignment would give
      wrong results when the hypothesis has a different length. We need to
      know which predicted char corresponds to which reference char.

    Parameters
    ----------
    pairs : list of (hypothesis, reference) tuples

    Returns
    -------
    matrix : np.ndarray  shape (V, V)  where V = len(VOCAB)
        matrix[i, j] = count of times char j (reference) was predicted as char i (hypothesis)
    labels : list of str  character labels (a–z)
    """
    labels = VOCAB  # 26 chars
    V      = len(labels)
    char_to_i = {c: i for i, c in enumerate(labels)}
    matrix = np.zeros((V, V), dtype=int)

    for hyp, ref in pairs:
        # Align hypothesis to reference using edit distance traceback
        alignment = _align_sequences(hyp, ref)
        for h_char, r_char in alignment:
            if r_char in char_to_i and h_char in char_to_i:
                # Substitution: reference char r_char, predicted as h_char
                matrix[char_to_i[h_char], char_to_i[r_char]] += 1
            # Insertions and deletions go to a separate count (not shown in matrix)

    return matrix, labels


def _align_sequences(hyp: str, ref: str) -> List[Tuple[str, str]]:
    """
    Returns character-level alignment between hypothesis and reference
    using standard edit distance with traceback.

    Returns list of (hyp_char, ref_char) tuples for matched/substituted positions.
    Insertions (hyp has char, ref doesn't) and deletions are skipped.
    """
    H, R = len(hyp), len(ref)
    if H == 0 or R == 0:
        return []

    # DP table
    dp = [[0] * (R + 1) for _ in range(H + 1)]
    for i in range(H + 1):
        dp[i][0] = i
    for j in range(R + 1):
        dp[0][j] = j
    for i in range(1, H + 1):
        for j in range(1, R + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Traceback
    alignment = []
    i, j = H, R
    while i > 0 and j > 0:
        if hyp[i - 1] == ref[j - 1]:
            alignment.append((hyp[i - 1], ref[j - 1]))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            alignment.append((hyp[i - 1], ref[j - 1]))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            # Deletion (ref char not predicted)
            i -= 1
        else:
            # Insertion (extra char in hyp)
            j -= 1

    return alignment


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    normalize: bool = True,
):
    """
    Plots the confusion matrix as a heatmap.

    Parameters
    ----------
    matrix    : (V, V) confusion counts
    labels    : list of char labels
    title     : plot title
    out_path  : where to save the PNG
    normalize : if True, normalize rows by reference count (shows recall per class)
    """
    if normalize:
        row_sums = matrix.sum(axis=0, keepdims=True) + 1e-9  # avoid div-by-zero
        plot_data = matrix / row_sums
        fmt_label = "Normalized Prediction Rate"
    else:
        plot_data = matrix.astype(float)
        fmt_label = "Count"

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(plot_data, interpolation="nearest", cmap="Blues", aspect="auto")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=fmt_label)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Reference Character", fontsize=12)
    ax.set_ylabel("Predicted Character", fontsize=12)
    ax.set_title(title, fontsize=13)

    # Annotate cells on the diagonal (correct predictions) in bold
    thresh = plot_data.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = plot_data[i, j]
            if val > 0.01:  # only annotate non-trivial cells
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:.2f}" if normalize else str(int(val)),
                        ha="center", va="center", fontsize=6, color=color)

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[confusion] Confusion matrix saved to {out_path}")


def report_top_confusions(matrix: np.ndarray, labels: List[str], top_k: int = 10):
    """
    Prints the most confused character pairs (off-diagonal matrix cells).

    This directly answers the thesis question: "Which letters are confused most?"
    """
    V = len(labels)
    pairs = []
    for i in range(V):
        for j in range(V):
            if i != j and matrix[i, j] > 0:
                pairs.append((matrix[i, j], labels[i], labels[j]))

    pairs.sort(reverse=True)

    print(f"\n  Top {top_k} Confusion Pairs (predicted → reference : count)")
    print("  " + "-" * 45)
    for count, pred, ref in pairs[:top_k]:
        print(f"    '{pred}' predicted as '{ref}'  :  {count} times")
    print()


def per_char_accuracy(matrix: np.ndarray, labels: List[str]) -> dict:
    """
    Computes per-character recognition accuracy.
    accuracy[c] = matrix[c, c] / total_reference_occurrences_of_c
    """
    acc = {}
    for j, c in enumerate(labels):
        total = matrix[:, j].sum()
        if total > 0:
            acc[c] = matrix[j, j] / total
        else:
            acc[c] = None   # character not seen in reference
    return acc


def run_confusion_analysis(pairs: List[Tuple[str, str]], out_prefix: str):
    """
    Full confusion analysis pipeline. Called after evaluation.

    Parameters
    ----------
    pairs      : list of (hypothesis, reference) — from evaluate_model()
    out_prefix : output path prefix (e.g., 'results/stage5_bilstm')
    """
    # Strip spaces (our vocab has no space)
    clean_pairs = [(h.replace(" ", ""), r.replace(" ", "")) for h, r in pairs]

    matrix, labels = build_char_confusion_matrix(clean_pairs)

    # Save normalized heatmap
    plot_confusion_matrix(
        matrix, labels,
        title=f"Character Confusion Matrix — {Path(out_prefix).name}",
        out_path=Path(out_prefix + "_confusion_normalized.png"),
        normalize=True,
    )
    # Save raw count heatmap
    plot_confusion_matrix(
        matrix, labels,
        title=f"Character Confusion Matrix (counts) — {Path(out_prefix).name}",
        out_path=Path(out_prefix + "_confusion_counts.png"),
        normalize=False,
    )

    report_top_confusions(matrix, labels, top_k=10)

    # Per-character accuracy
    acc = per_char_accuracy(matrix, labels)
    print("  Per-character accuracy (letters seen in reference):")
    for c, a in sorted(acc.items(), key=lambda x: (x[1] is None, x[1] or 0)):
        if a is not None:
            print(f"    '{c}'  :  {100*a:.1f}%")

    # Save pairs CSV for record-keeping
    pairs_csv = Path(out_prefix + "_pairs.csv")
    with open(pairs_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hypothesis", "reference"])
        writer.writerows(clean_pairs)
    print(f"\n  Pairs saved to {pairs_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=str, required=True,
                        help="CSV with 'hypothesis,reference' columns")
    parser.add_argument("--out-prefix", type=str,
                        default=str(RESULTS_DIR / "confusion"),
                        help="Output file prefix")
    args = parser.parse_args()

    import csv
    with open(args.pairs_file) as f:
        reader = csv.DictReader(f)
        pairs = [(row["hypothesis"], row["reference"]) for row in reader]

    run_confusion_analysis(pairs, args.out_prefix)
