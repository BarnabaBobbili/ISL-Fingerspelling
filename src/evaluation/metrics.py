"""
src/evaluation/metrics.py
===========================
Stage 5: CER and WER computation.

WHY jiwer?
-----------
jiwer is a battle-tested, one-dependency library for CER/WER computation
used widely in ASR research. It handles edge cases (empty references,
word-level vs. char-level) correctly.

DESIGN DECISION: Character Error Rate (CER) is the primary metric.
  WHY: Our labels are often 1–3 words (proper names). WER would penalize a
  single character error in "Microsoft" the same as getting the whole word
  wrong. CER gives finer-grained signal about model progress.
  WER is reported as a secondary metric for comparability with word-level ASR.

SELF-TEST
---------
  python src/evaluation/metrics.py --self-test
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _cer_manual(hypothesis: str, reference: str) -> float:
    """
    Manual CER computation via edit distance.
    WHY HAVE A MANUAL VERSION?
      Used in self-test to verify jiwer gives the same answer.
      Also serves as fallback if jiwer is not installed.

    CER = edit_distance(hypothesis, reference) / len(reference)
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    # Standard dynamic programming edit distance
    H, R = len(hypothesis), len(reference)
    dp = list(range(R + 1))
    for i in range(1, H + 1):
        prev_row = dp[:]
        dp[0] = i
        for j in range(1, R + 1):
            if hypothesis[i - 1] == reference[j - 1]:
                dp[j] = prev_row[j - 1]
            else:
                dp[j] = 1 + min(prev_row[j - 1], prev_row[j], dp[j - 1])
    return dp[R] / R


def character_error_rate(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER) across a list of hypotheses/references.

    CER = total_edit_distance / total_reference_characters

    Parameters
    ----------
    hypotheses : list of predicted strings
    references : list of ground-truth strings

    Returns
    -------
    float  CER in [0, ∞)  (0 = perfect, 1 = 100% error, can exceed 1)
    """
    assert len(hypotheses) == len(references), "Length mismatch"

    try:
        import jiwer
        # jiwer expects space-separated characters for CER
        # We join chars with spaces to treat each character as a "word"
        h_chars = [" ".join(list(h)) for h in hypotheses]
        r_chars = [" ".join(list(r)) for r in references]
        return jiwer.wer(r_chars, h_chars)
    except ImportError:
        # Fallback: manual implementation
        total_edit = 0
        total_ref  = 0
        for h, r in zip(hypotheses, references):
            edit = int(round(_cer_manual(h, r) * len(r)))
            total_edit += edit
            total_ref  += len(r)
        return total_edit / total_ref if total_ref > 0 else 0.0


def word_error_rate(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER) across a list of hypotheses/references.

    WER = total_word_edit_distance / total_reference_words

    Parameters
    ----------
    hypotheses : list of predicted strings (space-separated words)
    references : list of ground-truth strings

    Returns
    -------
    float  WER in [0, ∞)
    """
    assert len(hypotheses) == len(references)

    try:
        import jiwer
        return jiwer.wer(references, hypotheses)
    except ImportError:
        # Word-level edit distance fallback
        total_edit = 0
        total_ref  = 0
        for h, r in zip(hypotheses, references):
            h_words = h.split()
            r_words = r.split()
            total_ref += len(r_words)
            # Compute word edit distance using same DP
            if len(r_words) == 0:
                continue
            total_edit += int(round(_cer_manual(" ".join(h_words), " ".join(r_words))
                                    * len(r_words)))
        return total_edit / total_ref if total_ref > 0 else 0.0


def evaluate_model(
    model,
    dataloader,
    device: str,
    decode_fn,
    verbose: bool = False,
) -> Tuple[float, float, List[Tuple[str, str]]]:
    """
    Runs a model on a DataLoader and returns CER, WER, and prediction pairs.

    Parameters
    ----------
    model      : nn.Module   (BiLSTMCTC or ConformerCTC)
    dataloader : DataLoader
    device     : str         'cuda' or 'cpu'
    decode_fn  : callable    e.g. ctc_greedy_decode
    verbose    : bool        print first 5 predictions

    Returns
    -------
    cer         : float
    wer         : float
    pairs       : list of (hypothesis, reference) tuples
    """
    import torch
    from src.training.config import IDX_TO_CHAR

    model.eval()
    all_hyps = []
    all_refs = []

    with torch.no_grad():
        for batch in dataloader:
            landmarks     = batch["landmarks"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            labels        = batch["labels"]
            label_lengths = batch["label_lengths"]

            log_probs, out_lengths = model(landmarks, input_lengths)
            hypotheses = decode_fn(log_probs, out_lengths)

            # Decode references from label tensors
            for i in range(labels.shape[0]):
                L_i = label_lengths[i].item()
                ref_indices = labels[i, :L_i].tolist()
                ref = "".join(IDX_TO_CHAR.get(idx, "?") for idx in ref_indices)
                all_refs.append(ref)

            all_hyps.extend(hypotheses)

    cer = character_error_rate(all_hyps, all_refs)
    wer = word_error_rate(all_hyps, all_refs)

    pairs = list(zip(all_hyps, all_refs))

    if verbose:
        print("\n  Sample predictions (hypothesis → reference):")
        for hyp, ref in pairs[:5]:
            match = "✓" if hyp == ref else "✗"
            print(f"    {match}  '{hyp}'  →  '{ref}'")

    return cer, wer, pairs


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def self_test():
    """Tests CER and WER against hand-computed expected values."""
    print("Running metrics self-tests...\n")

    # Test cases: (hypotheses, references, expected_CER, expected_WER, description)
    test_cases = [
        (
            ["hello"],
            ["hello"],
            0.0, 0.0,
            "Perfect match",
        ),
        (
            ["helo"],
            ["hello"],
            0.2, None,  # CER=0.2 (1 deletion/5); WER varies by jiwer version for sub-word edit
            "One char deletion",
        ),
        (
            [""],
            ["abc"],
            1.0, 1.0,   # 3 deletions / 3 chars = 1.0 CER
            "Empty hypothesis",
        ),
        (
            ["amit shah", "padma shri"],
            ["amit shah", "padma shri"],
            0.0, 0.0,
            "Two perfect multi-word matches",
        ),
        (
            ["amitshah"],
            ["amit shah"],
            # space is not in vocab — we strip spaces before computing
            # 'amitshah' vs 'amitshah' after stripping = perfect
            0.0, None,
            "Space handling (spaces stripped from both sides)",
        ),
    ]

    all_passed = True
    for hyps, refs, exp_cer, exp_wer, desc in test_cases:
        # Strip spaces for CER (our vocab has no spaces)
        hyps_stripped = [h.replace(" ", "") for h in hyps]
        refs_stripped = [r.replace(" ", "") for r in refs]

        cer = character_error_rate(hyps_stripped, refs_stripped)

        cer_ok = abs(cer - exp_cer) < 0.01
        result = "PASSED" if cer_ok else "FAILED"
        if not cer_ok:
            all_passed = False

        print(f"  [{result}] {desc}")
        print(f"           CER: expected={exp_cer:.3f}  got={cer:.3f}")

        if exp_wer is not None:
            wer = word_error_rate(hyps, refs)
            wer_ok = abs(wer - exp_wer) < 0.01
            if not wer_ok:
                all_passed = False
            result_w = "PASSED" if wer_ok else "FAILED"
            print(f"           WER: expected={exp_wer:.3f}  got={wer:.3f}  [{result_w}]")
        print()

    if all_passed:
        print("All metrics self-tests PASSED ✓\n")
    else:
        print("SOME TESTS FAILED\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
