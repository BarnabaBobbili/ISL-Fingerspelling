"""
src/evaluation/decoder.py
===========================
Stage 5: CTC decoding — greedy and (optional) beam search.

WHY CTC DECODING IS NON-TRIVIAL
---------------------------------
CTC output is a T-length sequence of character probabilities (including blank).
The raw output contains:
  - Repeated characters (e.g., "aaa" = one 'a' held over 3 frames)
  - Blank tokens separating or filling between characters

Decoding collapses this into the actual character sequence:
  "aaa_b_bb___c" → "abc"  (where _ is blank)

Two decoding strategies:

1. GREEDY DECODE: take argmax at each frame, then collapse.
   Fast, O(T) per sequence. Used for training monitoring and quick eval.
   WHY GREEDY FIRST: It's interpretable and verifiable by hand.

2. BEAM SEARCH DECODE (optional, Stage 8+):
   Maintain top-K candidate sequences at each step.
   Gives lower CER but is O(T × K × vocab) — slower.
   Can incorporate language model scores.

SELF-TEST
---------
  python src/evaluation/decoder.py --self-test
  → Tests against hand-constructed examples with known expected output.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import IDX_TO_CHAR, BLANK_IDX, VOCAB


# ─────────────────────────────────────────────────────────────────────────────
# Greedy decoder
# ─────────────────────────────────────────────────────────────────────────────

def ctc_greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
) -> List[str]:
    """
    CTC greedy decoding: argmax → collapse repeats → remove blanks.

    Parameters
    ----------
    log_probs     : torch.Tensor  shape (T, B, num_classes)
        Log-probabilities as output by the model's log_softmax.
    input_lengths : torch.Tensor  shape (B,)
        Actual (non-padded) sequence length for each sample.

    Returns
    -------
    List[str]  decoded strings, one per batch sample
    """
    # Argmax along class dimension: best token at each time step
    best_tokens = log_probs.argmax(dim=-1)   # shape (T, B)
    best_tokens = best_tokens.permute(1, 0)  # (B, T)

    decoded = []
    for b in range(best_tokens.shape[0]):
        T_b    = input_lengths[b].item()
        tokens = best_tokens[b, :T_b].tolist()
        decoded.append(_collapse_ctc_tokens(tokens))

    return decoded


def _collapse_ctc_tokens(tokens: list) -> str:
    """
    Collapses a flat list of CTC token indices to a string.

    Rules (CTC standard):
      1. Remove consecutive duplicate tokens
      2. Remove blank tokens (index BLANK_IDX)

    Example:
      [0, 0, 26, 1, 26, 26, 1, 2]  →  'abc'
      (a=0, blank=26, b=1, c=2)
    """
    result = []
    prev   = None
    for t in tokens:
        if t != prev:
            if t != BLANK_IDX:
                result.append(IDX_TO_CHAR.get(t, "?"))
            prev = t
    return "".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# Beam search decoder (lightweight, no external library)
# ─────────────────────────────────────────────────────────────────────────────

def ctc_beam_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    beam_width: int = 5,
) -> List[str]:
    """
    Simple CTC beam search decoder (no language model).

    This is the same interface as ctc_greedy_decode — it's a drop-in
    replacement that gives lower CER at the cost of compute time.

    WHY BEAM SEARCH?
      Greedy always picks the single best token at each step, which can
      miss higher-probability sequences reachable via a lower-probability
      intermediate step. Beam search explores multiple paths simultaneously.

    Parameters
    ----------
    log_probs     : (T, B, num_classes)
    input_lengths : (B,)
    beam_width    : int   number of candidate sequences to maintain

    Returns
    -------
    List[str]  one decoded string per sample
    """
    T_full, B, C = log_probs.shape
    # Convert to regular probabilities for beam search (more numerically stable)
    probs = log_probs.exp().cpu().numpy()   # (T, B, C)

    results = []
    for b in range(B):
        T_b = input_lengths[b].item()
        results.append(_beam_search_single(probs[:T_b, b, :], beam_width))

    return results


def _beam_search_single(probs: np.ndarray, beam_width: int) -> str:
    """
    Beam search for a single sequence.

    State: (prefix_string, last_token) → log_probability
    WHY track last_token: needed to correctly handle repeated characters
    (a repeated character is only kept if there's a blank between them).
    """
    # Initial beam: empty prefix, no last token
    # beam = dict mapping (prefix, last_token) → log_prob
    NEG_INF = float("-inf")
    beam    = {("", BLANK_IDX): 0.0}   # log_prob=0 (prob=1)

    T, C = probs.shape
    eps  = 1e-9

    for t in range(T):
        new_beam = {}
        for (prefix, last_tok), log_p in beam.items():
            for c in range(C):
                lp_c = float(np.log(probs[t, c] + eps))
                new_key = None
                new_log_p = log_p + lp_c

                if c == BLANK_IDX:
                    new_key = (prefix, BLANK_IDX)
                elif c == last_tok:
                    # Same character as last — only extend if last was blank
                    new_key = (prefix, c)   # don't append (it's a repeat)
                else:
                    new_key = (prefix + IDX_TO_CHAR.get(c, ""), c)

                if new_key in new_beam:
                    # Merge paths (log-sum-exp)
                    existing = new_beam[new_key]
                    new_beam[new_key] = float(np.logaddexp(existing, new_log_p))
                else:
                    new_beam[new_key] = new_log_p

        # Keep top beam_width entries
        beam = dict(
            sorted(new_beam.items(), key=lambda x: -x[1])[:beam_width]
        )

    # Return the prefix from the best beam entry
    best = max(beam, key=lambda k: beam[k])
    return best[0]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def self_test():
    """
    Tests both decoders against hand-constructed examples with known answers.
    This verifies the decoder logic is correct before using it for evaluation.
    """
    print("Running decoder self-tests...\n")
    passed = 0
    failed = 0

    # Helper: make log_probs from a sequence of token indices
    def make_log_probs(token_seq: list, num_classes: int = 27) -> tuple:
        T = len(token_seq)
        # Create one-hot-ish probs (0.99 for correct token, 0.01 spread)
        probs = np.full((T, 1, num_classes), 0.001 / (num_classes - 1))
        for t, tok in enumerate(token_seq):
            probs[t, 0, :] = 0.001 / (num_classes - 1)
            probs[t, 0, tok] = 0.999
        log_probs    = torch.from_numpy(np.log(probs)).float()   # (T, 1, C)
        input_lengths = torch.tensor([T])
        return log_probs, input_lengths

    B = BLANK_IDX  # 26

    test_cases = [
        # (description, token_sequence, expected_output)
        (
            "Simple word: a-b-c with blanks",
            [0, B, 1, B, 2],                   # a, blank, b, blank, c
            "abc",
        ),
        (
            "Repeated token separated by blank: aa",
            [0, 0, B, 0, 0],                   # a, a, blank, a, a → "aa"
            "aa",
        ),
        (
            "Repeated token without blank: single a",
            [0, 0, 0, 0],                      # a, a, a, a → "a"
            "a",
        ),
        (
            "All blanks → empty string",
            [B, B, B],
            "",
        ),
        (
            "Real word: 'hello'",
            [7, 7, B, 4, 4, B, 11, B, 11, B, 14],  # h, h, _, e, e, _, l, _, l, _, o
            "hello",
        ),
    ]

    for desc, tokens, expected in test_cases:
        log_probs, lengths = make_log_probs(tokens)

        # Test greedy
        greedy_out = ctc_greedy_decode(log_probs, lengths)[0]
        if greedy_out == expected:
            print(f"  GREEDY PASSED [{desc}]: '{greedy_out}'")
            passed += 1
        else:
            print(f"  GREEDY FAILED [{desc}]: expected='{expected}', got='{greedy_out}'")
            failed += 1

        # Test beam search
        beam_out = ctc_beam_decode(log_probs, lengths, beam_width=3)[0]
        if beam_out == expected:
            print(f"  BEAM   PASSED [{desc}]: '{beam_out}'")
            passed += 1
        else:
            print(f"  BEAM   FAILED [{desc}]: expected='{expected}', got='{beam_out}'")
            failed += 1

    print(f"\n  {passed} / {passed+failed} tests passed")
    if failed == 0:
        print("  All decoder self-tests PASSED ✓\n")
    else:
        print("  SOME TESTS FAILED — fix decoder before evaluation!\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
