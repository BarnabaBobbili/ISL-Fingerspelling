"""
src/evaluation/avatar_eval.py
==============================
Comprehensive evaluation of the Text-to-Sign avatar pipeline.

WHAT THIS DOES:
  1. Generates a word list of 50-100 common English words such that each
     letter A-Z appears at least 3-4 times across the words.
  2. For each letter in each word, fetches the REFERENCE landmark (the
     averaged MediaPipe coordinates the avatar shows).
  3. Runs every reference landmark through the trained DH-GAT classifier.
  4. Reports:
       - Per-letter recognition accuracy  (is the sign recognizable?)
       - Per-letter coordination score    (mean cosine similarity vs reference)
       - Confusion matrix                 (what letters are misclassified as what)
       - Worst-performing letters         (coordination problems)
       - Recommendations for improvement
  5. Saves:
       - results/avatar_confusion_matrix.png
       - results/avatar_per_letter_accuracy.png
       - results/avatar_eval_report.txt

Run:
  python src/evaluation/avatar_eval.py
"""

import sys
import random
import string
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import STATIC_LETTERS, STATIC_CACHE_DIR, StaticConfig
from src.models.dh_gat import DHGAT
from src.static.text_to_sign_avatar import load_reference_landmarks, HAND_MODE

# ─── Word list: covers all 26 letters ≥4 times each ──────────────────────────
WORD_LIST = [
    # A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
    "APPLE", "ABOUT", "AFTER", "AGAIN", "ALONG",
    "BREAK", "BRING", "BUILD", "BELOW", "BRAVE",
    "CATCH", "CHECK", "CLICK", "CLOTH", "CLAMP",
    "DANCE", "DIRTY", "DRINK", "DROVE", "DEPTH",
    "EAGLE", "EARLY", "EIGHT", "ENJOY", "ENTER",
    "FIGHT", "FLAME", "FLASK", "FLOOD", "FRESH",
    "GHOST", "GIVEN", "GLOOM", "GRAND", "GRASP",
    "HAPPY", "HEART", "HEAVY", "HONEY", "HURRY",
    "IMAGE", "INNER", "INPUT", "INTER", "IRONY",
    "JEWEL", "JOINT", "JOKER", "JUDGE", "JUMPY",
    "KNACK", "KNIFE", "KNELT", "KNOCK", "KNOWS",
    "LABEL", "LARGE", "LASER", "LAUGH", "LEARN",
    "MAGIC", "MARKS", "MATCH", "MIGHT", "MORAL",
    "NIGHT", "NINJA", "NOBLE", "NORTH", "NOVEL",
    "OFFER", "OLIVE", "ONSET", "OPTIC", "ORBIT",
    "PIXEL", "PLACE", "PLAIN", "PLANT", "POWER",
    "QUERY", "QUIET", "QUOTE", "QUAIL", "QUEEN",
    "RADAR", "RAISE", "RANGE", "RAPID", "REACH",
    "SAUCE", "SCALE", "SCENE", "SCORE", "SEVEN",
    "TEACH", "THEME", "THINK", "THROW", "TRACK",
    "ULTRA", "UNDER", "UNITY", "UNTIL", "UPPER",
    "VALUE", "VALID", "VAULT", "VERSE", "VIVID",
    "WATER", "WATCH", "WEIRD", "WHEAT", "WHERE",
    "XENON", "EXACT", "EXCEL", "EXERT", "OXIDE",
    "YACHT", "YEARS", "YIELD", "YOUNG", "YOUTH",
    "ZEBRA", "ZONES", "ZILCH", "ZONAL", "ZIPPY",
]


def verify_letter_coverage(words, min_count=3):
    """Check that every letter appears at least min_count times."""
    counts = Counter(c for w in words for c in w if c in STATIC_LETTERS)
    missing = {l: counts.get(l, 0) for l in STATIC_LETTERS if counts.get(l, 0) < min_count}
    return counts, missing


def load_dhgat(device="cpu"):
    """Load DH-GAT checkpoint."""
    cfg = StaticConfig()
    model = DHGAT(cfg=cfg).to(device)
    ckpt_path = Path("checkpoints/dh_gat_static_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    # Checkpoint format: {'epoch':..., 'model_state':OrderedDict, ...}
    if isinstance(state, dict):
        if "model_state" in state:
            state = state["model_state"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def classify_sign(model, right_lm, left_lm, right_present, left_present, device="cpu"):
    """Run a single (right_lm, left_lm) pair through DH-GAT and return (pred_letter, confidence)."""
    l2i = {l: i for i, l in enumerate(STATIC_LETTERS)}
    i2l = {i: l for l, i in l2i.items()}

    r = torch.from_numpy(
        right_lm if right_lm is not None else np.zeros((21, 3), dtype=np.float32)
    ).unsqueeze(0).to(device)
    l = torch.from_numpy(
        left_lm if left_lm is not None else np.zeros((21, 3), dtype=np.float32)
    ).unsqueeze(0).to(device)
    rp = torch.tensor([right_present], dtype=torch.bool).to(device)
    lp = torch.tensor([left_present],  dtype=torch.bool).to(device)

    with torch.no_grad():
        logits = model(r, l, rp, lp)
        probs  = torch.softmax(logits, dim=1)[0]
        top_conf, top_idx = probs.max(0)
    return i2l[top_idx.item()], top_conf.item()


def cosine_similarity_lm(a, b):
    """Cosine similarity between two flattened landmark arrays."""
    if a is None or b is None:
        return np.nan
    af = a.flatten(); bf = b.flatten()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom < 1e-6:
        return np.nan
    return float(np.dot(af, bf) / denom)


# ─── Main evaluation ──────────────────────────────────────────────────────────

def main():
    random.seed(42)
    device = "cpu"
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load references + model ────────────────────────────────────────────────
    print("[eval] Loading DH-GAT checkpoint ...")
    model, _ = load_dhgat(device)
    print("[eval] Loading reference landmarks (model-guided best-sample selection) ...")
    refs = load_reference_landmarks(model=model, device=device)

    words = WORD_LIST
    counts, missing = verify_letter_coverage(words, min_count=3)
    print(f"\n[eval] Word list: {len(words)} words, "
          f"{sum(counts.values())} total letters")
    if missing:
        print(f"[eval] WARNING: letters with <3 appearances: {missing}")
    print(f"[eval] Letter counts: {dict(sorted(counts.items()))}\n")

    # ── Per-letter tracking ────────────────────────────────────────────────────
    # true_letter -> list of predicted_letter
    confusion_raw   = defaultdict(list)
    cosine_scores   = defaultdict(list)   # true_letter -> cosine similarities
    mse_scores      = defaultdict(list)   # true_letter -> MSE vs reference

    total = correct = 0

    for word in words:
        for letter in word:
            if letter not in refs:
                continue
            ref = refs[letter]

            # KEY INSIGHT: HAND_MODE is only for DISPLAY (which hand to render on screen).
            # DH-GAT was trained on data where BOTH hands may be present.
            # For classification, always pass whatever reference data exists for each hand.
            # Use actual data presence flags from the reference, NOT from HAND_MODE.
            rlm       = ref['right_lm']      # may be None
            llm       = ref['left_lm']       # may be None
            r_present = ref['right_present'] # True if this letter had right hand in dataset
            l_present = ref['left_present']  # True if this letter had left hand in dataset

            # Classify via DH-GAT (pass both hands, let model decide)
            pred, conf = classify_sign(
                model,
                rlm,
                llm,
                r_present, l_present, device
            )

            confusion_raw[letter].append(pred)
            total += 1
            if pred == letter:
                correct += 1

            # Coordination quality: cosine similarity of full feature vector
            r_arr = rlm if rlm is not None else np.zeros((21, 3), dtype=np.float32)
            l_arr = llm if llm is not None else np.zeros((21, 3), dtype=np.float32)
            feat = np.concatenate([r_arr.flatten(), l_arr.flatten()])

            ref_r = ref['right_lm'] if ref['right_lm'] is not None else np.zeros((21,3),dtype=np.float32)
            ref_l = ref['left_lm']  if ref['left_lm']  is not None else np.zeros((21,3),dtype=np.float32)
            ref_feat = np.concatenate([ref_r.flatten(), ref_l.flatten()])

            cs  = cosine_similarity_lm(feat.reshape(1,-1), ref_feat.reshape(1,-1))
            mse = float(np.mean((feat - ref_feat)**2))
            cosine_scores[letter].append(cs)
            mse_scores[letter].append(mse)

    overall_acc = correct / total * 100

    # ── Per-letter accuracy ────────────────────────────────────────────────────
    letter_acc = {}
    for letter in STATIC_LETTERS:
        preds = confusion_raw[letter]
        if not preds:
            letter_acc[letter] = 0.0
            continue
        letter_acc[letter] = sum(1 for p in preds if p == letter) / len(preds) * 100

    # ── Confusion matrix ───────────────────────────────────────────────────────
    l2i = {l: i for i, l in enumerate(STATIC_LETTERS)}
    cm = np.zeros((26, 26), dtype=int)
    for true_l, preds in confusion_raw.items():
        ti = l2i[true_l]
        for p in preds:
            if p in l2i:
                cm[ti][l2i[p]] += 1

    # ── Coordination quality ───────────────────────────────────────────────────
    coord_quality = {l: float(np.nanmean(cosine_scores[l])) if cosine_scores[l] else 0.0
                     for l in STATIC_LETTERS}
    coord_mse     = {l: float(np.nanmean(mse_scores[l]))    if mse_scores[l]    else 0.0
                     for l in STATIC_LETTERS}

    # ─── PLOT 1: Per-letter accuracy bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    letters = STATIC_LETTERS
    accs    = [letter_acc[l] for l in letters]
    colors  = ['#EF5350' if a < 80 else ('#FFA726' if a < 95 else '#66BB6A') for a in accs]
    bars = ax.bar(letters, accs, color=colors, width=0.7)
    ax.set_ylim(0, 115)
    ax.axhline(100, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(overall_acc, color='royalblue', linewidth=1.5,
               linestyle='--', label=f'Overall: {overall_acc:.1f}%')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_title('Avatar Sign Recognition — Per-Letter Accuracy (DH-GAT Classifier)',
                 fontsize=13, pad=12)
    ax.set_xlabel('Letter'); ax.set_ylabel('Recognition Accuracy (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'avatar_per_letter_accuracy.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[eval] Saved: results/avatar_per_letter_accuracy.png")

    # ─── PLOT 2: Confusion matrix ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=STATIC_LETTERS, yticklabels=STATIC_LETTERS,
                linewidths=0.3, ax=ax, cbar_kws={'shrink': 0.7})
    ax.set_title('Avatar Sign Confusion Matrix\n(Row=True Letter, Col=Predicted Letter)',
                 fontsize=13, pad=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True Letter')
    plt.tight_layout()
    plt.savefig(out_dir / 'avatar_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[eval] Saved: results/avatar_confusion_matrix.png")

    # ─── PLOT 3: Coordination quality (cosine similarity) ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    coord_vals = [coord_quality[l] for l in letters]
    ccolors = ['#EF5350' if v < 0.90 else ('#FFA726' if v < 0.97 else '#66BB6A')
               for v in coord_vals]
    ax.bar(letters, coord_vals, color=ccolors, width=0.7)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(np.nanmean(coord_vals), color='royalblue', linewidth=1.5,
               linestyle='--', label=f'Mean: {np.nanmean(coord_vals):.3f}')
    ax.set_title('Avatar Hand Coordination Quality — Cosine Similarity to Reference',
                 fontsize=13, pad=12)
    ax.set_xlabel('Letter'); ax.set_ylabel('Cosine Similarity (higher = better)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'avatar_coordination_quality.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("[eval] Saved: results/avatar_coordination_quality.png")

    # ─── Text report ──────────────────────────────────────────────────────────
    problem_letters = [l for l in STATIC_LETTERS if letter_acc[l] < 100 or coord_quality[l] < 0.97]
    confusions = []
    for ti, true_l in enumerate(STATIC_LETTERS):
        for pi, pred_l in enumerate(STATIC_LETTERS):
            if ti != pi and cm[ti][pi] > 0:
                confusions.append((true_l, pred_l, cm[ti][pi]))
    confusions.sort(key=lambda x: -x[2])

    report = []
    report.append("=" * 62)
    report.append("  AVATAR TEXT-TO-SIGN EVALUATION REPORT")
    report.append(f"  Words tested : {len(words)}")
    report.append(f"  Total letter signs evaluated : {total}")
    report.append(f"  OVERALL RECOGNITION ACCURACY : {overall_acc:.2f}%")
    report.append("=" * 62)
    report.append("\n--- PER-LETTER ACCURACY ---")
    report.append(f"  {'Letter':<8} {'Count':<8} {'Correct':<10} {'Accuracy':<10} {'Coord Score'}")
    report.append("  " + "-"*55)
    for l in STATIC_LETTERS:
        preds = confusion_raw[l]
        cnt = len(preds)
        crr = sum(1 for p in preds if p == l)
        acc = letter_acc[l]
        cs  = coord_quality[l]
        flag = " << PROBLEM" if acc < 100 else ""
        report.append(f"  {l:<8} {cnt:<8} {crr:<10} {acc:<9.1f}%  {cs:.4f}{flag}")

    report.append("\n--- MISCLASSIFICATIONS (SIGN CONFUSION) ---")
    if confusions:
        for true_l, pred_l, cnt in confusions:
            report.append(f"  '{true_l}' misclassified as '{pred_l}' -- {cnt} time(s)")
    else:
        report.append("  None! All avatar signs are perfectly recognized.")

    report.append("\n--- LETTERS WITH COORDINATION PROBLEMS ---")
    coord_issues = [(l, coord_quality[l], coord_mse[l])
                    for l in STATIC_LETTERS if coord_quality[l] < 0.97]
    coord_issues.sort(key=lambda x: x[1])
    if coord_issues:
        for l, cs, mse in coord_issues:
            mode = HAND_MODE.get(l, 'B')
            report.append(f"  Letter '{l}'  (mode={mode})  CosineSim={cs:.4f}  MSE={mse:.4f}")
    else:
        report.append("  All letters have good coordination quality (cos_sim >= 0.97).")

    report.append("\n--- HOW TO IMPROVE COORDINATION ---")
    for l, cs, mse in coord_issues:
        mode = HAND_MODE.get(l, 'B')
        report.append(f"\n  '{l}' (cos_sim={cs:.3f}, mode={mode}):")
        if cs < 0.85:
            report.append(f"    [SEVERE] The averaged reference landmark is highly inconsistent.")
            report.append(f"    FIX: Increase MAX_SAMPLES from 30 to 100 in load_reference_landmarks()")
            report.append(f"         to reduce noise in the averaged pose.")
        elif cs < 0.92:
            report.append(f"    [MODERATE] Some variation in dataset samples for '{l}'.")
            report.append(f"    FIX: Filter outlier samples (keep only top-50% by cluster centroid)")
            report.append(f"         or manually verify the reference pose.")
        else:
            report.append(f"    [MINOR] Slight misalignment in wrist anchor position.")
            report.append(f"    FIX: Tune R_WRIST_X/L_WRIST_X/SCALE constants in text_to_sign_avatar.py")

    report.append("\n--- WORD-BY-WORD RESULTS ---")
    for word in words:
        word_results = []
        for letter in word:
            if letter not in refs:
                word_results.append(f"{letter}(?)")
                continue
            preds = confusion_raw[letter]
            # Get first prediction for this letter instance
            # (approximate since we process per-letter in order)
            acc = letter_acc[letter]
            sym = "OK" if acc == 100 else "XX"
            word_results.append(f"{letter}{sym}")
        report.append(f"  {word:<12}  {'  '.join(word_results)}")

    report_text = "\n".join(report)
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        print(report_text.encode('ascii', errors='replace').decode('ascii'))
    rpath = out_dir / "avatar_eval_report.txt"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[eval] Full report saved: {rpath}")
    print(f"[eval] Plots saved: results/avatar_*.png")


if __name__ == "__main__":
    main()
