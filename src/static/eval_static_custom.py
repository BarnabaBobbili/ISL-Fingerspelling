"""
src/static/eval_static_custom.py
=================================
Evaluates the trained DH-GAT static classifier on a custom image dataset.
The dataset should be organized into folders by letter (e.g., A/, B/, ..., Z/).

This script runs MediaPipe on each image to extract landmarks, passes them
to the loaded DH-GAT model, and calculates accuracy.

Usage:
  python src/static/eval_static_custom.py --data-dir "data/static-frames-extracted"
  python src/static/eval_static_custom.py --data-dir "path/to/another/dataset" --checkpoint "checkpoints/dh_gat_static_best.pt"
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    StaticConfig, STATIC_LETTERS, LETTER_TO_IDX, IDX_TO_LETTER,
    NUM_LANDMARKS, LANDMARK_DIM, MP_MAX_NUM_HANDS_STATIC, ROOT,
)
from src.preprocessing.normalizer import normalize_hand_static
from src.models.dh_gat import build_dhgat
from src.static.train_static import compute_per_class_accuracy, save_per_class_bar, save_confusion_matrix

TASK_MODEL_PATH = ROOT / "hand_landmarker.task"

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe & Normalization (Same logic as extraction)
# ─────────────────────────────────────────────────────────────────────────────
def _build_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    if not TASK_MODEL_PATH.exists():
        print(f"[ERROR] MediaPipe task model not found at {TASK_MODEL_PATH}")
        sys.exit(1)

    base_options = mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=MP_MAX_NUM_HANDS_STATIC,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def extract_landmarks(img_path: Path, detector) -> dict:
    import mediapipe as mp
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    right_lm = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    left_lm = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    right_present = False
    left_present = False

    if result.hand_landmarks:
        for i, handedness_list in enumerate(result.handedness):
            label = handedness_list[0].display_name
            lms = result.hand_landmarks[i]
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            arr = normalize_hand_static(arr)

            if label == "Right":
                right_lm = arr
                right_present = True
            else:
                left_lm = arr
                left_present = True

    return {
        "right_lm": right_lm,
        "left_lm": left_lm,
        "right_present": right_present,
        "left_present": left_present,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_custom_dataset(data_dir: str, checkpoint_path: str, verbose: bool = False):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Dataset directory not found: {data_path}")
        return

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_file}")
        return

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = StaticConfig()
    model = build_dhgat(cfg).to(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[info] Loaded model from {ckpt_file} onto {device}")

    # Build Detector
    detector = _build_detector()

    # Collect images
    image_paths = []
    for letter in STATIC_LETTERS:
        folder = data_path / letter
        if folder.is_dir():
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                for img_file in folder.glob(ext):
                    image_paths.append((img_file, letter))
                for img_file in folder.glob(ext.upper()):
                    image_paths.append((img_file, letter))

    if not image_paths:
        print(f"[ERROR] No images found in {data_path}. Ensure it has A-Z subfolders.")
        return

    print(f"[info] Found {len(image_paths)} images in {data_path}")
    print("[info] Extracting landmarks and evaluating...")

    # pairs stores (image_name, predicted, true_label)
    pairs = []
    skipped_imgs = []
    skipped = 0

    t0 = time.time()
    with torch.no_grad():
        for i, (img_path, true_label) in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(image_paths)} images...")

            res = extract_landmarks(img_path, detector)
            if res is None or (not res["right_present"] and not res["left_present"]):
                skipped += 1
                skipped_imgs.append((img_path.name, true_label))
                continue

            # Convert to tensors
            right_lm = torch.from_numpy(res["right_lm"]).unsqueeze(0).to(device)
            left_lm = torch.from_numpy(res["left_lm"]).unsqueeze(0).to(device)
            right_present = torch.tensor([res["right_present"]], dtype=torch.bool).to(device)
            left_present = torch.tensor([res["left_present"]], dtype=torch.bool).to(device)

            # Predict
            logits = model(right_lm, left_lm, right_present, left_present)
            pred_idx = logits.argmax(dim=1).item()
            pred_letter = IDX_TO_LETTER[pred_idx]

            pairs.append((img_path.name, pred_letter, true_label))

    elapsed = time.time() - t0
    detector.close()

    if not pairs:
        print("[ERROR] No valid landmarks could be extracted from any image.")
        return

    # Results — pairs is (img_name, pred, true)
    correct = sum(1 for _, p, t in pairs if p == t)
    total = len(pairs)
    acc = correct / total

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Dataset        : {data_path}")
    print(f"  Total Images   : {len(image_paths)}")
    print(f"  Hands Detected : {total}")
    print(f"  Skipped (No H) : {skipped}")
    print(f"  Overall Acc    : {correct}/{total} ({100*acc:.2f}%)")
    print(f"  Time taken     : {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # compute_per_class_accuracy expects (pred, true) tuples
    pairs_for_stats = [(p, t) for _, p, t in pairs]
    per_class = compute_per_class_accuracy(pairs_for_stats)
    print("  Per-class accuracy:")
    for lt, c_acc in sorted(per_class.items(), key=lambda x: x[1]):
        bar = "#" * int(c_acc * 20)
        print(f"    '{lt}'  {100*c_acc:5.1f}%  {bar}")

    # ── Verbose: per-class image breakdown ───────────────────────────────────
    if verbose:
        print(f"\n{'='*70}")
        print(f"  DETAILED PREDICTIONS PER CLASS")
        print(f"{'='*70}")
        from collections import defaultdict
        class_groups = defaultdict(list)
        for img_name, pred, true in pairs:
            class_groups[true].append((img_name, pred))
        # Also show skipped
        skip_groups = defaultdict(list)
        for img_name, true in skipped_imgs:
            skip_groups[true].append(img_name)

        for lt in sorted(class_groups.keys()):
            items = class_groups[lt]
            c_correct = sum(1 for _, p in items if p == lt)
            print(f"\n  [{lt}]  {c_correct}/{len(items)} correct")
            print(f"  {'Image':<30}  {'Expected':<10}  {'Predicted':<10}  Result")
            print(f"  {'-'*65}")
            for img_name, pred in sorted(items):
                mark = "OK" if pred == lt else "WRONG"
                print(f"  {img_name:<30}  {lt:<10}  {pred:<10}  {mark}")
            if skip_groups[lt]:
                print(f"  -- {len(skip_groups[lt])} image(s) skipped (no hand detected): {', '.join(skip_groups[lt][:3])}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_prefix = data_path.name
    csv_path = results_dir / f"custom_{out_prefix}_predictions.csv"
    with open(csv_path, "w") as f:
        f.write("image,expected,predicted,correct\n")
        for img_name, pred, true in pairs:
            f.write(f"{img_name},{true},{pred},{pred==true}\n")
        for img_name, true in skipped_imgs:
            f.write(f"{img_name},{true},NO_HAND,False\n")
    print(f"[info] Per-image predictions saved to: {csv_path}")

    # Save plots
    save_per_class_bar(per_class, results_dir / f"custom_{out_prefix}_per_class.png")
    save_confusion_matrix(pairs_for_stats, results_dir / f"custom_{out_prefix}_confusion.png")
    print(f"[info] Plots saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DH-GAT on a custom dataset.")
    parser.add_argument("--data-dir",   type=str, required=True, help="Path to dataset with A-Z folders.")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "dh_gat_static_best.pt"),
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print expected vs predicted for every image, grouped by class.")
    args = parser.parse_args()

    evaluate_custom_dataset(args.data_dir, args.checkpoint, verbose=args.verbose)
