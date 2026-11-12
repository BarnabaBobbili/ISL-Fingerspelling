"""
src/static/predict.py
======================
Predict the ISL fingerspelling letter from a single image using the
trained DH-GAT model.

Usage:
    python src/static/predict.py --image "path/to/hand.jpg"
    python src/static/predict.py --image "path/to/hand.jpg" --top 5
    python src/static/predict.py --image "path/to/hand.jpg" --checkpoint "checkpoints/custom.pt"

Output example:
    Image : hand.jpg
    Prediction : A  (confidence: 94.23%)

    Top-5:
      1.  A   94.23%  ######################
      2.  S    3.10%  #
      3.  E    1.45%
      4.  N    0.72%
      5.  T    0.50%
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import (
    StaticConfig, IDX_TO_LETTER, NUM_LANDMARKS, LANDMARK_DIM,
    MP_MAX_NUM_HANDS_STATIC, ROOT
)
from src.models.dh_gat import build_dhgat
from src.preprocessing.normalizer import normalize_hand_static

TASK_MODEL_PATH  = ROOT / "hand_landmarker.task"
DEFAULT_CKPT     = ROOT / "checkpoints" / "dh_gat_static_best.pt"


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe detector
# ─────────────────────────────────────────────────────────────────────────────
def _build_detector():
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    if not TASK_MODEL_PATH.exists():
        print(f"[ERROR] hand_landmarker.task not found at {TASK_MODEL_PATH}")
        sys.exit(1)

    base = mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=MP_MAX_NUM_HANDS_STATIC,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────────────────────────────────────
# Extract landmarks from image
# ─────────────────────────────────────────────────────────────────────────────
def extract_landmarks(image_path: Path, detector):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    r_lm = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    l_lm = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    rp, lp = False, False

    if result.hand_landmarks:
        for i, handedness_list in enumerate(result.handedness):
            label = handedness_list[0].display_name
            arr = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[i]], dtype=np.float32)
            arr = normalize_hand_static(arr)
            if label == "Right":
                r_lm = arr; rp = True
            else:
                l_lm = arr; lp = True

    return r_lm, l_lm, rp, lp


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict(image_path: str, checkpoint_path: str, top_k: int = 5):
    img_path  = Path(image_path)
    ckpt_path = Path(checkpoint_path)

    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = StaticConfig()
    model  = build_dhgat(cfg).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Extract landmarks
    detector = _build_detector()
    r_lm, l_lm, rp, lp = extract_landmarks(img_path, detector)
    detector.close()

    if not rp and not lp:
        print(f"\n  Image  : {img_path.name}")
        print(f"  [WARN] No hand detected in image. Cannot predict.")
        return

    # Run inference
    with torch.no_grad():
        rr = torch.from_numpy(r_lm).unsqueeze(0).to(device)
        ll = torch.from_numpy(l_lm).unsqueeze(0).to(device)
        rp_t = torch.tensor([rp], dtype=torch.bool).to(device)
        lp_t = torch.tensor([lp], dtype=torch.bool).to(device)

        logits = model(rr, ll, rp_t, lp_t)
        probs  = torch.softmax(logits, dim=1)[0]

    # Sort predictions
    top_vals, top_idxs = probs.topk(min(top_k, 26))

    best_letter = IDX_TO_LETTER[top_idxs[0].item()]
    best_conf   = top_vals[0].item()

    # Print results
    print(f"\n  Image      : {img_path.name}")
    print(f"  Hand(s)    : {'Right ' if rp else ''}{'Left' if lp else ''}")
    print(f"  Prediction : {best_letter}   (confidence: {100*best_conf:.2f}%)")
    print(f"\n  Top-{top_k}:")
    for rank, (idx, prob) in enumerate(zip(top_idxs, top_vals), 1):
        letter  = IDX_TO_LETTER[idx.item()]
        pct     = prob.item() * 100
        bar     = "#" * int(pct / 2)
        print(f"    {rank}.  {letter}   {pct:6.2f}%  {bar}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict ISL letter from a single image.")
    parser.add_argument("--image",      type=str, required=True,            help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT), help="Path to model checkpoint.")
    parser.add_argument("--top",        type=int, default=5,                help="Number of top predictions to show.")
    args = parser.parse_args()

    predict(args.image, args.checkpoint, args.top)
