---
sidebar_position: 2
title: Training and calibration
---

# Training and calibration

```bash
python src/static/train_v2.py --tag main                        # final model
python src/static/train_v2.py --tag xsrc --holdout isl_data     # cross-source experiment
```

## Data

| Pool | Content | Role |
|---|---|---|
| `train_static` | Static_ISL + ISL_DATA + Kaggle, original train split | bulk of the data |
| `train_real` | aligned news-video letters from **training videos** | target domain, sampled 4× more often |
| `val_real` | aligned letters from validation videos | **model selection** and calibration |
| `test_real` | aligned letters from test videos | final test |

Model selection uses **validation real-video accuracy**, never the near-duplicate static validation set (that one saturates at 98 % and would select an over-fitted model).

## Sampling

Each step draws 256 samples with replacement, weighted by *domain weight / √(class frequency in that domain)*, so rare letters are not drowned and the real-video domain makes up ≈ 69 % of the sampling mass.

## Augmentation (batched, on raw coordinates)

| Augmentation | Range | Simulates |
|---|---|---|
| per-hand scale | ± 10 % | hands at different distances |
| per-hand z-rotation (about its wrist) | ± 10° | wrist tilt differences |
| per-hand translation | σ = 0.12 hand sizes | the hands drifting relative to each other |
| global 3D rotation of the whole scene | x ± 20°, y ± 20°, z ± 30° | camera viewpoint |
| landmark noise | σ = 0.02 hand size (tips 0.03, palm 0.01) | tracker jitter |
| horizontal mirror + slot swap | p = 0.3 | left-handed signer / mirrored preview |
| drop one hand | p = 0.04 | missed detection |

## Optimisation

AdamW (lr 2·10⁻³, weight decay 10⁻²), one-cycle schedule, label smoothing 0.05, gradient clipping 1.0, **EMA of the weights** (decay 0.998; the EMA model is what is evaluated and saved), 30 epochs × 32,000 samples, early stopping (patience 8) on validation real-video accuracy. About 20 minutes on a laptop CPU.

## Calibration

Raw soft-max scores are rarely trustworthy. After training, a single **temperature** `T` is fitted on the validation real-video logits by minimising negative log-likelihood (L-BFGS); the deployed confidence is `softmax(logits / T)` with `T ≈ 0.84` stored in the checkpoint. The **expected calibration error (ECE)** on real test videos fell from **0.17** (v1) to **0.011** (final model): a displayed 96 % really means about 96 %.

:::caution Calibration does not transfer to new setups
On a completely unseen source (ISL_DATA, never trained on) the model is *over-confident*: ECE 0.08 uncalibrated, 0.13 after the real-video temperature (0.14 → 0.18 without second-hand recovery). Confidence is only as good as the match between the live camera and the training domain — see [Limitations](../evaluation/limitations).
:::

## Second-hand recovery (adopted)

Every image was re-extracted with [second-hand recovery](../data/landmark-extraction#second-hand-recovery) and the model retrained (`--tag main_r`, `--tag xsrc_r`). Compared with the same training on plain landmarks:

| | plain landmarks | with recovery |
|---|---:|---:|
| real news-video test accuracy | 95.7 % | **95.9 %** |
| expected calibration error (real, calibrated) | 0.015 | **0.011** |
| **cross-source** accuracy on ISL_DATA | 64.0 % | **70.0 %** |

The cross-source gain (+6 points) decided it: **the app's static classifier is the recovery model** (`checkpoints/dh_gat_v2_main_r.pt`) and the static tracker runs with recovery on (`HandTracker(recover=True)`). Checkpoint names ending in `_r` select this automatically. The continuous model keeps the plain tracker, because its features were computed without recovery.

One side effect is worth knowing: recovery sometimes adds a *spurious* second hand to one-handed letters (about 18 % for O). The classifier tolerates it, and the hold detector ignores the brief prediction flicker it can cause (see [Hold detection](hold-detection)).
