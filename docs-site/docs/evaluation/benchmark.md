---
sidebar_position: 1
title: Benchmark
---

# Benchmark

All numbers are produced by scripts in the repository and written into this page by
`src/evaluation/make_benchmark_doc.py` from the files in `results/`. Nothing is typed by hand.

## Static letters

Three tests are reported together (see [design decision DD-008](../reference/design-decisions)):

| Test | What it measures |
|---|---|
| **In-domain** | the original random split of ≈ 45 k images (near-duplicate frames: optimistic) |
| **Real-video frames** | 1,069 letter frames CTC-aligned from **held-out test videos** of the Continuous Fingerspelling Dataset |
| **Cross-dataset** | trained *without* Static Dataset B and tested on all 13,061 of its images (different camera and background) |

### In-domain and real-video

| Model | In-domain (orig. split) | **Real-video test** | ECE uncalibrated | ECE calibrated |
|---|---:|---:|---:|---:|
| Baseline (per-hand normalisation) | 98.23 % | **38.35 %** | 0.170 | — |
| Shared-frame DH-GAT | 97.99 % | **95.70 %** | 0.028 | 0.015 |
| Shared-frame DH-GAT + second-hand recovery (proposed) | 98.04 % | **95.88 %** | 0.023 | 0.011 |

*ECE = expected calibration error (10 bins); lower is better. "Calibrated" uses the temperature fitted on the real-video validation logits.*

### Cross-dataset (approximating a new camera)

| Model (trained **without** Static Dataset B) | Static Dataset B accuracy | …with both hands detected | …with one hand detected | ECE (calibrated) |
|---|---:|---:|---:|---:|
| Shared-frame DH-GAT | **64.0 %** | 76.6 % | 53.4 % | 0.181 |
| Shared-frame DH-GAT + second-hand recovery (proposed) | **70.0 %** | 72.9 % | 65.8 % | 0.126 |

Accuracy is lower than on the real-video frames because Static Dataset B is shot close-up from below with overlapping hands, which also causes MediaPipe to miss the second hand. The columns show how strongly accuracy depends on both hands being detected.

## Continuous fingerspelling

Character error rate (CER = edit distance / reference length, corpus level, greedy CTC decoding) on the validation and test videos of the dataset:

| Model | Parameters | Validation CER | **Test CER** |
|---|---:|---:|---:|
| BiLSTM-CTC, baseline (63-d landmarks, one hand) | — | 14.24 % | **14.75 %** |
| Conformer-CTC, baseline (63-d landmarks, one hand) | — | 18.72 % | **19.26 %** |
| BiLSTM-CTC — features: all (351-d) | 1,469,049 | 6.30 % | **4.28 %** |
| BiLSTM-CTC — ablation: raw geometry + motion only (133-d) | 1,437,221 | 6.85 % | **5.12 %** |
| Conformer-CTC — features: all (351-d) | 2,569,140 | 5.75 % | **3.91 %** |
| Conformer-CTC — ablation: raw geometry + motion only (133-d) | 2,537,312 | 7.67 % | **5.91 %** |
| Conformer-CTC — features: all (351-d) + streaming training (proposed) | 2,569,140 | 5.94 % | **3.63 %** |

- The baseline Conformer was *worse* than the baseline BiLSTM. The proposed input features (pretrained hand-shape embedding, both hands and motion), the augmentation, a smaller convolution kernel and the intermediate CTC loss reverse this ordering.
- The **ablation** rows use no trunk/posterior features at all, so the difference to the "all features" rows is what the pretrained DH-GAT contributes: **Conformer test CER 5.91 % → 3.91 %; BiLSTM test CER 5.12 % → 4.28 %**.
- Without the pretrained features the Conformer is *not* better than the BiLSTM — consistent with Conformers needing more data than the ≈ 70 minutes available. The pretrained hand-shape embedding supplies the missing prior, which is why the Conformer becomes the best model only with it.
- Validation CER selects the model; the test set is used once.

## Live decoding

The live recogniser (30 Hz resampling, sliding window, stable-prefix commit) replayed over the held-out test videos, at normal and faster signing speeds and on slower cameras. Character error rate; lower is better.

| Condition | Without streaming training (live) | Whole clip | **Live decoder** | Letter delay (median) |
|---|---:|---:|---:|---:|
| speed x1 | 4.56 % | 3.96 % | **3.77 %** | 0.40 s |
| speed x1.5 | 5.82 % | 4.23 % | **4.14 %** | 0.40 s |
| speed x2 | 11.63 % | 5.77 % | **5.77 %** | 0.40 s |
| camera 15 fps | 5.03 % | 5.30 % | **4.33 %** | 0.40 s |
| camera 10 fps | 6.38 % | 13.22 % | **5.68 %** | 0.43 s |

*Faster signing time-compresses the hand track; slower cameras drop frames and keep true timestamps. A letter is final about 0.4 s after it was signed. “Without streaming training” is the same model trained without the streaming augmentations.*

## Contact alignment of the 3D hands

Mean absolute contact gap in hand sizes (1 hand size ≈ 9.5 cm; 0 = skin on skin).

| Letter | Hands | Candidates | Contacts | Mean gap, raw | Mean gap, aligned |
|---|---:|---:|---:|---:|---:|
| A | 2 | 500 | 1 | 0.5741 | 0.004 |
| B | 2 | 500 | 4 | 0.0561 | 0.0136 |
| C | 1 | 500 | 0 | — | — |
| D | 2 | 500 | 4 | 0.1109 | 0.019 |
| E | 2 | 500 | 3 | 0.1363 | 0.0065 |
| F | 2 | 500 | 4 | 0.1245 | 0.0194 |
| G | 2 | 500 | 3 | 0.3326 | 0.0171 |
| H | 2 | 500 | 4 | 0.0809 | 0.0183 |
| I | 1 | 500 | 0 | — | — |
| J | 2 | 500 | 0 | — | — |
| K | 2 | 500 | 3 | 0.2465 | 0.02 |
| L | 1 | 500 | 0 | — | — |
| M | 2 | 500 | 1 | 0.0475 | 0.0024 |
| N | 2 | 500 | 2 | 0.8125 | 0.0328 |
| O | 1 | 500 | 0 | — | — |
| P | 2 | 500 | 3 | 0.2252 | 0.046 |
| Q | 2 | 63 | 0 | — | — |
| R | 2 | 500 | 2 | 0.123 | 0.0212 |
| S | 2 | 500 | 0 | — | — |
| T | 2 | 500 | 2 | 0.161 | 0.0173 |
| U | 1 | 500 | 0 | — | — |
| V | 1 | 500 | 0 | — | — |
| W | 2 | 500 | 4 | 0.2416 | 0.0186 |
| X | 2 | 500 | 3 | 0.224 | 0.0294 |
| Y | 2 | 500 | 2 | 0.0589 | 0.0043 |
| Z | 2 | 500 | 4 | 0.0878 | 0.0031 |

Over the 17 letters that have contacts, the mean gap falls from **0.214** to **0.017** hand sizes (≈ 2.0 cm → 1.6 mm).

## Reproducing

```bash
python src/static/benchmark.py                 # baseline model on all protocols
python src/static/train_v2.py --tag main       # trains the DH-GAT and prints its table
python src/training/select_best_cont.py        # continuous comparison
python src/evaluation/make_benchmark_doc.py    # regenerate this page
```
