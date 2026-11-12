---
sidebar_position: 1
title: Benchmark
---

# Benchmark

All numbers are produced by scripts in the repository and written into this page by
`src/evaluation/make_benchmark_doc.py` from the files in `results/`. Nothing is typed by hand.

## Static letters

Three tests are always reported together (see [design decision DD-008](../reference/design-decisions)):

| Test | What it measures |
|---|---|
| **In-domain** | the original random split of ≈ 45 k images (near-duplicate frames: optimistic) |
| **Real news video** | 1,069 letter frames CTC-aligned from **held-out test videos** (small hands, full-body shot) |
| **Cross-source** | train *without* ISL_DATA, test on all 13,061 of its images (new camera, background, signer) |

### In-domain and real-video

| Model | In-domain (orig. split) | **Real news-video test** | ECE uncalibrated | ECE calibrated |
|---|---:|---:|---:|---:|
| DH-GAT v1 (original) | 98.23 % | **38.35 %** | 0.170 | — |
| DH-GAT v2 | 97.99 % | **95.70 %** | 0.028 | 0.015 |
| DH-GAT v2 + second-hand recovery | 98.04 % | **95.88 %** | 0.023 | 0.011 |

*ECE = expected calibration error (10 bins); lower is better. "Calibrated" uses the temperature fitted on the real-video validation logits.*

### Cross-source (the realistic test for a new camera)

| Model (trained **without** ISL_DATA) | ISL_DATA accuracy | …with both hands detected | …with one hand detected | ECE (calibrated) |
|---|---:|---:|---:|---:|
| DH-GAT v2 | **64.0 %** | 76.6 % | 53.4 % | 0.181 |
| DH-GAT v2 + second-hand recovery | **70.0 %** | 72.9 % | 65.8 % | 0.126 |

Accuracy is far lower than on the real news videos because ISL_DATA is shot close-up from below with overlapping hands, which also makes MediaPipe miss the second hand. The columns show how strongly accuracy depends on both hands being detected.

## Continuous fingerspelling

Character error rate (CER = edit distance / reference length, corpus level, greedy CTC decoding) on the validation and test videos of the dataset:

| Model | Parameters | Validation CER | **Test CER** |
|---|---:|---:|---:|
| BiLSTM-CTC v1 (63-d landmarks, one hand) | — | 14.24 % | **14.75 %** |
| Conformer-CTC v1 (63-d landmarks, one hand) | — | 18.72 % | **19.26 %** |
| BiLSTM-CTC v2 — features: all (351-d) | 1,469,049 | 6.30 % | **4.28 %** |
| BiLSTM-CTC v2 — ablation: raw geometry + motion only (133-d) | 1,437,221 | 6.85 % | **5.12 %** |
| Conformer-CTC v2 — features: all (351-d) | 2,569,140 | 5.75 % | **3.91 %** |
| Conformer-CTC v2 — ablation: raw geometry + motion only (133-d) | 2,537,312 | 7.67 % | **5.91 %** |

- The **v1** Conformer was *worse* than the v1 BiLSTM. The v2 features (pretrained hand-shape embedding + both hands + motion), augmentation, a smaller convolution kernel and the intermediate CTC loss turn this around.
- The **ablation** rows use no trunk/posterior features at all, so the difference to the "all features" rows is what the pretrained DH-GAT contributes: **Conformer test CER 5.91 % → 3.91 %; BiLSTM test CER 5.12 % → 4.28 %**.
- Without the pretrained features the Conformer is *not* better than the BiLSTM — consistent with Conformers needing more data than the ≈ 70 minutes available. The pretrained hand-shape embedding supplies the missing prior, which is why the Conformer becomes the best model only with it.
- Validation CER selects the model; the test set is used once.

## Avatar contact alignment

Mean absolute contact gap in hand sizes (1 hand size ≈ 9.5 cm; 0 = skin on skin).

| Letter | Hands | Candidates | Contacts | Mean gap, raw | Mean gap, aligned |
|---|---:|---:|---:|---:|---:|
| A | 2 | 500 | 1 | 0.7021 | 0.0013 |
| B | 2 | 500 | 4 | 0.0561 | 0.0136 |
| C | 1 | 500 | 0 | — | — |
| D | 2 | 500 | 4 | 0.115 | 0.0207 |
| E | 2 | 500 | 3 | 0.081 | 0.0047 |
| F | 2 | 500 | 4 | 0.1245 | 0.0194 |
| G | 2 | 500 | 3 | 0.3154 | 0.0151 |
| H | 2 | 500 | 4 | 0.0643 | 0.0168 |
| I | 1 | 500 | 0 | — | — |
| J | 2 | 202 | 0 | — | — |
| K | 2 | 500 | 3 | 0.2303 | 0.0172 |
| L | 1 | 500 | 0 | — | — |
| M | 2 | 500 | 1 | 0.0937 | 0.005 |
| N | 2 | 500 | 2 | 0.2585 | 0.0074 |
| O | 1 | 500 | 0 | — | — |
| P | 2 | 500 | 3 | 0.2342 | 0.0435 |
| Q | 2 | 26 | 0 | — | — |
| R | 2 | 500 | 1 | 0.2629 | 0.0014 |
| S | 2 | 500 | 0 | — | — |
| T | 2 | 500 | 2 | 0.1945 | 0.0184 |
| U | 1 | 500 | 0 | — | — |
| V | 1 | 500 | 0 | — | — |
| W | 2 | 500 | 4 | 0.2416 | 0.0186 |
| X | 2 | 500 | 3 | 0.205 | 0.0416 |
| Y | 2 | 500 | 2 | 0.087 | 0.0242 |
| Z | 2 | 500 | 4 | 0.1047 | 0.0069 |

Over the 17 letters that have contacts, the mean gap falls from **0.198** to **0.016** hand sizes (≈ 1.9 cm → 1.5 mm).

## Reproducing

```bash
python src/static/benchmark.py                 # legacy v1 on all protocols
python src/static/train_v2.py --tag main       # prints the v2 table
python src/training/select_best_cont.py        # continuous comparison
python src/evaluation/make_benchmark_doc.py    # regenerate this page
```
