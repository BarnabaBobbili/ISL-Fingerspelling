---
sidebar_position: 2
title: Training, decoding and key frames
---

# Training, decoding and key frames

## Training

```bash
python src/training/build_cont_features.py            # features for all 1,308 videos (2.5 min)
python src/training/train_cont_v2.py --kind conformer --parts all --tag conformer_all --bs 32 --epochs 60
```

| Setting | Value |
|---|---|
| Data | 1,308 videos split by *video*: train / validation / test as in the dataset (see [Datasets](../data/datasets)) |
| Normalisation | per-dimension mean/std of the training frames, stored in the checkpoint |
| Loss | CTC (blank = 26, `zero_infinity`) + 0.3 × intermediate CTC (Conformer) |
| Optimiser | AdamW, weight decay 10⁻², one-cycle schedule, lr 7·10⁻⁴ (Conformer) / 1.5·10⁻³ (BiLSTM) |
| Batch / epochs | 32 sequences, up to 60 epochs, early stopping on validation CER (patience 12) |
| Model selection | lowest **validation** CER; the test set is touched once at the end |

### Augmentation (on the feature sequence)

| Augmentation | Details | Simulates |
|---|---|---|
| time-stretch | factor 0.8 – 1.25 (linear interpolation), p = 0.7 | faster / slower signers |
| time masks | 0 – 2 masks of 2 – 8 frames set to zero | detector dropouts, occlusion |
| feature dropout | each value dropped with p = 0.05 | noise robustness |
| trunk dropout | the 218 trunk + posterior values zeroed, p = 0.25 | forces use of the raw-geometry path (see the stacking note) |
| Gaussian noise | σ = 0.08 (standardised units) | tracker jitter |

## Decoding

*Greedy CTC decoding*: take the arg-max class of every frame, merge repeated labels, drop blanks. `src/pipeline/ctc.py: greedy_with_pos` additionally returns, for each emitted letter, the **frame of maximum probability inside its run** — the letter's key frame.

## One key frame per letter

For an uploaded video:

1. every frame → hands → features → log-probabilities (T × 27);
2. greedy decoding gives the letters **and** their peak frames;
3. each peak frame is saved as a thumbnail (`keyframes/NN_X.jpg`) with timestamp and confidence (the posterior at the peak);
4. the annotated video types each letter at its peak frame.

(`ctc.forced_align` — Viterbi alignment of a *known* transcript — is used offline to build the [aligned real frames](../data/aligned-real-frames).)

## Live webcam

`ContinuousSession` keeps a buffer of the features of the most recent frames (up to 12 s). Every third frame the sequence model re-decodes the whole buffer and the text on screen is `committed words + current hypothesis`. When the hands leave the picture for **1.2 s** the word is committed (with a trailing space) and the buffer is cleared. The current letter and its confidence come from the last decoded letter.

Typical cost on a laptop CPU: MediaPipe 20 – 40 ms + trunk 5 ms + sequence model (≤ 360 frames, every third frame) ≈ 10 – 30 ms.

:::note No space symbol
The dataset transcripts contain no spaces, so the model cannot emit them. Word breaks in live mode come from the pause rule above.
:::

## Example

Held-out test videos (names spelled by an interpreter):

| Truth | Prediction |
|---|---|
| `mggaikwad` | `moggaikwad` |
| `devendrafadnavis` | `devendrafadnavis` |
| `ksivan` | `ksivan` |
| `sushilchandra` | `sushilchandra` |
| `jiteshgupta` | `jiteshgupta` |
| `dikshasingh` | `dikshasingh` |
| `deepakdadich` | `deepakdadich` |
| `ashokgehlot` | `ashokgehlot` |
