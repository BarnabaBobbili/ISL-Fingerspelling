---
sidebar_position: 1
title: Datasets
---

# Datasets

Four collections are used. None of them is redistributed with the repository.

## Static letters (images)

| Source | Images | Letters | Look |
|---|---:|---|---|
| **Static_ISL** | 31,200 (exactly 1,200 per letter) | A–Z | close-up of two hands on a dark background; many near-duplicate video frames |
| **ISL_DATA** | 13,061 (≈ 460 – 700 per letter, plus digits 1–9 that are *not* used) | A–Z | plain light-grey wall, hands viewed from below, a different signer and camera |
| **Kaggle ISL** | 702 (≈ 25 per letter, 23 letters) | most letters | varied backgrounds, low quality |
| **Total** | **44,963** | | |

![Letters across sources](/img/sources_montage.jpg)

### Which letters use one hand?

By inspecting the images (and the detection statistics), **six letters are one-handed — C, I, L, O, U, V — and the other 20 are two-handed**. MediaPipe's own hand count is *not* a reliable indicator: for letters whose hands overlap (J, Q, R…) it reports a single hand most of the time. The pose library and the app use this fixed split.

## Continuous fingerspelling (video)

The *Continuous Fingerspelling Dataset for Indian Sign Language* (WSLP @ AACL-IJCNLP 2025):

| | |
|---|---|
| Video segments | 1,308 |
| Duration | 70.85 min |
| Characters | 14,814 |
| Content | news interpreters spelling names (e.g. `vijaymallya`, `sulochana das`) |
| Annotation | one transcript per segment (`fingerspelling_annotations.csv`) |

Only the 26 lower-case letters are used as the CTC vocabulary (digits/symbols appear a handful of times and are annotation artefacts). **Spaces are not part of the vocabulary**, so the continuous model outputs words without spaces.

### Splits

- Provided train/test split, **by source video** (all segments of one YouTube video stay together).
- 10 % of the training *videos* are held out as validation (seed 42).
- The same video-level split is reused for the aligned real frames (see [Aligned real frames](aligned-real-frames)) so no frame of a test video is ever seen in training.

:::caution Signer independence
The split is video-independent, **not signer-independent**: the same few news interpreters appear in many videos. Scores on the real-video test therefore measure generalisation to new *clips*, not to new *people*. The cross-source test (below) is the cleaner check for new people.
:::
