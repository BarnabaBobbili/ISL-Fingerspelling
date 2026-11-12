---
sidebar_position: 1
title: Datasets
---

# Datasets

Five data sources are used. None of them is redistributed with the repository.

## Static letters (images)

| Source | Images | Letters | Characteristics |
|---|---:|---|---|
| **Static Dataset A** | 31,200 (1,200 per letter) | A–Z | close-up of two hands on a dark background; many near-duplicate video frames |
| **Static Dataset B** | 13,061 (≈ 460 – 700 per letter; digits 1–9 are present in the collection but not used) | A–Z | plain light-grey wall, hands viewed from below, different camera |
| **Static Dataset C** | 702 (≈ 25 per letter, 23 letters) | most letters | varied backgrounds, low image quality |
| **Total** | **44,963** | | |

![Letters across the static datasets](/img/sources_montage.jpg)

### Which letters use one hand?

From the images and the detection statistics, **six letters are one-handed (C, I, L, O, U, V) and the other 20 are two-handed**. MediaPipe's own hand count is not a reliable indicator: for letters whose hands overlap (J, Q, R and others) it reports a single hand most of the time. The pose library and the application use this fixed assignment.

## Continuous fingerspelling (video)

The *Continuous Fingerspelling Dataset for Indian Sign Language* (Kirandevraj, Kurmi, Namboodiri and Jawahar; WSLP @ AACL-IJCNLP 2025) is the first continuous fingerspelling dataset for ISL. It is released under the CC BY-NC 4.0 licence; project page: [kirandevraj.github.io/ISL-Fingerspelling](https://kirandevraj.github.io/ISL-Fingerspelling).

| | |
|---|---|
| Video segments | 1,308, cut from 499 YouTube videos |
| Duration | 70.85 min |
| Characters | 14,814 (13,803 letters after removing spaces) |
| Annotation | one transcript per segment (`fingerspelling_annotations.csv`), with the temporal location of each segment in its source video |

Only the 26 lower-case letters are used as the CTC vocabulary (digits and symbols occur a handful of times and are annotation artefacts). **Spaces are removed from the transcripts**, so the continuous model outputs words without word boundaries.

### Splits

- The split provided with the dataset (1,104 training and 204 test segments) is used; it is made **by source video**, so all segments of one YouTube video stay together.
- 10 % of the training *videos* are held out as validation (seed 42), leaving 998 / 106 / 204 segments.
- The same video-level split is reused for the aligned real frames (see [Aligned real frames](aligned-real-frames)), so no frame of a test video is used in training.

:::caution Signer independence
The dataset documentation does not state how many signers appear, and the same signer may appear in several videos. The split is therefore video-independent but not necessarily signer-independent, and scores on the real-video test measure generalisation to new *clips* rather than to new *people*. The cross-dataset test (see [Benchmark](../evaluation/benchmark)) is the more reliable indicator for a new setting.
:::
