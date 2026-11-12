---
sidebar_position: 2
title: Limitations
---

# Limitations

## 1 · Generalisation to a new camera

In the cross-dataset test the model has never seen Static Dataset B and reaches **70 %** (64 % before second-hand recovery), compared with 95 % and more on held-out real-video frames. The causes are the camera viewpoint (Static Dataset B is shot close-up from below with overlapping hands) and missed second hands. A drop in accuracy is to be expected with a new webcam. Fine-tuning on a short recording from the target camera is expected to be the most effective remedy; the repository includes `src/static/collect_webcam_data.py` and a fine-tuning option in the baseline trainer for this purpose.

## 2 · Signer independence

The real-video validation and test sets consist of different *videos*, but the documentation of the continuous dataset does not state the number or identity of the signers, and the same signer may appear in several videos (see [Datasets](../data/datasets)). The scores therefore measure generalisation to new clips and not necessarily to new people.

## 3 · Confidence on unfamiliar footage

Calibration is fitted on real-video validation logits. On an unseen dataset the model is over-confident (ECE 0.08 – 0.18). A confidence of 90 % on a very different camera should be read as "probably correct" and not as a measured 90 %.

## 4 · Overlapping hands

MediaPipe often misses one of two overlapping hands. [Second-hand recovery](../data/landmark-extraction#second-hand-recovery) improves this (J: 0 → 52 %), but Q and R remain difficult (≈ 6 – 20 % recovered). When only one hand is found, accuracy for two-handed letters is lower (88 % against 97 % on the real test frames).

## 5 · Repeated letters in single-letter mode

Two identical letters in a row require a pause, a slight lowering of the hands, or a hold of at least 1.6 s. In **flowing** mode CTC handles repeats natively, because labels are separated by blanks.

## 6 · No spaces in continuous mode

Spaces are removed from the transcripts, so the continuous models cannot output them. The application uses a **pause** (hands out of view for about 1 s) to end a word.

## 7 · Vocabulary

A–Z only. The datasets also contain digits 1–9 (static) and a few digits and symbols in the transcripts (continuous); these are deliberately out of scope ([design decision DD-001](../reference/design-decisions)).

## 8 · 3D hands

- Poses are medoids of real samples; they are plausible but not a linguistic reference. Three letters (J, Q, S) have no reliable contact points, and Q is built from only 26 candidates.
- The non-signing hand of one-handed letters waits out of frame; the position of the body and arms in ISL grammar is not modelled.
- Motion between letters is interpolated, whereas real signers use co-articulated paths.

## 9 · Browser tracking speed

In-browser hand tracking uses WebAssembly and, where available, the GPU, and its speed depends on the machine. The delay and accuracy figures in this manual were measured by replaying recorded landmarks, and the server's share of the delay (≈ 5 ms per frame) was measured directly; the live tracking throughput on a typical laptop has not been benchmarked. If tracking is too slow, *Single letters* mode or the server-side fallback can be used.

## 10 · Hardware

CPU only. The static classifier runs in ≈ 5 ms per frame; MediaPipe dominates (≈ 20 – 40 ms). A 1280×720 webcam image is down-scaled to at most 960 px wide before it is sent to the server in the fallback path.
