---
sidebar_position: 2
title: Limitations
---

# Limitations and honest caveats

## 1 · Generalisation to a new camera / signer

The cross-source test — the model never saw ISL_DATA — gives only **70 %** (64 % before second-hand recovery), against 95 %+ on the real news videos. The causes are camera viewpoint (ISL_DATA is shot close-up from below with overlapping hands) and missed second hands. **Expect a drop on your own webcam**; recording a few minutes of your own signs and fine-tuning is the most effective remedy (the repository includes `src/static/collect_webcam_data.py` and the `--fine-tune` hook of the v1 trainer for exactly this).

## 2 · Same interpreters across videos

The real-video validation and test sets are different *videos* but the same few news interpreters appear in many of them (see [Datasets](../data/datasets)). Scores on them measure generalisation to new clips, not to new people.

## 3 · Confidence on unfamiliar footage

Calibration is fitted on real-video validation logits. On an unseen source the model is over-confident (ECE 0.08 – 0.18). A confidence of 90 % on a very different camera should be read as "probably right", not as a measured 90 %.

## 4 · Overlapping hands

MediaPipe often misses one of two overlapping hands. [Second-hand recovery](../data/landmark-extraction#second-hand-recovery) helps (J: 0 → 52 %) but Q and R remain hard (≈ 6 – 20 % recovered). When only one hand is found, accuracy for two-handed letters drops (88 % vs 97 % on real test frames).

## 5 · Repeated letters in static mode

Two identical letters in a row need a pause, a slight lowering of the hands, or a hold of ≥ 1.6 s. In **continuous** mode CTC handles repeats natively (it is blank-separated).

## 6 · No spaces in continuous mode

The dataset transcripts contain no spaces, so the continuous models cannot output them. The app uses a **pause** (hands out of view ≈ 1.2 s) to end a word.

## 7 · Vocabulary

A–Z only. The datasets also contain digits 1–9 (static) and a few digits/symbols in captions (continuous); they are deliberately out of scope ([design decision DD-001](../reference/design-decisions)).

## 8 · Avatar

- Poses are medoids of real samples; they are plausible, not a linguistic reference. Three letters (J, Q, S) have no reliable contact points; Q is built from only 26 candidates.
- The non-signing hand of one-handed letters simply rests; ISL grammar for body/arm position is not modelled.
- Motion between letters is interpolated; real signers use co-articulated paths.

## 9 · Hardware

CPU only. The static classifier runs in ≈ 5 ms per frame; MediaPipe dominates (≈ 20 – 40 ms). A 1280×720 webcam is down-scaled to at most 960 px wide before sending to the server.
