---
slug: /
sidebar_position: 1
title: Overview
---

# ISL fingerspelling

**ISL fingerspelling** is a system that recognises and generates **Indian Sign Language (ISL) fingerspelling**, the A–Z manual alphabet used to spell names and words. It works in both directions:

| Direction | What it does | Where |
|---|---|---|
| **Sign → text / speech** | Reads letters from a live webcam or an uploaded video, shows the **letter and its confidence score on top of the video**, builds the text and can speak it | *Recognise* page (live camera or a video file) |
| **Text / speech → sign** | Turns typed or dictated text into 3D hands that sign it letter by letter, with the current character shown on top | *Sign* page |
| **Every character** | A reference of all 26 letters rendered on the 3D hands (one-handed or two-handed, contact points) | *Alphabet* page |

![The 3D hands spelling the letter H](/img/av_H.png)

## What is inside

- **Static recognition — DH-GAT.** A *Dual-Hand Graph Attention* network that represents both hands in one coordinate frame and models which finger touches which part of the other hand. Confidence scores are calibrated.
- **Continuous recognition — Conformer-CTC / BiLSTM-CTC.** Reads flowing fingerspelling (names, words) from video without requiring pauses, and extracts **one key frame per letter**.
- **Video and webcam pipeline.** MediaPipe hand tracking — in the browser for the live camera — → classifier → text, drawn as a HUD on top of the video. The live decoder commits letters about 0.4 s after they are signed and keeps up with fast signing and slow cameras ([Live streaming](continuous-recognition/live-streaming)).
- **3D hands.** Two skinned hands with sleeves (not a skeleton), animated between letter poses, using **geometric transformations — rigid alignment (Kabsch) and finger inverse kinematics (FABRIK) — to make the joining points touch**.
- **Web application.** A single-page site (light and dark themes) with this manual built in, served by a small Python back end. All processing runs on the local computer.

## Results at a glance

import ResultsGlance from './_results_glance.md';

<ResultsGlance />

:::info Evaluation protocol
Static recognition is reported under **three** tests instead of one, because a random split of the training images considerably overestimates real-world accuracy. See [Benchmark](evaluation/benchmark) and [Limitations](evaluation/limitations).
:::

## Contents

1. **[Getting started](getting-started/installation)** – installation and first run.
2. **[Architecture](architecture/overview)** – how the pieces fit together.
3. **[Data](data/datasets)** – the datasets, landmark extraction and the aligned real-frame benchmark.
4. **[Static](static-recognition/dh-gat)** and **[continuous](continuous-recognition/features-and-models)** recognition – models, training and decoding.
5. **[3D hands](avatar/pose-library)** – pose library, contact geometry and renderer.
6. **[Web app](web-app/user-manual)** – user manual and API.
7. **[Evaluation](evaluation/benchmark)** and **[Reference](reference/commands)** – numbers, commands, design decisions, troubleshooting.
