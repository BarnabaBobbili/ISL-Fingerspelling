---
slug: /
sidebar_position: 1
title: Overview
---

# Varna — ISL fingerspelling

**Varna** (Sanskrit for *letter*) recognises and synthesises **Indian Sign Language (ISL) fingerspelling** — the A–Z hand alphabet used to spell names and words. It works in both directions:

| Direction | What it does | Where |
|---|---|---|
| **Sign → text / speech** | Reads letters from a live webcam or an uploaded video, shows the **letter and its confidence score on top of the video**, builds the text and can speak it | *Recognise* page (live camera or a video file) |
| **Text / speech → sign** | Turns typed or dictated text into a 3D avatar that signs it letter by letter, with the current character shown on top | *Sign* page |
| **Every character** | A reference of all 26 letters rendered on the avatar (one-handed vs two-handed, contact points) | *Alphabet* page |

![The signer spelling the letter H](/img/av_H.png)

## What is inside

- **Static recognition — DH-GAT v2.** A *Dual-Hand Graph Attention* network that sees both hands in one coordinate frame and models which finger touches which part of the other hand. Calibrated confidence scores.
- **Continuous recognition — Conformer-CTC / BiLSTM-CTC.** Reads flowing fingerspelling (names, words) from video, no pauses needed, and extracts **one key frame per letter**.
- **Video and webcam pipeline.** MediaPipe hand tracking — in the browser for the live camera — → classifier → text, drawn as a HUD on top of the video. The live decoder commits letters about 0.4 s after they are signed and keeps up with fast signing and slow cameras ([Live streaming](continuous-recognition/live-streaming)).
- **3D hands.** Two skinned hands with sleeves (not a skeleton), animated between letter poses, using **geometric transformations — rigid alignment (Kabsch) and finger inverse kinematics (FABRIK) — to make the joining points touch**.
- **Web app.** A designed single-page site (light and dark themes) with this manual built in, served by a small Python backend. Everything runs on your computer.

## Results at a glance

import ResultsGlance from './_results_glance.md';

<ResultsGlance />

:::info Honest numbers
The project reports **three** static-recognition tests instead of one, because a random split of this dataset hugely overestimates real-world accuracy. See [Benchmark](evaluation/benchmark) and [Limitations](evaluation/limitations).
:::

## How to read this manual

1. **[Getting started](getting-started/installation)** – install and run the app in five minutes.
2. **[Architecture](architecture/overview)** – how the pieces fit together.
3. **[Data](data/datasets)** – the datasets, landmark extraction and the aligned real-frame benchmark.
4. **[Static](static-recognition/dh-gat-v2)** and **[continuous](continuous-recognition/features-and-models)** recognition – models, training and decoding.
5. **[Avatar](avatar/pose-library)** – pose library, contact geometry and renderer.
6. **[Web app](web-app/user-manual)** – user manual and API.
7. **[Evaluation](evaluation/benchmark)** and **[Reference](reference/commands)** – numbers, commands, design decisions, troubleshooting.
