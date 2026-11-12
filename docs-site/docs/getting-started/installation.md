---
sidebar_position: 1
title: Installation
---

# Installation

## Requirements

| Item | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 | tested on 3.13 |
| OS | Windows / Linux / macOS | developed on Windows 11 |
| RAM | 8 GB or more | training needs ≈ 4 GB free |
| Camera | any webcam | only for the live tab |
| Browser | Chrome or Edge | speech dictation (microphone) needs Chromium; everything else works in any modern browser |
| GPU | **not required** | all models are small and run on CPU |

## Install

```bash
git clone https://github.com/BarnabaBobbili/ISL-Fingerspelling.git
cd ISL-Fingerspelling
pip install -r requirements.txt
```

`requirements.txt` lists PyTorch, MediaPipe, OpenCV, NumPy/Pandas/SciPy/scikit-learn, Starlette + uvicorn (web server) and a few utilities.

:::tip Why Starlette and not FastAPI?
The server uses [Starlette](https://www.starlette.io/) directly (FastAPI is built on top of it). This avoids version conflicts between FastAPI and Starlette on machines that already have both installed.
:::

## Model files

The repository ships the trained weights needed to run the app:

| File | Size | Purpose |
|---|---|---|
| `hand_landmarker.task` | 7.8 MB | MediaPipe hand landmark model |
| `checkpoints/dh_gat_v2_main.pt` | ≈ 2 MB | static letter classifier (DH-GAT) |
| `checkpoints/cont_best.pt` | few MB | continuous (CTC) model |

The **datasets are not in the repository** because of their size. They are required only for retraining; see [Commands](../reference/commands).

## Verify the installation

```bash
python -c "import torch, mediapipe, cv2, starlette; print('ok')"
python src/models/dh_gat_v2.py          # prints parameter count, finite outputs
```
