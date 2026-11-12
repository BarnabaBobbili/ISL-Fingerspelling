---
sidebar_position: 2
title: Repository layout
---

# Repository layout

```text
ISL-Fingerspelling/
├── app/                         web application
│   ├── server.py                Starlette back end (WebSocket + upload API)
│   └── static/                  index.html, css/, js/ (app, live, hud, avatar), data/poses.json, vendor/ (Three.js)
├── src/
│   ├── models/                  dh_gat.py (v1), dh_gat_v2.py, bilstm_ctc.py, conformer_ctc.py, cont_v2.py, …
│   ├── static/                  extract_v2.py, train_v2.py, benchmark.py, webcam/data tools (v1 era)
│   ├── extraction/              landmark_extractor.py, align_real_frames.py, extract_continuous_v2.py
│   ├── pipeline/                hands, classifier, hold, features, continuous, video, overlay, ctc, cli
│   ├── avatar/                  touch.py (geometry), build_poses.py
│   ├── training/                config.py, train.py (v1), build_cont_features.py, train_cont_v2.py, select_best_cont.py
│   ├── preprocessing/           normalizer.py, feature_engineer.py (v1 features)
│   ├── evaluation/              metrics, decoder, confusion analysis, baselines
│   └── data/                    dataset_loader.py, split_manager.py (v1 continuous data)
├── tests/                       synthetic-video pipeline test, geometry tests
├── checkpoints/                 trained weights (the small ones are committed)
├── docs-site/                   this documentation (Docusaurus)
├── hand_landmarker.task         MediaPipe model
└── requirements.txt
```

Not in the repository (listed in `.gitignore`): `data/` (datasets), `cache/` (landmark caches), `logs/`, `results/`.

## Two generations of code

The project started with a first generation (v1) — per-hand-normalised DH-GAT and landmark-only BiLSTM/Conformer. The `v2` modules replace them for the app, but v1 is kept because the evaluation compares against it:

| | v1 | v2 |
|---|---|---|
| Static model | `models/dh_gat.py`, `static/train_static.py` | `models/dh_gat_v2.py`, `static/train_v2.py` |
| Static cache | `cache/static/*.npz` (wrist-normalised) | `cache/static_v2/*.npz` (raw two-hand coordinates) |
| Continuous model | `models/bilstm_ctc.py`, `models/conformer_ctc.py` on 63-d landmarks | `models/cont_v2.py` on 351-d features |
| Evaluation | random split only | `static/benchmark.py`: three protocols |
