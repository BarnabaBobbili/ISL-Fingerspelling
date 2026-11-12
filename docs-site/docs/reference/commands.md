---
sidebar_position: 1
title: Commands and reproducibility
---

# Commands and reproducibility

All commands run from the repository root. The datasets must be placed under `data/`:

```text
data/
├── static-isl-dataset/Static_ISL/<A-Z>/…      data/static-isl-dataset/Kaggle_ISL_Dataset/<A-Z>/…
├── ISL_DATA/<A-Z>/…
└── continious-Fingerspelling-dataset/{videos/, fingerspelling_annotations.csv, split_info.csv, …}
```

## Run

```bash
python app/server.py                                  # web app on :8000
python -m src.pipeline.cli video clip.mp4 --mode static|continuous
python -m src.pipeline.cli webcam
```

## Static pipeline (rebuild from scratch)

```bash
# 1. raw two-hand landmarks of all images (add --recover for second-hand recovery, --out DIR to redirect)
python src/static/extract_v2.py
python src/static/extract_v2.py --recover --out cache/static_v2r

# 2. CTC-aligned real frames from the continuous videos (needs the v1 BiLSTM + cache/*.npy)
python src/extraction/align_real_frames.py
python src/static/extract_v2.py --real --recover --out cache/static_v2r      # re-detect them with recovery

# 3. train + evaluate DH-GAT v2     (ISL_V2_DIR selects the cache folder, ISL_THREADS the CPU threads)
python src/static/train_v2.py --tag main
python src/static/train_v2.py --tag xsrc --holdout isl_data
ISL_V2_DIR=static_v2r python src/static/train_v2.py --tag main_r

# 4. benchmark the legacy v1 model
python src/static/benchmark.py
```

## Continuous pipeline

```bash
python src/extraction/extract_continuous_v2.py                 # two-hand landmarks of all 1,308 videos
python src/training/build_cont_features.py                     # 351-d per-frame features → cache/cont_feat.pt
python src/training/train_cont_v2.py --kind conformer --parts all --tag conformer_all --bs 32 --epochs 60
python src/training/train_cont_v2.py --kind bilstm    --parts all --tag bilstm_all    --bs 32 --epochs 60
python src/training/train_cont_v2.py --kind conformer --parts raw --tag conformer_raw --bs 32 --epochs 60   # ablation
python src/training/select_best_cont.py                        # lowest validation CER → checkpoints/cont_best.pt
```

## Avatar

```bash
ISL_V2_DIR=static_v2r python -m src.avatar.build_poses         # → app/static/data/poses.json, results/avatar_contacts.csv
```

## Tests

```bash
python tests/test_static_pipeline.py        # synthetic held-letter video, with and without gaps
python tests/test_touch.py                  # contact alignment reduces gaps; bone lengths preserved
python tests/test_ctc.py                    # forced alignment and greedy decoding
```

## Documentation site

```bash
cd docs-site
npm install --prefer-online
npm start          # live preview on http://localhost:3000
npm run build      # static site in docs-site/build
```

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ISL_V2_DIR` | `static_v2` | which `cache/<dir>` the static benchmark/training reads |
| `ISL_THREADS` | 10 (static) / 8 (continuous) | PyTorch CPU threads |

## Approximate run times (16-core laptop CPU)

| Step | Time |
|---|---:|
| raw landmark extraction of all images | 7 min |
| second-hand recovery extraction | 9 min |
| CTC alignment + frame extraction | 7 min |
| DH-GAT v2 training | 19 – 25 min |
| continuous landmark extraction | 25 min |
| continuous feature build | 2.5 min |
| continuous model training | 25 – 45 min each |
