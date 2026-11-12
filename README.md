# ISL fingerspelling

Recognition **and** generation of **Indian Sign Language (ISL) fingerspelling** (A–Z). All processing runs on the local computer.

The project accompanies the M.Tech thesis *Dual-Hand Graph Attention and Bidirectional Sequence Modeling for Continuous Indian Sign Language Fingerspelling Recognition*.

| | |
|---|---|
| **Sign → text / speech** | live webcam or uploaded video; the **letter and its confidence score are drawn on top of the video**; the text is built up and can be spoken |
| **Continuous fingerspelling** | reads flowing spelling (names, words) and extracts **one key frame per letter** |
| **Text / speech → sign** | two skinned 3D hands with sleeves sign typed or dictated text; for one-handed letters the other hand waits out of frame; joining points are brought to touch with **geometric transformations** (Kabsch alignment and FABRIK inverse kinematics) |
| **All 26 characters** | alphabet reference rendered on the 3D hands |

## Models

- **Static letters — DH-GAT**: a Dual-Hand Graph Attention network that represents both hands in one coordinate frame (intra-hand and cross-hand attention), with calibrated confidence.
- **Continuous — Conformer-CTC / BiLSTM-CTC** on per-frame features (pretrained DH-GAT embedding, class posteriors, raw two-hand geometry and motion).

The documentation describes the architecture, the training procedure and the benchmark, which reports three tests (in-domain, cross-dataset and real-video frames).

## Results

| Task | Test | Result |
|---|---|---|
| Static letters (DH-GAT) | real-video frames, held-out videos | **95.9 %** (baseline: 38.4 %) |
| Static letters | original random split (near-duplicate images) | 98.0 % |
| Static letters | **cross-dataset**: Static Dataset B never seen in training | **70.0 %** |
| Confidence calibration | expected calibration error on real-video frames | **0.011** (baseline: 0.170) |
| Continuous (Conformer-CTC) | character error rate, held-out test videos | **3.63 %** (baseline: 14.75 %) |
| Live decoding | error at 2× signing speed / on a 10 fps camera | **5.8 % / 5.7 %** |
| 3D hands | mean joint-contact gap after alignment | **0.214 → 0.017** hand sizes |

The cross-dataset and real-video figures are lower than a random split suggests because they reflect conditions that differ from the training data. The documentation contains the full benchmark and the limitations.

## Quick start

```bash
pip install -r requirements.txt
python app/server.py            # open http://127.0.0.1:8000
```

Live recognition tracks the hands **in the browser** and sends only joint positions to the local server. The web application has a light and a dark theme, and the documentation is built into it (`/docs`).

Command line:

```bash
python -m src.pipeline.cli video my_clip.mp4 --mode static        # or --mode continuous
python -m src.pipeline.cli webcam
```

The trained weights needed to run the application are in `checkpoints/`. The datasets are **not** included; the manual states where to place them for retraining.

## Documentation

A technical manual (Docusaurus) is in [`docs-site/`](docs-site):

```bash
cd docs-site
npm install --prefer-online
npm start            # http://localhost:3000
```

## Repository layout

```text
app/            web application (Starlette server, single-page UI, Three.js 3D hands)
src/pipeline/   hand tracking, classifier, hold detection, continuous recogniser, video and CLI
src/models/     DH-GAT (baseline and proposed), BiLSTM-CTC, Conformer-CTC
src/static/     static training, benchmark, extraction
src/avatar/     pose library and contact geometry
src/training/   continuous feature building and training
tests/          pipeline, geometry and CTC tests
docs-site/      documentation manual
```

## Tests

```bash
python tests/test_ctc.py
python tests/test_touch.py
python tests/test_static_pipeline.py
```

## Dataset

The continuous-recognition experiments use the *Continuous Fingerspelling Dataset for Indian Sign Language* (Kirandevraj, Kurmi, Namboodiri and Jawahar, WSLP @ AACL-IJCNLP 2025; CC BY-NC 4.0), <https://kirandevraj.github.io/ISL-Fingerspelling>.
