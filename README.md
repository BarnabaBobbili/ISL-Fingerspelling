# Varna — ISL fingerspelling

Recognition **and** synthesis of **Indian Sign Language (ISL) fingerspelling** (A–Z), running entirely on your computer.

| | |
|---|---|
| **Sign → text / speech** | live webcam or uploaded video; the **letter and its confidence score are drawn on top of the video**; text is built up and can be spoken |
| **Continuous fingerspelling** | reads flowing spelling (names, words) and extracts **one key frame per letter** |
| **Text / speech → sign** | a 3D upper-body signer with skinned hands signs typed or dictated text; one-handed letters leave the other arm at rest; joining points are brought to touch with **geometric transformations** (Kabsch alignment + FABRIK inverse kinematics) |
| **All 26 characters** | alphabet reference rendered on the avatar |

## Models

- **Static letters — DH-GAT v2**: a dual-hand graph-attention network that sees both hands in one coordinate frame (intra-hand + cross-hand attention), with calibrated confidence.
- **Continuous — Conformer-CTC / BiLSTM-CTC** on per-frame features (pretrained DH-GAT embedding + class posteriors + raw two-hand geometry + motion).

See the documentation for architecture, training, and **honest benchmark numbers** (in-domain, cross-source and real news-video tests).

## Results

| Task | Test | Result |
|---|---|---|
| Static letters (DH-GAT v2) | real news-video letters, held-out videos | **95.9 %** (previous model: 38.4 %) |
| Static letters | original random split (near-duplicate frames, optimistic) | 98.0 % |
| Static letters | **cross-source**: ISL_DATA never seen in training | **70.0 %** |
| Confidence calibration | expected calibration error on real videos | **0.011** (previous: 0.170) |
| Continuous (Conformer-CTC v2) | character error rate, held-out test videos | **3.63 %** (previous best: 14.75 %) |
| Live decoding | error at 2× signing speed / on a 10 fps camera | **5.8 % / 5.7 %** |
| 3D avatar | mean joint-contact gap after alignment | **0.226 → 0.018** hand sizes |

The cross-source and real-video numbers are lower than a random split would suggest on purpose: they are the honest ones. See the documentation for the full benchmark and the limitations.

## Quick start

```bash
pip install -r requirements.txt
python app/server.py            # open http://127.0.0.1:8000
```

Live recognition tracks the hands **in the browser** and sends only joint positions to the local server; a light/dark site with the documentation built in is served at the same address (`/docs`).

Command line:

```bash
python -m src.pipeline.cli video my_clip.mp4 --mode static        # or --mode continuous
python -m src.pipeline.cli webcam
```

The trained weights needed to run the app are in `checkpoints/`. The datasets are **not** included (see the manual for where to place them if you want to retrain).

## Documentation

A full technical manual (Docusaurus) is in [`docs-site/`](docs-site):

```bash
cd docs-site
npm install --prefer-online
npm start            # http://localhost:3000
```

## Repository layout

```text
app/            web application (Starlette server + single-page UI + Three.js avatar)
src/pipeline/   hand tracking, classifier, hold detection, continuous recogniser, video + CLI
src/models/     DH-GAT (v1, v2), BiLSTM-CTC, Conformer-CTC, continuous v2 models
src/static/     static training, benchmark, extraction
src/avatar/     pose library + contact geometry
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
