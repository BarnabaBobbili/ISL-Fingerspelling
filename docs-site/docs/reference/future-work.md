---
sidebar_position: 4
title: Future work
---

# Future work

| Idea | Expected benefit |
|---|---|
| **Personal calibration** — record 2 – 3 minutes of one's own signs (`collect_webcam_data.py`) and fine-tune DH-GAT | the largest single gain on a new camera/signer; also re-fit the temperature on that data |
| **Signer-independent split** — collect or annotate signer identity | a true measure of generalisation to new people |
| **Better two-hand detection** — a detector trained on overlapping hands, or running MediaPipe on tiles/crops | the cross-dataset error is dominated by missed second hands |
| **Character language model** for continuous decoding (prefix beam search with a character n-gram over Indian names) | lower CER on names |
| **Word segmentation** in continuous mode — add a space symbol by re-annotating pauses | words without relying on pauses |
| **Out-of-fold features** for the continuous trunk — train the DH-GAT trunk in K folds of videos | removes the slight optimism of stacking a model on features it was trained on |
| **Digits and ISL words** — extend the vocabulary (digits 1–9 images already exist) | broader coverage |
| **Hand model detail** — hand meshes with nails and finger webbing, and further skin tones | a more natural appearance |
| **Co-articulated hand motion** — learn transition paths from the continuous dataset | fluent, human-like spelling |
| **ONNX / WebAssembly in the browser** — run MediaPipe and the classifier client-side | no server needed; static hosting of the whole app |
