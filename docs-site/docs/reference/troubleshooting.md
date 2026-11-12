---
sidebar_position: 3
title: Troubleshooting
---

# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| The status badge stays on *loading models…* | The first start loads PyTorch and MediaPipe (5 – 15 s). If it never turns green, look at the terminal for a traceback. |
| `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` | FastAPI and Starlette versions are incompatible. This project uses Starlette directly, so it is not affected; if you add FastAPI, upgrade both together. |
| *Camera unavailable* | Allow camera access in the browser, close other apps using the camera. Cameras need `http://localhost` or HTTPS; opening the page via a LAN IP over HTTP will be blocked by the browser. |
| Letters flicker or never lock | Move closer, light the hands evenly, keep a plain background; hold the sign still for ≈ 0.3 s. Check that the **hands count** under the text box matches what you show. |
| A two-handed letter is read as another letter when only one hand is found | The detector missed the second hand (overlapping hands). Slightly separate the hands or change the angle. |
| Confidence is high but the letter is wrong | Calibration was done on the news-video domain; on an unfamiliar camera the model is over-confident. See [Limitations](../evaluation/limitations). |
| *continuous model unavailable* | `checkpoints/cont_best.pt` is missing. Train one (`train_cont_v2.py` + `select_best_cont.py`) or pull the committed weights. |
| Uploaded video is slow | Analysis runs on the CPU at ≈ 10 – 30 frames per second; long 1080p clips take minutes. Close other heavy programs. |
| The annotated video does not play | It is a WebM (VP8); use Chrome, Edge or Firefox. |
| Dictation button does nothing | Needs Chrome/Edge and internet (the browser's speech service). |
| Avatar page is blank | WebGL is required. Check `chrome://gpu`; update graphics drivers or enable hardware acceleration. |
| `npm install` fails with `No matching version found for @babel/types` | Stale registry metadata: run `npm install --prefer-online`. |
| Windows: training stops with *low memory* | Close other applications; lower threads (`ISL_THREADS=4`); training needs ≈ 4 GB free. |
