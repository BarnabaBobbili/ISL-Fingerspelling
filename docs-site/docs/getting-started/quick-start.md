---
sidebar_position: 2
title: Quick start
---

# Quick start

## 1 · Start the web app

```bash
python app/server.py            # http://127.0.0.1:8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The status badge (top right) turns green — *models ready* — after a few seconds.

## 2 · Try the four tabs

| Tab | Try this |
|---|---|
| **Live webcam** | *Start camera*, show a letter and hold it still for about a third of a second. The big letter, its confidence bar and the top-3 alternatives are drawn over the video; the typed text appears below. |
| **Upload video** | Drop any video. Static mode finds held signs; *Continuous* mode reads flowing spelling. Click a key-frame thumbnail to jump to that letter. |
| **Text → Sign** | Type `HELLO INDIA`, press *Sign it*. Toggle *Contact alignment* to compare raw and corrected poses. |
| **Alphabet** | All 26 letters on the avatar; click one to watch it. |

## 3 · Use the command line instead

```bash
# analyse a video (writes key frames + annotated video to results/video_out/<name>/)
python -m src.pipeline.cli video my_clip.mp4 --mode static
python -m src.pipeline.cli video my_clip.mp4 --mode continuous

# webcam in an OpenCV window  (q quit · c clear · space · b backspace)
python -m src.pipeline.cli webcam
```

## 4 · Tips for good recognition

- Show **both hands** in the picture when the letter needs them (20 of 26 letters are two-handed — see [Pose library](../avatar/pose-library)).
- Keep the hands **inside the frame**, reasonably large, with a plain background and good light.
- In static mode **hold the sign still**. To type the same letter twice, drop the hands slightly between the two signs (or keep holding for 1.6 s).
- Remove your hands for ≈ 1.4 s to insert a **space**.
