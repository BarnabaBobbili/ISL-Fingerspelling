---
sidebar_position: 2
title: Landmark extraction
---

# Landmark extraction

All models start from MediaPipe **HandLandmarker** (`hand_landmarker.task`): up to two hands, 21 joints each, `(x, y, z)` with `x, y ∈ [0, 1]` of the image and `z` relative to that hand's wrist.

## The cache keeps raw coordinates

Storing every hand **already wrist-centred and scaled** would discard where one hand is relative to the other, so the cache stores the raw detector output:

| Field | Shape | Meaning |
|---|---|---|
| `lm` | `(N, 2, 21, 3)` | raw MediaPipe coordinates |
| `present` | `(N, 2)` | which slots hold a hand |
| `wh` | `(N, 2)` | image width / height — needed to undo the aspect ratio |
| `label`, `name` | | letter and provenance |

Two conventions matter:

1. **Hands are slotted by wrist-x (image-left first), not by MediaPipe's *Left/Right* label.** That label flips between mirrored webcam preview and normal video, so using it would make the same pose look different. A lone hand always sits in slot 0.
2. **Aspect-corrected units.** Coordinates are multiplied by `(W, H, W)` before use so that distances are isotropic. A square 300×300 image squashed to 4:3 changes the hand shape, which was measured as H→M and N→R errors and is why the tests letter-box images instead of resizing them.

Extraction is parallel (`src/static/extract_v2.py`), about 100 – 130 images per second on 8 workers.

## Second-hand recovery

Overlapping hands are hard for the detector: it often finds one hand and misses the other. For *J*, **0 %** of the images of Static Dataset A yielded two hands, for *S* 30 %, for *B* 80 %.

**Recovery**: if exactly one hand is found, blank out the found hand (its convex hull plus a margin, filled with the median colour) and run the detector again; accept a new hand only if its wrist lies outside the blanked area.

| Letter | Two hands before | After recovery |
|---|---:|---:|
| J | 0 % | 52 % |
| S | 30 % | 76 % |
| B | 80 % | 100 % |

False recoveries on the six one-handed letters are 0 – 2 % (O: 18 %). A variant that also lowered the detection threshold to 0.15 recovered more hands but hallucinated on one-handed letters (I: 20 %, O: 48 %), so it was rejected.

Overall, two-hand detections on Static Dataset B rose from **49 % to 71.5 %**. Recovery is used by the application's static model: `extract_v2.py --recover` for training data and `HandTracker(recover=True)` at run time.

## Continuous videos

`src/extraction/extract_continuous_v2.py` runs the same detector over every frame of all 1,308 videos (127,348 frames) and stores `cache/cont_v2/<uid>.npz` with `lm (T, 2, 21, 3)`, `present` and `wh`. This took about 25 minutes on 6 worker processes.
