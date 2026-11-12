---
sidebar_position: 2
title: API reference
---

# API reference

The server (`app/server.py`, Starlette + uvicorn) exposes a small HTTP/WebSocket API on `http://127.0.0.1:8000`.

## `GET /api/info`

```json
{"name": "ISL Fingerspelling", "characters": ["A", "…", "Z"], "one_handed": ["C","I","L","O","U","V"],
 "static_model": "DH-GAT", "continuous_model": "Conformer-CTC", "static_recover": true,
 "continuous_available": true, "ready": true}
```

## `WS /ws/live?mode=static|continuous`

The client sends one message per camera frame and receives one reply per message. Two message types are accepted:

- **Landmark packet** (text, JSON), used when hand tracking runs in the browser: `{"type": "lm", "t": seconds, "w": width, "h": height, "hands": [[[x, y, z] × 21], …]}`, with `x`, `y` normalised to the image and `z` relative to the wrist of that hand. The packet is about 1 KB.
- **JPEG frame** (binary), used by the server-side tracking fallback.

Text commands `{"cmd": "reset" | "space" | "backspace"}` edit the recognised text.

Reply (one JSON object per frame):

| Field | Type | Meaning |
|---|---|---|
| `hands` | `[[ [x,y] × 21 ], …]` | detected hands, normalised image coordinates |
| `letter` | string \| null | current best letter |
| `conf` | float | calibrated confidence 0–1 |
| `top3` | `[[letter, p], …]` | alternatives |
| `status` | string | `no_hand`, `moving`, `holding`, `committed` (static) · `reading`, `no_hand` (continuous) |
| `text` | string | text typed so far (committed letters) |
| `tentative` | string | continuous mode: letters not yet final |
| `event` | object \| null | `{type: "letter"\|"space"\|"word", …}` when something was just typed |
| `ms` | float | server processing time |
| `error` | string | only if the mode is unavailable (e.g. no continuous checkpoint) |

## `POST /api/video`

`multipart/form-data` with `file` (video) and `mode` (`static` or `continuous`). Returns `{"job": "<id>"}`. Errors: `400` (type / mode), `413` (> 300 MB).

## `GET /api/job/{id}`

```json
{"status": "queued|running|done|error", "progress": 0.42, "error": "…", "result": { … }}
```

`result` (when `done`):

| Field | Meaning |
|---|---|
| `mode`, `fps`, `n_frames`, `duration`, `width`, `height` | video info |
| `text` | recognised text |
| `letters` | `[{letter, conf, frame, t, thumb}]` — one key frame per letter |
| `frames` | per-frame HUD data `{f, t, letter, conf, status, text, top3, hands}` |
| `annotated` | file name of the annotated WebM (or `null`) |
| `base` | URL prefix for `thumb` / `annotated` files |
| `elapsed` | seconds spent analysing |

Files of a job are served from `/jobs/{id}/…` (original upload, `keyframes/NN_X.jpg`, `annotated.webm`, `result.json`).

## `POST /api/text2sign`

Body `{"text": "Hello India"}` → `{"text": "HELLO INDIA", "letters": [...], "words": ["HELLO", "INDIA"]}` (non A–Z characters are dropped).

## Python API

```python
from src.pipeline.hands import HandTracker
from src.pipeline.classifier import StaticClassifier
from src.pipeline.hold import Recognizer
from src.pipeline.video import analyze_static
from src.pipeline.continuous import analyze_continuous, ContinuousModel

res = analyze_static("clip.mp4", HandTracker(), StaticClassifier(), "out/")   # dict as above
res = analyze_continuous("clip.mp4", HandTracker(), "out/")
```
