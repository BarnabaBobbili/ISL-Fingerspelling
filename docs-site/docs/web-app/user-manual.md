---
sidebar_position: 1
title: User manual
---

# User manual

Start the server with `python app/server.py` and open [http://127.0.0.1:8000](http://127.0.0.1:8000). The header has five pages — **Recognise**, **Sign**, **Alphabet**, **Results** and **Docs** (this manual, shown inside the site) — a status light (*models ready*) and a **theme switch** (light or dark, remembered).

## Recognise

![The Recognise page](/img/live_tab.png)

Two sources, chosen with the tabs: **Live camera** and **Video file**.

### Live camera

| Control | What it does |
|---|---|
| **Start camera / Stop** | opens the webcam (the browser asks permission). Hand tracking runs **inside the page**, so no pictures leave the computer — only about 1 KB of joint positions per frame goes to the local server. |
| **Signing style** | *Flowing spelling* reads names and words spelled at a natural pace; *Single letters* types a letter each time one is held still. |
| **Mirror the preview** | flips the picture like a mirror (recognition is unaffected) |
| **Show tracked joints** | draws the detected joints on the video |
| **Speak, Copy, space, ⌫, Clear** | work on the text |

On the picture: a **tile with the current letter**, a **confidence meter** and a status (*reading*, *holding*, *typed*), the **top-3 alternatives**, and the **text so far** along the bottom. In flowing mode the newest letters are shown **greyed**: they are tentative and may still change; a letter turns solid about 0.4 s after it was signed.

Under the text box: frames per second, the delay between a frame and its answer, the number of hands seen, and whether tracking runs in the browser or (as a fallback) on the server.

**Tips.** Keep both hands in view for two-handed letters, use even lighting and a plain background, and lower the hands out of the picture briefly between words (this ends a word in flowing mode).

### Video file

Drop a video (mp4, mov, avi, mkv or webm, up to 300 MB), choose the signing style and wait for the progress bar. Then:

- the video plays with the same HUD and joints drawn **on top**, synchronised with playback;
- **Key frames** — one thumbnail per recognised letter with its confidence and time; click one to jump there;
- **Text** with *Speak*, *Copy*, a download link for the **annotated video** (WebM) and the raw **JSON**.

Everything is processed locally. Uploaded files are kept under `results/web_jobs/` on the local computer.

## Sign

![The Sign page](/img/av_H.png)

Type text (A–Z and spaces; other characters are ignored) or press **Dictate** and speak (Chrome or Edge), then **Sign it**. The hands spell the text; the big letter and the word (current letter highlighted) are overlaid on the stage. Click a letter chip to jump to a single letter.

- **Speed** and **Say each letter aloud**.
- **Bring touching joints together** toggles between the raw landmark pose and the corrected one; **Mark the contact points** shows where the aligned joints meet; **Show the joint skeleton** overlays the 21 joints.
- **Skin tone**, and the camera buttons **Front / Close / Above**.
- **Record a clip** saves the animation as a WebM video.

## Alphabet

All 26 letters as the hands make them, with *one hand / two hands* and the number of aligned contacts. Filter by hand count; click a letter for a larger view and **Sign it**.

## Results

The measured numbers behind the project — static letters, the unseen-camera test, whole names, live decoding and contact alignment — and where the system falls short.

## Browser notes

- Hand tracking in the browser uses WebAssembly (and the GPU when available); it works in current Chrome, Edge and Firefox. If it cannot start, the page falls back to server-side tracking automatically.
- Microphone dictation needs **Chrome or Edge** and an internet connection (the browser's own speech service); everything else is offline.
- Webcam access requires `http://localhost` or HTTPS.
