---
sidebar_position: 1
title: User manual
---

# User manual

Start the server with `python app/server.py` and open [http://127.0.0.1:8000](http://127.0.0.1:8000). The coloured dot in the header shows the server state (*connecting*, *loading models…*, **models ready**, *server offline*).

## Live webcam

![Live tab](/img/live_tab.png)

| Control | What it does |
|---|---|
| **Start camera / Stop** | opens the webcam (the browser asks for permission) and the WebSocket to the server |
| **Recognition mode** | *Static letters* — hold each sign; *Continuous fingerspelling* — spell fluently, the model re-reads the last few seconds |
| **Mirror preview** | flips the picture like a mirror (recognition is unaffected) |
| **Show hand skeleton** | draws the detected joints on the video |
| **Speak** | reads the recognised text aloud (browser speech synthesis) |
| **Copy / Space / ⌫ / Clear** | edit the text |

On the video:

- **Top-left card** — the current letter, a **confidence bar** and percentage, and a status (`NO HAND`, `MOVING`, `HOLDING`, `LOCKED`).
- **Three chips** — the top-3 alternatives.
- **Bottom banner** — the text so far.

Below the text box: frames per second, round-trip latency and the number of hands detected.

### Tips

- Static mode: hold each letter still for ≈ 0.3 s; the card turns green and says `LOCKED` when it is typed.
- Remove both hands for ≈ 1.4 s for a space (static) or to finish a word (continuous).
- Continuous mode has no space symbol; words are separated by the pause.

## Upload video

1. Drop a video (mp4, mov, avi, mkv, webm; up to 300 MB) on the panel or click to choose.
2. Pick **Static** (held signs) or **Continuous** (flowing spelling).
3. A progress bar shows the analysis. When done:
   - the video plays with the **same HUD and skeleton drawn on top**, synchronised to the playback;
   - **Extracted key frames** — one thumbnail per recognised letter with its confidence and time; click a thumbnail to jump to that moment;
   - **Result** — the final text with *Speak*, *Copy*, a download link for the **annotated video** (WebM) and the raw **JSON**.

Everything is processed locally; uploaded files are stored under `results/web_jobs/` on your computer.

## Text → Sign

![Avatar](/img/av_H.png)

1. Type text (A–Z and spaces; other characters are ignored) or press **Dictate** and speak (Chrome/Edge).
2. **Sign it** plays the animation. The big letter and the word (current character highlighted) are shown on top of the avatar.
3. **Speed** changes the pace; **Speak while signing** says each letter's name.
4. **Geometry** options: *Contact alignment* (toggle between raw and corrected poses), *contact markers*, *skeleton*, *skin tone*.
5. **Record clip** saves the animation as a WebM video.

Click any chip under the text box to jump to that single letter.

## Alphabet

![Alphabet](/img/alphabet_tab.png)

All 26 letters rendered on the avatar, with *1 hand / 2 hands* and the number of aligned contacts. Click a card to watch it in the *Text → Sign* tab.

## Keyboard / browser notes

- Microphone dictation needs **Chrome or Edge** and internet access for the browser's speech service; everything else is offline.
- Speech synthesis uses your system voices; the language is set to English (India) when available.
- If the camera is busy in another app, close it and press *Start camera* again.
