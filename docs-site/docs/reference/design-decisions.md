---
sidebar_position: 2
title: Design decisions
---

# Design decisions

Every significant choice, with the reason and the alternative that was rejected — material for a viva.

## Original decisions (stages 1 – 6)

| ID | Decision | Why | Rejected alternative |
|---|---|---|---|
| DD-001 | CTC vocabulary = 26 lower-case letters, blank = 26 | digits/symbols in the transcripts are annotation artefacts with near-zero frequency | keep all 38 characters |
| DD-002 | Use the authors' video-level train/test split; hold out 10 % of training *videos* for validation | avoids segments of one video in two splits; comparable with the paper | random segment-level split (signer leakage) |
| DD-003 | Missing landmarks: forward-fill ≤ 5 frames, then zero-fill; drop segments with > 50 % missing | brief occlusions should not create jumps to the origin | zero-fill everything |
| DD-004 | Cache landmarks as `.npy` per segment | memory-mapped, no extra dependency | HDF5 |
| DD-005 | Normalise: wrist → origin, scale by wrist → middle-MCP distance | wrist is the most stable joint; scale is robust to hand orientation | image-frame coordinates |
| DD-006 | BiLSTM-CTC baseline (Linear → 2 × BiLSTM(256) → CTC) | standard strong baseline, future context helps co-articulation | uni-directional LSTM |
| DD-007 | Conformer-CTC main model (d = 144, 4 blocks) | convolution for local handshape patterns, attention for global context | pure Transformer (no local bias) |

## Decisions of this phase

| ID | Decision | Why | Rejected alternative |
|---|---|---|---|
| DD-008 | Report **three** static tests (in-domain, cross-source, real video) | random split of near-duplicate frames gave 98 % while real accuracy was 33 % | single random split |
| DD-009 | Build real frames with **CTC forced alignment** | even time division mislabels frames; alignment gives true letter timing | manual labelling (too slow), even split |
| DD-010 | **Shared-frame two-hand graph** with cross-hand attention | most ISL letters are defined by contact between the hands | per-hand normalisation (information loss) |
| DD-011 | Slot hands by **wrist-x**, mirror-augment | MediaPipe's Left/Right label flips between mirrored and normal video | use the handedness label |
| DD-012 | **Temperature scaling** on real-video validation logits | one parameter, preserves accuracy, large ECE gain | no calibration; Platt/isotonic (more parameters than needed) |
| DD-013 | Model selection by **real-video validation** accuracy | static validation saturates at 98 % (near-duplicates) | static validation accuracy |
| DD-014 | Second-hand recovery with the **same 0.4 threshold** | recovers J/S/B hands with ≤ 2 % false hands on one-handed letters | extra pass at threshold 0.15 (hallucinated hands: I 20 %, O 48 %) |
| DD-015 | Continuous input = trunk embedding + posteriors + raw geometry + motion | pretrained hand-shape knowledge plus a path independent of the trunk | landmarks only (v1) |
| DD-016 | Ablation without trunk features (`--parts raw`) | shows what pre-training contributes | assume it helps |
| DD-017 | Smaller convolution kernel (15) and an **intermediate CTC loss** | ~0.5 s per letter at 30 fps; auxiliary loss regularises small data | kernel 31, single loss |
| DD-018 | Hold detector: motion-still + stable arg-max + re-arming | per-frame guesses include transitions; repeats need a release | commit on every confident frame |
| DD-019 | Avatar poses = **medoid of real samples**, not hand-modelled | the avatar signs exactly what the recogniser knows | manually posed hands |
| DD-020 | Contact discovery in the **image plane**; depth registration before alignment | MediaPipe z is relative to each wrist | full 3D distances (polluted by unobservable depth offset) |
| DD-021 | Kabsch rigid alignment (clamped to 20°) + FABRIK | rigid body keeps the hand shape; IK resolves competing contacts | per-joint snapping (distorts the hand) |
| DD-022 | Server on **Starlette**, WebSocket pull-based frames | FastAPI/Starlette version conflict on the target machine; one frame in flight bounds latency | FastAPI; push-based streaming (queueing) |
| DD-023 | HUD drawn client-side on a canvas | zero extra latency, identical look for webcam and uploaded video | server-side annotated frames only |
| DD-024 | Three.js vendored locally | works offline | CDN |
