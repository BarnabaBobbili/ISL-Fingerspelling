---
sidebar_position: 3
title: The 3D hands
---

# The 3D hands

`app/static/js/avatar.js` draws two **skinned 3D hands with cloth sleeves** with [Three.js](https://threejs.org/) (vendored in `app/static/vendor/`, so the app works offline). There is deliberately no head or body: the signing is shown the way a learner needs it, as large hands and forearms in the orientation given by the data.

![The hands spelling the letter H](/img/av_H.png)

## Hands: a skinned mesh driven by the landmarks

The hands are a real skinned mesh (the MIT-licensed *generic hand* from the WebXR input-profiles project, 95 KB each, 25 joints). `signer/rig.js` (`HandRig`) poses it directly from the 21 MediaPipe landmarks of each hand:

1. every bone is **placed at its landmark** (world position);
2. the **wrist** gets its full orientation from the palm frame (wrist → middle knuckle, and the palm normal from the index and pinky knuckles);
3. every finger bone is turned so that its axis points at the next landmark, as the **shortest rotation from the orientation the previous bone in the chain just received** — each bone inherits the twist of the one before, so fingers stay flat against the palm and curl in the right plane;
4. skinning then deforms the mesh, so a fingertip that the landmarks put on a palm really touches it.

:::note Implementation note
This hand model's skeleton is *flat*: all 25 bones hang directly off one `Armature` node with absolute transforms. Reading "child offsets" and "parent rotations" from it, as most retargeting code does, produces shredded fingers. The rig therefore works from **bind-pose world transforms** and carries rotations along the finger chains itself. A test feeds the hand its own bind-pose joints and requires every bone to come back to within 0.001 of its original position and rotation.
:::

### Left and right hands — never mirrored

Each pose records the handedness its landmarks actually show (the sign of the triple product *(index knuckle × pinky knuckle) · thumb knuckle*, stored as `chir` in `poses.json`). The renderer keeps a **right-type and a left-type mesh per hand slot and shows the one that matches**, so every letter keeps exactly the shape and orientation in the data. (Forcing each slot to a fixed handedness by mirroring the pose across the palm plane was rejected because it changed the shape of some signs.) When two consecutive letters need different types for a hand, the mesh switches halfway through the movement.

## Sleeves

`signer/sleeves.js` adds a cloth forearm behind each hand. It follows the hand's own axis (from the knuckles back to the wrist), is blended a little toward "down" so it never points at the camera, and runs out of the frame. A short skin wrist and a marigold trim ring join it to the hand.

**One-handed letters** (C, I, L, O, U, V) show a single hand, centred. The other hand simply **waits below the frame** and rises into view when a letter needs it — so there is never a floating second hand, and the hands also lower out of sight between words.

## Look and camera

- Physically-based materials, three skin tones, soft studio environment lighting, a warm key light with soft shadows and a cool rim light.
- Camera buttons: **Front**, **Close** and **Above**. Drag to orbit, scroll to zoom.

## Animation

`Avatar.play(text)` builds a timeline: **0.55 s** to move into a letter and **0.85 s** to hold it, hands lowering out of frame between words, and a short relax toward rest before a repeated letter so that the repeat is visible. All times scale with the *Speed* slider. For every frame:

```text
P(t) = lerp(P_A, P_B, ease(t))                 # all joint positions of both hands
for each bone (parent → child), in order:
    dir = normalise(P(child) − P(parent))        # direction from the blend
    len = lerp(len_A, len_B, ease(t))            # bone length from the two poses
    P(child) = P(parent) + dir · len             # forward-kinematics re-projection
```

The re-projection stops fingers shrinking mid-transition when two poses point in very different directions.

## On the page

The current **character** (large) and the **whole word** with the current letter highlighted are overlaid on the stage, with tags such as *two-handed* and *4 contacts aligned*. Each letter can also be **spoken**, the text can be **dictated** (Chrome or Edge), and *Record a clip* saves the animation as a WebM video.
