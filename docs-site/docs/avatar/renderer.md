---
sidebar_position: 3
title: 3D renderer
---

# The 3D renderer

`app/static/js/avatar.js` draws **solid hands**, not a skeleton, using [Three.js](https://threejs.org/) (vendored in `app/static/vendor/`, so the app works offline).

![Avatar signing H](/img/av_H.png)

## Hand model

Built every frame from the 21 joint positions:

| Part | Geometry |
|---|---|
| Fingers | tapered cylinders (child radius → parent radius) between consecutive joints |
| Joints | spheres with the joint radius — they round off every knuckle and fingertip |
| Palm | custom mesh: seven outline joints (wrist, thumb base, four knuckles) pushed outward, a top and bottom ring offset along the palm normal, two domed centres; shared vertices give soft rounded edges (normals recomputed each frame) |
| Forearm | skin-coloured cylinder pointing away from the fingers |
| Sleeve and cuff | larger cloth cylinder plus a torus cuff, long enough to leave the frame |

## Look

- Physically-based standard material, three skin tones (light, medium, deep).
- Soft studio environment (`RoomEnvironment`) for reflections, a warm key light with **soft shadows** (so fingers shadow the palm), a cool rim light and a hemisphere fill.
- ACES tone-mapping and sRGB output, subtle idle float, a radial studio backdrop.
- Orbit controls: drag to rotate, scroll to zoom.

## Animation

`Avatar.play(text)` builds a timeline from the text:

| Item | Transition | Hold |
|---|---:|---:|
| letter | 0.50 s | 0.85 s |
| space (back to the relaxed pose) | 0.55 s | 0.35 s |
| repeated letter (e.g. the second L) | a short relax toward rest (28 %), then the letter again | |

All times scale with the *Speed* slider. Interpolation is eased (smoothstep). For every frame:

```text
P(t) = lerp(P_A, P_B, ease(t))                 # all joint positions
for each bone (parent → child), in order:
    dir = normalise(P(child) − P(parent))        # direction from the blend
    len = lerp(len_A, len_B, ease(t))            # bone length from the two poses
    P(child) = P(parent) + dir · len             # forward-kinematics re-projection
```

The re-projection stops fingers shrinking mid-transition when two poses point in very different directions.

## On top of the video

While signing, the avatar page overlays the **current character** (large) and the **whole word** with the current letter highlighted, and tags such as *two-handed sign* and *4 contact points aligned*. The current letter can also be **spoken**, and the text can be **dictated** with the microphone (Chrome/Edge speech recognition). *Record clip* saves the animation as a WebM video.

## Alphabet thumbnails

The *Alphabet* tab renders all 26 poses once through a shared off-screen renderer and shows them as cards (hand count and number of aligned contacts); clicking a card plays the letter.
