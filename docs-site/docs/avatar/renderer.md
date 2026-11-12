---
sidebar_position: 3
title: The 3D signer
---

# The 3D signer

`app/static/js/avatar.js` draws an **upper-body signer** — head, shoulders, torso in a kurta, two sleeved arms and two skinned hands — with [Three.js](https://threejs.org/) (vendored in `app/static/vendor/`, so the app works offline).

![The signer spelling the letter H](/img/av_H.png)

## Hands: a skinned mesh driven by the landmarks

The hands are a real skinned mesh (the MIT-licensed *generic hand* from the WebXR input-profiles project, 95 KB each, 25 joints). `signer/rig.js` (`HandRig`) poses it directly from the 21 MediaPipe landmarks of each hand:

1. every bone is **placed at its landmark** (world position);
2. the **wrist** gets its full orientation from the palm frame (wrist → middle knuckle, and the palm normal from the index and pinky knuckles);
3. every finger bone is turned so that its axis points at the next landmark, as the **shortest rotation from the orientation its parent just gave it** — each bone inherits the twist of the one before, so fingers stay flat against the palm and curl in the right plane;
4. skinning then deforms the mesh, so a fingertip that the landmarks put on a palm really touches it.

:::note A detail that mattered
This hand model's skeleton is *flat*: all 25 bones hang directly off one `Armature` node with absolute transforms. Reading "child offsets" and "parent rotations" from it, as most retargeting code does, produces shredded fingers. The rig therefore works from **bind-pose world transforms** and carries rotations along the finger chains itself. A test feeds the hand its own bind-pose joints and requires every bone to come back to within 0.001 of its original position and rotation.
:::

### Left and right hands

Slot 0 is the signer's **right** hand (it appears on the viewer's left), slot 1 the left. MediaPipe's depth is only an estimate, so for about one pose in ten the 3D geometry implies the wrong handedness for its slot. `src/avatar/build_poses.py` detects that with the sign of the triple product *(index knuckle × pinky knuckle) · thumb knuckle* and mirrors such a hand across its own palm plane — a valid hand of the opposite chirality with the same finger shapes — before the contact alignment runs. (Eight letters had a hand mirrored; the list is stored in `poses.json` under `mirrored`.)

## Body and arms

`signer/body.js` builds a stylised figure in the app's palette (peacock kurta, marigold trim). The arms use **two-bone inverse kinematics**: from the shoulder to the pose's wrist, with upper-arm and forearm lengths of 3.15 and 2.75 hand sizes. The elbow is placed on the circle of solutions nearest to a point *behind the hand's own axis* (so the forearm roughly follows the hand), blended with an outward-and-down default so it never flips.

**One-handed letters** (C, I, L, O, U, V) are signed with the right hand. The other arm simply **hangs relaxed at the side**, with its hand out of the way — there is no floating second hand.

## Look

- Physically-based materials, three skin tones, soft studio environment lighting, a warm key light with soft shadows and a cool rim light.
- Camera presets: **Front** (the whole signer), **Hands** (close-up) and **Signer’s view** (from the signer's own eyes, looking down at the hands). Drag to orbit, scroll to zoom.

## Animation

`Avatar.play(text)` builds a timeline: **0.55 s** to move into a letter and **0.85 s** to hold it, a relaxed pose between words, and a short relax toward rest before a repeated letter so that the repeat is visible. All times scale with the *Speed* slider. For every frame:

```text
P(t) = lerp(P_A, P_B, ease(t))                 # all joint positions of both hands
for each bone (parent → child), in order:
    dir = normalise(P(child) − P(parent))        # direction from the blend
    len = lerp(len_A, len_B, ease(t))            # bone length from the two poses
    P(child) = P(parent) + dir · len             # forward-kinematics re-projection
```

The re-projection stops fingers shrinking mid-transition when two poses point in very different directions. The arms follow the hands through the IK.

## On the page

The current **character** (large) and the **whole word** with the current letter highlighted are overlaid on the signer, with tags such as *two-handed* and *4 contacts aligned*. Each letter can also be **spoken**, the text can be **dictated** (Chrome or Edge), and *Record a clip* saves the animation as a WebM video.
