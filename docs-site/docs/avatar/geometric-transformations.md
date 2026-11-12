---
sidebar_position: 2
title: Geometric transformations (touching joints)
---

# Geometric transformations: making the joining points touch

A two-handed ISL letter is defined by contacts such as *"index fingertip rests on the other palm"*. Raw landmarks only give noisy **joint centres**, so a raw pose shows gaps or fingers that poke through the other hand. `src/avatar/touch.py` repairs every pose with four geometric steps.

![Raw versus aligned](/img/F_compare.png)

*Letter **F**: left — raw pose, the fingertips of the upper hand penetrate the lower hand; right — after contact alignment the fingertips rest on the surface (yellow dots mark the aligned contact points).*

## What "touching" means

A contact pair `(i, j)` joins joint *i* of hand A with joint *j* of hand B. The hands touch when the distance between the two joint centres equals the **sum of their radii** — skin on skin:

```text
target_ij = r_i + r_j            r in hand-size units: wrist 0.165 · MCP 0.115 · PIP 0.10 · DIP 0.09 · tip 0.08 …
gap_ij    = |B_j − A_i| − target_ij        (0 = touching, < 0 = interpenetrating)
```

## The four steps

### 0 · Depth registration

MediaPipe's `z` is measured relative to **each hand's own wrist**, so the depth offset between the two hands is *unknown*. Hand B is slid along `z` until the mean depth of its contact joints equals that of hand A's:

```text
Δz = mean_ij ( A_i.z − B_j.z )          B ← B + (0, 0, Δz)
```

### 1 · Bone regularisation

Each finger is rebuilt by forward kinematics, keeping the wrist and the bone directions but replacing bone lengths by the canonical (median) lengths. Result: no stretching fingers.

### 2 · Rigid alignment (Kabsch / Procrustes)

Hand B is moved **as a rigid body** so all its contact joints reach their targets. For each pair the desired position keeps the approach direction:

```text
q_j = A_i + target_ij · unit(B_j − A_i)
```

Wrist and MCP joints are added as low-weight *anchors* (they should stay where they are). The optimal rotation `R` and translation `t` minimising the weighted squared error `Σ w |R·B_j + t − q_j|²` come from the SVD of the covariance matrix (**Kabsch algorithm**). The rotation is **clamped to 20°** so the hand never twists unnaturally, the translation is recomputed for the clamped rotation, and the step is iterated three times.

### 3 · Finger inverse kinematics (FABRIK)

A single rigid motion cannot satisfy several contacts at once. Any pair still more than 0.01 hand sizes from its target is fixed by **FABRIK** (Forward And Backward Reaching Inverse Kinematics) on that finger's chain: the base joint (MCP) stays fixed, the chain is pulled toward the target from the tip and pushed back from the root, alternately, preserving **bone lengths**; if the target is out of reach the finger stretches straight toward it. Joints beyond the contact joint follow it.

Steps 2 and 3 run for up to four rounds and the round with the smallest mean `|gap|` wins.

### 4 · Report

`results/avatar_contacts.csv` lists, for every letter, the mean absolute contact gap before and after.

## Results

| | Value |
|---|---:|
| letters with contacts | 17 of 20 two-handed (J, Q, S have no reliable contact) |
| mean absolute gap, raw | **0.198** hand sizes ≈ 1.9 cm |
| mean absolute gap, after alignment | **0.016** hand sizes ≈ 1.5 mm |
| worst letter after alignment | 0.044 (P) |

Examples: A 0.70 → 0.001 · G 0.32 → 0.015 · R 0.26 → 0.001 · K 0.23 → 0.017 · W 0.24 → 0.019.

## In the browser

The same positions are stored in `poses.json` (`raw` and `touched`). The **Contact alignment** switch on the avatar page toggles between them, and **Show contact markers** draws the aligned contact points while a letter is held. During transitions the app blends joint positions and then re-projects every bone to its length (forward kinematics), so fingers never shrink or stretch between letters.
