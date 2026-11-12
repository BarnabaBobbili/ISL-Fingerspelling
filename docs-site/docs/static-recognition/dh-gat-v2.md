---
sidebar_position: 1
title: DH-GAT v2 model
---

# DH-GAT v2 — Dual-Hand Graph Attention

DH-GAT classifies a single frame into one of the 26 letters from the two hands' 21 joints each.

## Why v2

The Phase-1 benchmark showed the first model (v1) scored **98 %** on a random split but only **33 %** on real news-video letters. Two causes were identified:

1. **Information was thrown away.** v1 normalised each hand separately, so *"right fingertip touching the left palm"* looked identical to *"right fingertip far from the left hand"*.
2. **Domain gap.** v1 never saw the target domain (small hands, full-body shots) and the near-duplicate random split hid the problem.

## Graph

The two hands form a graph of **42 nodes** (2 × 21 joints) living in one **shared coordinate frame** (origin = mean of the two wrists, unit = mean hand size, the wrist → middle-finger-MCP distance). Layers alternate:

```mermaid
flowchart TB
    F["Node features 11 per joint<br/>+ 4 global features"] --> E["Embedding MLP + joint-id + hand-slot embeddings"]
    E --> L1["Intra-hand attention<br/>(joints within 2 bones — a graph-attention layer on the skeleton)"]
    L1 --> L2["Cross-hand attention<br/>(every present joint attends to every present joint)"]
    L2 --> L3["Intra-hand attention"]
    L3 --> L4["Cross-hand attention"]
    L4 --> P["Masked mean ‖ masked max pooling (192-d)"]
    P --> H["MLP head → 26 logits"]
```

Absent hands are masked out of the attention (a node can always attend to itself), so the same network handles one-handed letters, two-handed letters and detector misses.

### Node features (11 per joint)

| Feature | Dim | Meaning |
|---|---:|---|
| `g` | 3 | position in the shared frame |
| `l` | 3 | position relative to its own wrist, scaled by its own hand size (v1-style) |
| `b` | 3 + 1 | unit vector and length of the bone to the parent joint |
| `m` | 1 | distance to the **nearest joint of the other hand** — a direct "contact" cue |

Global features (4): both-hands flag and the wrist-to-wrist vector in hand sizes.

### Size

2 intra + 2 cross blocks, `d = 96`, 4 heads, feed-forward `4d`, pre-norm, GELU — **502,970 parameters**.

### Hand slotting

Hands are ordered by wrist-x, **not** by MediaPipe's handedness label (which flips when a webcam preview is mirrored). A lone hand is always slot 0. Training augments with horizontal mirroring so the model is robust to left-handed signers and mirrored footage.

## Source

`src/models/dh_gat_v2.py` (`featurize()`, `DHGATv2`), `src/pipeline/classifier.py` (inference wrapper).
