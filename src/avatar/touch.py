"""
src/avatar/touch.py
===================
Geometric transformations that make the "joining points" of a two-hand letter TOUCH.

A two-handed ISL letter is defined by contacts such as "right index tip rests on the left
palm". MediaPipe gives only noisy joint centres, so a raw pose shows small gaps or
interpenetration. We repair it in four steps (all in hand-size units, hand-size = wrist ->
middle-finger MCP distance):

  0. register_depth     MediaPipe's z is relative to EACH hand's own wrist, so the depth
                        offset between the two hands is unknown; slide hand B along z until
                        its contact joints sit at the same depth as hand A's
  1. regularise_bones   rebuild every finger by forward kinematics with canonical bone
                        lengths (kills the stretchy-finger jitter of raw landmarks)
  2. rigid_align        Kabsch / Procrustes: rotate + translate the SECOND hand as a rigid
                        body so that all contact joints reach their target distances
                        (target = sum of the two joint radii = skin-to-skin contact)
  3. fabrik_reach       FABRIK inverse kinematics on the finger chains that still miss
                        their target, root joint fixed, bone lengths preserved
  4. report             contact gap before / after (|distance - target|)

The same maths runs in the browser (app/static/js/avatar.js) when the user toggles
"contact alignment" on the 3D avatar.
"""

import numpy as np

# MediaPipe joint tree
PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
CHAINS = {  # root -> tip joint indices (wrist excluded: it is the root of the hand)
    "thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
# joint radii (hand-size units) -> skin-to-skin contact distance = r_i + r_j
RADIUS = np.array([0.165,
                   0.130, 0.115, 0.100, 0.085,
                   0.115, 0.100, 0.090, 0.080,
                   0.115, 0.100, 0.090, 0.080,
                   0.110, 0.095, 0.085, 0.075,
                   0.100, 0.085, 0.075, 0.068], np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def canonical_bone_lengths(hands: np.ndarray) -> np.ndarray:
    """median bone length per joint over many hands (N,21,3) in hand-size units."""
    L = np.zeros((len(hands), 21), np.float32)
    for j in range(1, 21):
        L[:, j] = np.linalg.norm(hands[:, j] - hands[:, PARENT[j]], axis=1)
    return np.median(L, axis=0)


def regularise_bones(P: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """FK rebuild: keep wrist + bone directions, replace bone lengths by `lengths`."""
    Q = P.copy()
    for j in range(1, 21):
        d = P[j] - P[PARENT[j]]
        n = np.linalg.norm(d)
        d = d / n if n > 1e-8 else np.array([0, 1, 0], np.float32)
        Q[j] = Q[PARENT[j]] + d * lengths[j]
    return Q


def kabsch(src: np.ndarray, dst: np.ndarray, w: np.ndarray = None):
    """weighted rigid transform (R, t) minimising sum w |R src + t - dst|^2."""
    w = np.ones(len(src)) if w is None else w
    w = w / w.sum()
    cs, cd = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    H = ((src - cs) * w[:, None]).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def _limit_rotation(R: np.ndarray, max_deg: float) -> np.ndarray:
    """clamp a rotation matrix to at most max_deg (axis-angle scaling)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang <= np.radians(max_deg) or ang < 1e-6:
        return R
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= np.linalg.norm(axis) + 1e-12
    a = np.radians(max_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def register_depth(A: np.ndarray, B: np.ndarray, pairs) -> np.ndarray:
    """translate B along z so the mean depth of its contact joints equals hand A's."""
    if not pairs:
        return B.copy()
    dz = np.mean([A[i, 2] - B[j, 2] for i, j in pairs])
    B2 = B.copy()
    B2[:, 2] += dz
    return B2


def contact_targets(A, B, pairs):
    """[(i, j, target)] with target = r_i + r_j."""
    return [(i, j, float(RADIUS[i] + RADIUS[j])) for i, j in pairs]


def rigid_align(A: np.ndarray, B: np.ndarray, pairs, max_rot_deg: float = 20.0, anchor_w: float = 0.15):
    """
    Move hand B (rigidly) so contact joints touch hand A's.
    desired position of B_j: A_i + target * unit(B_j - A_i)   (keeps the approach direction).
    A few wrist/MCP anchors (small weight) keep B close to its original pose, and the
    rotation is clamped to `max_rot_deg` so the hand never twists unnaturally.
    """
    if not pairs:
        return B.copy()
    B2 = B.copy()
    for _ in range(3):                                    # fixed-point iterations
        src, dst, w = [], [], []
        for i, j, tgt in contact_targets(A, B2, pairs):
            d = B2[j] - A[i]
            n = np.linalg.norm(d)
            u = d / n if n > 1e-8 else np.array([0, 0, 1.0])
            src.append(B2[j]); dst.append(A[i] + tgt * u); w.append(1.0)
        for a in (0, 5, 9, 13, 17):                       # anchors: stay where they are
            src.append(B2[a]); dst.append(B2[a]); w.append(anchor_w)
        src, dst, w = np.array(src), np.array(dst), np.array(w)
        R, _ = kabsch(src, dst, w)
        R = _limit_rotation(R, max_rot_deg)
        wn = w / w.sum()
        t = (wn[:, None] * dst).sum(0) - R @ (wn[:, None] * src).sum(0)   # translation for the clamped R
        B2 = B2 @ R.T + t
    return B2


def fabrik_reach(P: np.ndarray, joint: int, target: np.ndarray, lengths: np.ndarray, iters: int = 16):
    """
    FABRIK inverse kinematics: move `joint` to `target` along its finger chain
    (base joint fixed, bone lengths preserved). Joints distal to `joint` follow it.
    """
    chain, j = [], joint
    while j > 0 and PARENT[j] != 0:                      # climb to the finger base (parent == wrist)
        chain.append(j)
        j = PARENT[j]
    chain.append(j)
    chain = chain[::-1]                                   # base -> ... -> joint
    if len(chain) < 2:
        return P
    pts = P[chain].copy()
    seg = [lengths[chain[k + 1]] for k in range(len(chain) - 1)]
    root = pts[0].copy()
    if np.linalg.norm(target - root) > sum(seg):          # unreachable: stretch straight toward it
        d = (target - root) / np.linalg.norm(target - root)
        for k in range(1, len(chain)):
            pts[k] = pts[k - 1] + d * seg[k - 1]
    else:
        for _ in range(iters):
            pts[-1] = target
            for k in range(len(chain) - 2, -1, -1):       # backward pass
                d = pts[k] - pts[k + 1]; d /= np.linalg.norm(d) + 1e-12
                pts[k] = pts[k + 1] + d * seg[k]
            pts[0] = root
            for k in range(1, len(chain)):                # forward pass
                d = pts[k] - pts[k - 1]; d /= np.linalg.norm(d) + 1e-12
                pts[k] = pts[k - 1] + d * seg[k - 1]
            if np.linalg.norm(pts[-1] - target) < 1e-4:
                break
    Q = P.copy()
    Q[chain] = pts
    shift = Q[joint] - P[joint]
    for k in range(1, 21):                                # distal joints ride along
        a = k
        while a > 0 and a != joint:
            a = PARENT[a]
        if a == joint and k != joint:
            Q[k] = P[k] + shift
    return Q


def make_touch(A: np.ndarray, B: np.ndarray, pairs, lengths: np.ndarray, tol: float = 0.01, rounds: int = 4):
    """
    Full contact repair for hands A (fixed) and B (moved): depth registration, then up to
    `rounds` x [rigid_align + FABRIK on every pair still more than `tol` from its target].
    The round with the smallest mean |gap| wins. Returns (A, B2).
    """
    B2 = register_depth(A, B, pairs)
    best, best_err = B2.copy(), np.mean(np.abs(gaps(A, B2, pairs, register=False))) if pairs else 0.0
    for _ in range(rounds):
        B2 = rigid_align(A, B2, pairs)
        for i, j, tgt in contact_targets(A, B2, pairs):
            d = B2[j] - A[i]
            n = np.linalg.norm(d)
            if abs(n - tgt) > tol:
                u = d / n if n > 1e-8 else np.array([0, 0, 1.0])
                B2 = fabrik_reach(B2, j, A[i] + tgt * u, lengths)
        err = np.mean(np.abs(gaps(A, B2, pairs, register=False))) if pairs else 0.0
        if err < best_err:
            best, best_err = B2.copy(), err
        if best_err < tol:
            break
    return A, best


def gaps(A, B, pairs, register: bool = True):
    """
    signed gap (distance - target) per contact pair; 0 = touching, <0 = interpenetrating.
    With register=True the unknown inter-hand depth offset is first removed, so the number
    reflects the in-plane + residual error (this is how raw poses are scored fairly).
    """
    if register:
        B = register_depth(A, B, pairs)
    return [float(np.linalg.norm(B[j] - A[i]) - (RADIUS[i] + RADIUS[j])) for i, j in pairs]
