"""
src/avatar/handmodel.py
=======================
A small articulated hand model that produces MediaPipe-style landmarks (21 x 3) from joint angles.

Why: landmarks detected in photographs are unreliable where the two hands overlap, which is exactly where
most ISL letters are formed. This module lets a letter be *authored* from its description (which fingers
are curled, where the palm faces, which fingertip touches what) and still use the same 21-joint format, so
the renderer, the contact geometry and the pose file stay unchanged.

Local palm frame of a RIGHT hand (hand-size units, wrist at the origin):
    +y  from the wrist toward the middle-finger knuckle (finger direction)
    +z  palm normal (the side the fingers curl toward)
    +x  toward the thumb
A left hand is the mirror image (x -> -x). With that convention the triple product used by
build_poses.handedness_sign() is +1 for a right hand and -1 for a left hand.

World frame (Three.js): x right, y up, z toward the viewer. The signer faces the viewer, so the signer's
right hand is on the viewer's left (-x) and the signer's left hand on the viewer's right (+x).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
_META = json.loads((ROOT / "app" / "static" / "data" / "poses.json").read_text())["meta"]
BONE = np.array(_META["bone_lengths"], np.float64)         # BONE[j] = length of the bone parent(j) -> j
PARENT = _META["parent"]
RADIUS = np.array(_META["radius"], np.float64)

FINGER_BASES = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
# palm template (median of the real poses, right-hand frame)
MCP = {"index": (0.17, 1.04, 0.0), "middle": (0.0, 1.0, 0.0), "ring": (-0.2, 0.90, 0.0), "pinky": (-0.40, 0.78, 0.0)}
CMC = np.array([0.20, 0.29, 0.21])
CMC = CMC / np.linalg.norm(CMC) * BONE[1]

_unit = lambda v: np.asarray(v, np.float64) / (np.linalg.norm(v) + 1e-12)

# finger shapes: (spread, MCP flexion, PIP flexion, DIP flexion) in degrees; flexion curls toward the palm (+z)
EXT = (0, 0, 0, 0)
CURL = (0, 88, 100, 55)          # fist
HOOK = (0, 8, 88, 40)            # hooked finger
HALF = (0, 45, 55, 25)           # relaxed, half curled
CLAW = (0, 25, 45, 20)

# thumb directions (unit vectors of the three thumb segments, right-hand frame)
THUMB_OUT = [(0.62, 0.73, 0.31), (0.74, 0.46, 0.48), (0.73, 0.23, 0.64)]
THUMB_UP = [(0.15, 0.97, 0.20), (0.10, 0.99, 0.10), (0.05, 1.00, 0.05)]   # thumb along the index finger
THUMB_BACK = [(0.55, 0.80, -0.20), (0.75, 0.55, -0.35), (0.85, 0.35, -0.40)]  # out to the thumb side, behind the palm plane


def _finger(name, spec):
    """positions of MCP, PIP, DIP, TIP for a finger given (spread, f1, f2, f3) degrees."""
    spread, f1, f2, f3 = spec[:4]
    psi = np.radians(spec[4]) if len(spec) > 4 else 0.0       # direction of the curl: 0 = toward the palm, 90 = toward the thumb side
    base = 5 + 4 * list(FINGER_BASES).index(name)
    m = np.array(MCP[name], np.float64)
    a = np.radians(spread)
    ax = np.array([np.sin(a), np.cos(a), 0.0])
    z = np.array([np.sin(psi), 0.0, np.cos(psi)])
    pts = [m]
    phi = 0.0
    for k, f in enumerate((f1, f2, f3)):
        phi += np.radians(f)
        d = np.cos(phi) * ax + np.sin(phi) * z
        pts.append(pts[-1] + d * BONE[base + 1 + k])
    return pts


def fabrik(chain, target, lengths, iters=40):
    """FABRIK on a free chain (root fixed). chain: list of 3-vectors."""
    pts = [np.array(p, np.float64) for p in chain]
    root = pts[0].copy()
    if np.linalg.norm(target - root) >= sum(lengths):
        d = _unit(target - root)
        for k in range(1, len(pts)):
            pts[k] = pts[k - 1] + d * lengths[k - 1]
        return pts
    for _ in range(iters):
        pts[-1] = np.array(target, np.float64)
        for k in range(len(pts) - 2, -1, -1):
            pts[k] = pts[k + 1] + _unit(pts[k] - pts[k + 1]) * lengths[k]
        pts[0] = root
        for k in range(1, len(pts)):
            pts[k] = pts[k - 1] + _unit(pts[k] - pts[k - 1]) * lengths[k - 1]
        if np.linalg.norm(pts[-1] - target) < 1e-5:
            break
    return pts


def hand_local(fingers=None, thumb=THUMB_OUT, thumb_tip=None, spread=None):
    """
    Right-hand landmarks (21, 3) in the local palm frame.
      fingers   dict name -> (spread, f1, f2, f3) or a preset; missing fingers are extended
      thumb     three unit direction vectors for the thumb segments (used unless thumb_tip is given)
      thumb_tip a point in the local frame the thumb tip must reach (FABRIK, CMC fixed)
    """
    fingers = dict(fingers or {})
    if spread:                                            # a default fan for extended fingers
        for n, s in zip(("index", "middle", "ring", "pinky"), spread):
            fingers.setdefault(n, (s, 0, 0, 0))
    P = np.zeros((21, 3))
    for n in FINGER_BASES:
        spec = fingers.get(n, EXT)
        pts = _finger(n, spec)
        b = FINGER_BASES[n]
        for k in range(4):
            P[b + k] = pts[k]
    P[1] = CMC
    if thumb_tip is not None:
        d0 = [_unit(v) for v in THUMB_OUT]
        chain = [P[1]]
        for k in range(3):
            chain.append(chain[-1] + d0[k] * BONE[2 + k])
        chain = fabrik(chain, np.asarray(thumb_tip, np.float64), [BONE[2], BONE[3], BONE[4]])
        for k in range(1, 4):
            P[1 + k] = chain[k]
    else:
        for k in range(3):
            P[2 + k] = P[1 + k] + _unit(thumb[k]) * BONE[2 + k]
    return P


def basis(d, n):
    """world axes of the local frame from the finger direction d and the palm normal n (n need not be orthogonal)."""
    y = _unit(d)
    z = np.asarray(n, np.float64)
    z = _unit(z - (z @ y) * y)
    x = np.cross(y, z)
    return x, y, z


def to_world(Q, d, n, wrist, left=False):
    """place local landmarks: the palm normal points along n, the fingers along d, the wrist at `wrist`."""
    Q = Q.copy()
    if left:
        Q[:, 0] *= -1
    x, y, z = basis(d, n)
    return np.asarray(wrist, np.float64) + Q[:, 0:1] * x + Q[:, 1:2] * y + Q[:, 2:3] * z


def handedness(P):
    u, v, t = P[5] - P[0], P[17] - P[0], P[2] - P[0]
    return 1 if float(np.cross(u, v) @ t) > 0 else -1


def reach(P, joint, target):
    """FABRIK a finger/thumb tip (or any joint) onto `target`, base fixed, bone lengths kept."""
    from src.avatar.touch import fabrik_reach
    return fabrik_reach(P.astype(np.float32), joint, np.asarray(target, np.float32), BONE.astype(np.float32)).astype(np.float64)


def place_contact(P, a, Q, b, u, wrist_hint=None):
    """translate hand P so that its joint `a` touches hand Q's joint `b` (skin radii), approaching along unit u
    (u points from Q's joint toward P's joint)."""
    tgt = Q[b] + _unit(u) * (RADIUS[a] + RADIUS[b])
    return P + (tgt - P[a])
