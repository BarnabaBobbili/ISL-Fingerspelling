"""
src/avatar/spec_letters.py
==========================
The two-handed manual alphabet written as data: for every letter, how each hand is shaped, where the palm
faces and which parts touch (see docs 'Pose library').  The base hand is the signer's LEFT hand, the active
hand the signer's RIGHT hand.  The letters are written with the signer facing the viewer and then turned half way
round, so the viewer sees them as the signer does: left hand on the left of the screen, right hand on the right.

author(letter) -> dict(
    hands   : [left (21,3), right (21,3)]   slot 0 = left hand (left of the screen), slot 1 = right hand; one hand: [right]
    chir    : [-1, +1]
    contacts: list of dict(a, b, pt)         markers: joint a of slot 0 / joint b of slot 1, or a fixed point
    keys    : None or list of [left, right] for strokes (H, J)
)
"""

import numpy as np

from src.avatar import handmodel as hm
from src.avatar.handmodel import CURL, EXT, HALF, HOOK, THUMB_OUT, hand_local, place_contact, reach, to_world, RADIUS, _unit

import os
YAW = np.radians(float(os.environ.get("SPEC_YAW", "208")))   # the letters are written facing the viewer; the whole figure is turned about the vertical axis
                                                             # (180 deg = the signer's own view from behind, the left hand on the left of the screen; 208 = turned 28 deg further;
                                                             #  a few degrees more or less shows the palms and contacts better)
CENTER = np.array([0.1, 0.0, 0.0])

BW = np.array([1.0, -0.9, 0.0])                   # left (base) wrist
V = lambda *a: np.array(a, np.float64)
UP, DOWN, FWD, BACK, RIGHT_X, LEFT_X = V(0, 1, 0), V(0, -1, 0), V(0, 0, 1), V(0, 0, -1), V(-1, 0, 0), V(1, 0, 0)

FAN = (10, 3, -4, -12)                            # spread of extended fingers (index .. little)


# ---------------------------------------------------------------- shapes
def shape(kind, **kw):
    f = {}
    thumb = dict(thumb=THUMB_OUT)
    if kind == "open":
        return hand_local(spread=FAN, thumb=THUMB_OUT)
    if kind == "flat":
        return hand_local(spread=(1, 0, -1, -2), thumb=THUMB_OUT)
    if kind == "flat_in":        # flat hand, fingers together, thumb alongside the index
        return hand_local(spread=(0, 0, 0, 0), thumb=[(0.35, 0.93, 0.12), (0.2, 0.97, 0.10), (0.10, 0.99, 0.05)])
    if kind == "fist":
        f = dict(index=CURL, middle=CURL, ring=CURL, pinky=CURL)
        return hand_local(f, thumb_tip=(0.10, 0.80, 0.64))
    if kind == "point":          # index out, the rest curled, thumb over the fist
        f = dict(middle=CURL, ring=CURL, pinky=CURL)
        return hand_local(f, thumb_tip=(0.10, 0.80, 0.64))
    if kind == "two":            # index + middle out
        f = dict(ring=CURL, pinky=CURL)
        return hand_local(f, thumb_tip=(-0.05, 0.72, 0.60))
    if kind == "three":
        f = dict(pinky=CURL)
        return hand_local(f, thumb_tip=(-0.30, 0.62, 0.52))
    if kind == "vee":
        f = dict(index=(14, 0, 0, 0), middle=(-14, 0, 0, 0), ring=CURL, pinky=CURL)
        return hand_local(f, thumb_tip=(-0.05, 0.72, 0.60))
    if kind == "hook":           # hooked index, rest curled
        f = dict(index=HOOK, middle=CURL, ring=CURL, pinky=CURL)
        return hand_local(f, thumb_tip=(0.10, 0.80, 0.64))
    if kind == "little":         # little finger out
        f = dict(index=CURL, middle=CURL, ring=CURL)
        return hand_local(f, thumb_tip=(0.12, 0.80, 0.64))
    if kind == "little_hook":
        f = dict(index=CURL, middle=CURL, ring=CURL, pinky=(0, 10, 85, 40))
        return hand_local(f, thumb_tip=(0.12, 0.80, 0.64))
    if kind in ("ring", "ring_closed"):   # thumb and index make an O (the index curls toward the thumb side)
        rest = CURL if kind == "ring_closed" else HALF
        f = dict(index=(0, 38, 66, 40, 78), middle=rest, ring=rest, pinky=rest)
        Q = hand_local(f, thumb=THUMB_OUT)
        return hand_local(f, thumb_tip=Q[8])
    if kind == "vee_thumb":      # thumb out + index up (left hand of Y)
        f = dict(middle=CURL, ring=CURL, pinky=CURL)
        return hand_local(f, thumb=[(0.85, 0.50, 0.15), (0.92, 0.35, 0.15), (0.95, 0.25, 0.12)])
    if kind == "cee":
        f = {n: (0, 42, 46, 32) for n in ("index", "middle", "ring", "pinky")}
        return hand_local(f, thumb_tip=(0.10, 0.62, 0.95))
    raise KeyError(kind)


def R_(Q, d, n, wrist):
    return to_world(Q, d, n, wrist, left=False)


def L_(Q, d, n, wrist):
    return to_world(Q, d, n, wrist, left=True)


def palm_point(P, n, off=0.0):
    """centre of the palm of hand P (a point on the palm surface), n = palm normal in world."""
    c = (P[0] + P[5] + P[9] + P[13] + P[17]) / 5.0 * 0.0 + (P[0] * 0.3 + P[5] * 0.2 + P[9] * 0.2 + P[13] * 0.15 + P[17] * 0.15)
    return c + _unit(n) * (0.10 + off)


def rotate_y(P, ang, center=CENTER):
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (P - center) @ R.T + center


def slide_until_touch(A, B, axis, ja, jb):
    """move A along unit `axis` until the closest pair (a in ja, b in jb) is exactly skin-to-skin."""
    axis = _unit(axis)
    def gap(t):
        return min(np.linalg.norm(A[a] + axis * t - B[b]) - (RADIUS[a] + RADIUS[b]) for a in ja for b in jb)
    t = -3.0
    while t < 6.0 and gap(t) > 0:           # first contact when moving along the axis
        t += 0.05
    lo, hi = t - 0.05, t
    for _ in range(40):
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return A + axis * lo


def closest_pair(A, B, ja, jb):
    return min(((a, b) for a in ja for b in jb), key=lambda ab: np.linalg.norm(A[ab[0]] - B[ab[1]]) - (RADIUS[ab[0]] + RADIUS[ab[1]]))


def ring_pose(side, wrist, d=UP, n=FWD, kind="ring"):
    Q = shape(kind)
    return (R_ if side == "R" else L_)(Q, d, n, wrist)


def FD(*v):
    """a direction given as the viewer sees the finished figure (x right, y up, z toward the viewer), as a direction in the authoring frame"""
    return rotate_y(V(*v), -YAW, center=np.zeros(3))


def FP(*v):
    """the same for a point"""
    return rotate_y(V(*v), -YAW)


NB = _unit(FD(0.7, 0, 0.7))          # the base palm faces the active hand and is turned toward the viewer, so its fingers fan out on screen


# ---------------------------------------------------------------- the letters
def _base(kind, d=UP, n=None, wrist=BW):
    return L_(shape(kind), d, NB if n is None else n, wrist)


def _vowel(b_joint):
    B = _base("open")
    A = R_(shape("point"), -NB, DOWN, V(0, 0, 0))                             # pointing toward the base hand, palm down
    A = place_contact(A, 8, B, b_joint, NB)
    return A, B, [dict(a=8, b=b_joint)]


def _flat_press(kind, tips, hook=False, palm_to_viewer=False):
    """flat base; the right fingers point at the left palm and touch it with `tips` (joint ids)."""
    B = _base("flat")
    n = NB                                         # left palm normal
    pc = palm_point(B, n)
    A = R_(shape(kind), -NB, FD(0, 0, 1) if palm_to_viewer else DOWN, V(0, 0, 0))
    mid = tips[len(tips) // 2]
    A = place_contact(A, mid, B, 9, NB)       # first touch the palm region around the middle MCP
    # slide to the palm centre
    A = A + (pc + n * RADIUS[mid] - A[mid])
    return A, B, [dict(a=mid, b=-1, pt=(pc + n * RADIUS[mid] * 0.5).tolist())]


def author(L):
    contacts, keys = [], None
    if L in "AEIOU":
        A, B, contacts = _vowel({"A": 4, "E": 8, "I": 12, "O": 16, "U": 20}[L])
    elif L == "B":
        A = ring_pose("R", V(-1.2, -0.9, 0)); B = ring_pose("L", V(1.2, -0.9, 0))
        A = slide_until_touch(A, B, RIGHT_X * -1, (2, 3, 4, 6, 7, 8), (2, 3, 4, 6, 7, 8))   # toward the other ring until they touch
        a_, b_ = closest_pair(A, B, (2, 3, 4, 6, 7, 8), (2, 3, 4, 6, 7, 8)); contacts = [dict(a=a_, b=b_)]
    elif L == "C":
        A = R_(shape("cee"), UP, LEFT_X, V(0, 0, 0)); B = None
    elif L == "D":
        # left index up (the stem of the D); the right hand is the bowl: its index curves up and over to touch the tip of the stem,
        # its thumb touches the base of the stem, so the two fingers and the stem close the loop
        B = L_(shape("point"), UP, NB, BW)
        Q = hand_local(dict(index=(0, 22, 52, 38), middle=CURL, ring=CURL, pinky=CURL), thumb=THUMB_OUT)
        A = R_(Q, UP, -NB, V(0, 0, 0))
        A = place_contact(A, 8, B, 8, NB)
        A = reach(A, 4, B[5] + NB * (RADIUS[4] + RADIUS[5]))
        contacts = [dict(a=8, b=8), dict(a=4, b=5)]
    elif L == "F":
        B = L_(shape("two"), UP, NB, BW)
        A = R_(shape("two"), -NB, DOWN, V(0, 0, 0))
        mid = (B[6] + B[10]) / 2
        A = A + (mid + NB * 0.2 - (A[6] + A[10]) / 2)
        contacts = []
    elif L == "G":
        B = L_(shape("fist"), V(-0.5, 0.5, 0.7), V(0.7, 0.1, 0.7), V(0.3, -1.45, -0.6))
        A = R_(shape("fist"), V(0.5, 0.5, 0.7), V(-0.7, 0.1, 0.7), V(-0.1, -0.45, -0.6))
        contacts = []
    elif L == "H":
        # as the signer sees it: the left hand lies palm up, fingers to the right, in front (nearer the viewer);
        # the flat right hand slides over it from the heel to the fingertips
        B = L_(shape("flat"), FD(1, 0, 0), FD(0, 1, 0), FP(-1.4, -1.15, 0.35))
        Q = shape("flat_in")
        keys = []
        for t in np.linspace(0, 1, 9):
            A0 = R_(Q, FD(1, 0, 0), FD(0, -1, 0), FP(-1.4 + 2.9 * t, -0.93, -0.25))
            keys.append([A0, B])
        A = keys[0][0]
        contacts = []
    elif L == "J":
        # left hand up, palm toward the viewer and a little to the right; the right hand (palm outward) comes in from the right
        # and its index draws the J: from the tip of the middle finger down the palm and up to the thumb tip
        B = L_(shape("open"), FD(0.1, 1, 0), FD(0.5, 0, 0.86), FP(-1.1, -0.9, 0.0))
        Q = shape("point")
        nB = _unit(FD(0.5, 0, 0.86))
        path = [B[12], B[11] * 0.7 + B[9] * 0.3, B[10], B[9], B[0] * 0.2 + B[9] * 0.8, B[0] * 0.5 + B[1] * 0.5 + FD(0.1, 0, 0),
                B[1] * 0.6 + B[2] * 0.4, B[3], B[4]]
        keys = []
        for p in path:
            A = R_(Q, FD(-0.8, 0.3, -0.3), FD(0.1, -0.2, -0.95), V(0, 0, 0))
            A = A + (p + nB * 0.2 - A[8])
            keys.append([A, B])
        A = keys[0][0]
    elif L == "K":
        B = L_(shape("point"), UP, NB, BW)
        A = R_(shape("hook"), -NB, DOWN, V(0, 0, 0))
        A = place_contact(A, 6, B, 6, NB)
        contacts = [dict(a=6, b=6)]
    elif L == "L":
        B = _base("flat")
        A = R_(shape("point"), _unit(np.cross(UP, NB)) * -1 + NB * -0.25, DOWN, V(0, 0, 0))
        pc = palm_point(B, NB)
        A = A + (pc + NB * 0.08 - A[7])
        contacts = []
    elif L in "MN":
        A, B, contacts = _flat_press({"M": "three", "N": "two"}[L], {"M": (8, 12, 16), "N": (8, 12)}[L], palm_to_viewer=True)
    elif L == "V":
        A, B, contacts = _flat_press("vee", (8, 12), palm_to_viewer=True)
    elif L == "P":
        B = L_(shape("point"), UP, RIGHT_X, BW)
        A = ring_pose("R", V(-1.2, -0.9, 0.2))
        ctr = A[[3, 4, 7, 8]].mean(0)
        A = A + V(0, B[8][1] - ctr[1] - 0.05, B[8][2] - ctr[2])
        A = slide_until_touch(A, B, LEFT_X, (2, 3, 4, 6, 7, 8), (8,))
        a_, _ = closest_pair(A, B, (2, 3, 4, 6, 7, 8), (8,)); contacts = [dict(a=a_, b=8)]
    elif L == "Q":
        # left hand: a ring (thumb + index) facing the viewer; the right index hooks up into the bottom of the ring (the tail of the Q)
        B = ring_pose("L", V(1.3, -0.9, 0), kind="ring_closed")
        low = min((2, 3, 6, 7), key=lambda j: B[j][1])
        A = R_(shape("hook"), V(1.0, 0.3, 0.0), V(-0.3, 1.0, 0.0), V(0, 0, 0))
        A = place_contact(A, 8, B, low, V(0.0, -1.0, 0.15))
        contacts = [dict(a=8, b=low)]
    elif L == "R":
        A, B, contacts = _flat_press("hook", (8,), palm_to_viewer=True)
    elif L == "S":
        B = L_(shape("little_hook"), UP, RIGHT_X, BW)
        A = R_(shape("little_hook"), UP, LEFT_X, V(0, 0, 0))
        A = place_contact(A, 20, B, 20, V(-1, 0, 0))
        contacts = [dict(a=20, b=20)]
    elif L == "T":
        # left hand open, palm toward the right hand; the right index rests against the lower edge of the left palm / wrist
        B = _base("open")
        A = R_(shape("point"), -NB * 0.85 + UP * 0.5, DOWN, V(0, 0, 0))
        tgt = (B[0] * 0.6 + B[9] * 0.4) + NB * (RADIUS[8] + 0.10)
        A = A + (tgt - A[8])
        contacts = [dict(a=8, b=-1, pt=tgt.tolist())]
    elif L == "W":
        A = R_(shape("open"), V(0.45, 1, 0), LEFT_X, V(-0.7, -0.9, 0.0))
        B = L_(shape("open"), V(-0.45, 1, 0), RIGHT_X, V(0.7, -0.9, 0.1))
        contacts = []
    elif L == "X":
        A = R_(shape("point"), V(0.6, 1, 0), FWD, V(-0.8, -0.9, 0.12))
        B = L_(shape("point"), V(-0.6, 1, 0), FWD, V(0.8, -0.9, -0.12))
        contacts = []
    elif L == "Y":
        # left thumb and index make a V (palm toward the viewer so the V opens in the picture plane); the right index is the stem
        B = L_(shape("vee_thumb"), UP, NB, V(1.1, -0.9, 0))
        A = R_(shape("point"), UP + (-NB) * 0.3, -NB, V(0, 0, 0))
        web = (B[1] + B[5]) / 2 + UP * 0.15 + NB * 0.12
        A = A + (web - A[8])
        contacts = [dict(a=8, b=-1, pt=web.tolist())]
    elif L == "Z":
        B = _base("flat")
        pc = palm_point(B, NB)
        A = R_(shape("flat"), -NB, DOWN, V(0, 0, 0))
        A = A + (pc + NB * 0.08 - A[12])
        contacts = []
    else:
        raise KeyError(L)

    def fin(P):
        return None if P is None else rotate_y(P, YAW)

    if keys is not None:
        keys = [[fin(a), fin(b)] for a, b in keys]
        A, B = keys[0]
    else:
        A, B = fin(A), fin(B)
    for c in contacts:
        if "pt" in c:
            c["pt"] = (rotate_y(np.array([c["pt"]]), YAW)[0]).tolist()
    if B is None:
        return dict(hands=[A], chir=[1], contacts=contacts, keys=None)
    # seen from behind the signer the left (base) hand is on the left of the screen: slot 0 = left hand, slot 1 = right hand
    contacts = [dict(c, a=c["b"], b=c["a"]) for c in contacts]
    if keys is not None:
        keys = [[b, a] for a, b in keys]
    return dict(hands=[B, A], chir=[-1, 1], contacts=contacts, keys=keys)
