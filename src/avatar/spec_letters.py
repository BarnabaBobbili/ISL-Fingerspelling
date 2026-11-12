"""
src/avatar/spec_letters.py
==========================
The two-handed manual alphabet written as data: for every letter, how each hand is shaped, where the palm
faces and which parts touch (see docs 'Pose library').  The base hand is the signer's LEFT hand, the active
hand the signer's RIGHT hand; the signer faces the viewer, so the right hand is on the viewer's left (-x).

author(letter) -> dict(
    hands   : [right (21,3), left (21,3)]   slot 0 = right hand (viewer's left), slot 1 = left hand
    chir    : [+1, -1]
    contacts: list of dict(a, b, pt)         markers: joint a of slot 0 / joint b of slot 1, or a fixed point
    keys    : None or list of [right, left] for strokes (H, J)
)
"""

import numpy as np

from src.avatar import handmodel as hm
from src.avatar.handmodel import CURL, EXT, HALF, HOOK, THUMB_OUT, hand_local, place_contact, reach, to_world, RADIUS, _unit

YAW = np.radians(28.0)        # the whole figure is turned a little toward the viewer so palms and contacts read from the front
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
        f = dict(index=(8, 0, 0, 0), middle=(-8, 0, 0, 0), ring=CURL, pinky=CURL)
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
        return hand_local(f, thumb=THUMB_OUT)
    if kind == "cee":
        f = {n: (0, 30, 35, 25) for n in ("index", "middle", "ring", "pinky")}
        return hand_local(f, thumb_tip=(0.55, 0.95, 0.62))
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


# ---------------------------------------------------------------- the letters
def _base(kind, d=UP, n=RIGHT_X, wrist=BW):
    return L_(shape(kind), d, n, wrist)


def _vowel(b_joint):
    B = _base("open")
    A = R_(shape("point"), RIGHT_X * -1, DOWN, V(0, 0, 0))                    # pointing toward the base hand, palm down
    A = place_contact(A, 8, B, b_joint, RIGHT_X)
    return A, B, [dict(a=8, b=b_joint)]


def _flat_press(kind, tips, hook=False):
    """flat base; the right fingers point at the left palm and touch it with `tips` (joint ids)."""
    B = _base("flat")
    n = RIGHT_X                                    # left palm normal
    pc = palm_point(B, n)
    A = R_(shape(kind), LEFT_X, DOWN, V(0, 0, 0))
    mid = tips[len(tips) // 2]
    A = place_contact(A, mid, B, 9, RIGHT_X)       # first touch the palm region around the middle MCP
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
        # left index up; the right hand reaches up and back from the front: its index touches the tip of the left index,
        # its thumb the base of the left index (the two fingers and the left index close the "D")
        B = L_(shape("point"), UP, RIGHT_X, BW)
        A = R_(shape("point"), V(-0.1, 0.9, -0.5), LEFT_X, V(0, 0, 0))
        A = place_contact(A, 8, B, 8, V(-0.15, 0.0, 1.0))
        A = reach(A, 4, B[5] + V(-0.2, 0.0, 1.0) * (RADIUS[4] + RADIUS[5]))
        contacts = [dict(a=8, b=8), dict(a=4, b=5)]
    elif L == "F":
        B = L_(shape("two"), UP, RIGHT_X, BW)
        A = R_(shape("two"), FWD, DOWN, V(0, 0, 0))
        mid = (B[6] + B[10]) / 2
        A = A + (mid + RIGHT_X * 0.2 - (A[6] + A[10]) / 2)
        contacts = []
    elif L == "G":
        B = L_(shape("fist"), FWD, RIGHT_X, V(0.9, -1.4, -0.6))
        A = R_(shape("fist"), FWD, RIGHT_X, V(0.9, -0.5, -0.6))
        contacts = []
    elif L == "H":
        # left palm up in front, fingers toward the signer's right; the flat right hand slides over it, heel to fingertips
        B = L_(shape("flat"), LEFT_X * -1, UP, V(1.7, -1.1, 0.35))
        Q = shape("flat_in")
        keys = []
        for t in np.linspace(0, 1, 9):
            A0 = R_(Q, LEFT_X * -1, DOWN, V(1.7 - 2.9 * t, -0.90, -0.30))
            keys.append([A0, B])
        A = keys[0][0]
        contacts = []
    elif L == "J":
        B = L_(shape("open"), UP, FWD, V(0.9, -0.9, 0))
        Q = shape("point")
        # the stroke: from the tip of the middle finger down the palm and up to the thumb tip
        path = [B[12], B[11] * 0.7 + B[9] * 0.3, B[10], B[9], B[0] * 0.2 + B[9] * 0.8, B[0] * 0.5 + B[1] * 0.5 + V(-0.1, 0.0, 0),
                B[1] * 0.6 + B[2] * 0.4, B[3], B[4]]
        keys = []
        for p in path:
            A = R_(Q, V(-0.5, 0.4, -0.75), RIGHT_X, V(0, 0, 0))
            A = A + (p + FWD * 0.2 - A[8])
            keys.append([A, B])
        A = keys[0][0]
    elif L == "K":
        B = L_(shape("point"), UP, RIGHT_X, BW)
        A = R_(shape("hook"), FWD, DOWN, V(0, 0, 0))
        A = place_contact(A, 6, B, 6, RIGHT_X)
        contacts = [dict(a=6, b=6)]
    elif L == "L":
        B = _base("flat")
        A = R_(shape("point"), FWD, DOWN, V(0, 0, 0))
        pc = palm_point(B, RIGHT_X)
        A = A + (pc + RIGHT_X * 0.08 - A[7])
        contacts = []
    elif L in "MNV":
        A, B, contacts = _flat_press({"M": "three", "N": "two", "V": "vee"}[L], {"M": (8, 12, 16), "N": (8, 12), "V": (8, 12)}[L])
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
        A, B, contacts = _flat_press("hook", (8,))
    elif L == "S":
        B = L_(shape("little_hook"), UP, RIGHT_X, BW)
        A = R_(shape("little_hook"), UP, LEFT_X, V(0, 0, 0))
        A = place_contact(A, 20, B, 20, V(-1, 0, 0))
        contacts = [dict(a=20, b=20)]
    elif L == "T":
        B = _base("open")
        A = R_(shape("point"), UP, LEFT_X, V(0, 0, 0))
        A = place_contact(A, 8, B, 0, V(-1, 0.0, 0.0))
        contacts = [dict(a=8, b=0)]
    elif L == "W":
        A = R_(shape("open"), V(0.45, 1, 0), LEFT_X, V(-0.7, -0.9, 0.0))
        B = L_(shape("open"), V(-0.45, 1, 0), RIGHT_X, V(0.7, -0.9, 0.1))
        contacts = []
    elif L == "X":
        A = R_(shape("point"), V(0.6, 1, 0), FWD, V(-0.8, -0.9, 0.12))
        B = L_(shape("point"), V(-0.6, 1, 0), FWD, V(0.8, -0.9, -0.12))
        contacts = []
    elif L == "Y":
        B = L_(shape("vee_thumb"), UP, RIGHT_X, BW)
        A = R_(shape("point"), UP, LEFT_X, V(0, 0, 0))
        A = A + ((B[4] + B[8]) / 2 + V(-0.15, 0, 0) - A[8])
        contacts = []
    elif L == "Z":
        B = _base("flat")
        pc = palm_point(B, RIGHT_X)
        A = R_(shape("flat"), LEFT_X, DOWN, V(0, 0, 0))
        A = A + (pc + RIGHT_X * 0.08 - A[12])
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
    return dict(hands=[A, B], chir=[1, -1] if B is not None else [1], contacts=contacts, keys=keys)
