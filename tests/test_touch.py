"""
tests/test_touch.py
===================
Checks of the contact-geometry code (src/avatar/touch.py) and of the shipped pose library.

  python tests/test_touch.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.avatar import touch  # noqa: E402


def test_kabsch_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(12, 3))
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    t = np.array([0.3, -1.2, 0.5])
    Rh, th = touch.kabsch(P, P @ R.T + t)
    assert np.allclose(Rh, R, atol=1e-8) and np.allclose(th, t, atol=1e-8)


def test_fabrik_reaches_target_and_keeps_bone_lengths():
    lengths = np.zeros(21)
    P = np.zeros((21, 3))
    for j in range(5, 9):                                  # index finger, straight along +y
        lengths[j] = 0.45
        P[j] = P[touch.PARENT[j]] + np.array([0, 0.45, 0]) if j > 5 else np.array([0.0, 0.9, 0.0])
    target = P[8] + np.array([0.25, -0.1, 0.1])
    Q = touch.fabrik_reach(P, 8, target, lengths)
    assert np.linalg.norm(Q[8] - target) < 1e-2
    for j in (6, 7, 8):
        assert abs(np.linalg.norm(Q[j] - Q[j - 1]) - lengths[j]) < 1e-3


def test_regularise_bones_sets_canonical_lengths():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(21, 3))
    L = np.linspace(0.1, 0.5, 21)
    Q = touch.regularise_bones(P, L)
    for j in range(1, 21):
        assert abs(np.linalg.norm(Q[j] - Q[touch.PARENT[j]]) - L[j]) < 1e-6


def test_pose_library():
    d = json.loads((ROOT / "app" / "static" / "data" / "poses.json").read_text())
    L = d["letters"]
    assert all(chr(65 + i) in L for i in range(26)) and "_rest" in L
    assert {k for k in L if k != "_rest" and L[k]["hands"] == 1} == set(d["meta"]["one_handed"])
    improved = n = 0
    for k, v in L.items():
        if not v["contacts"]:
            continue
        raw = np.mean([abs(c["gap_raw"]) for c in v["contacts"]])
        new = np.mean([abs(c["gap_touched"]) for c in v["contacts"]])
        assert new < 0.06, f"{k}: contact gap after alignment {new:.3f} too large"
        if v.get("source") == "spec":        # authored poses are built touching: there is no detected 'raw' pose to improve on
            continue
        n += 1
        improved += new < raw
    spec = sum(1 for k, v in L.items() if k != "_rest" and v.get("source") == "spec")
    assert improved == n and (n >= 8 or spec >= 18), "alignment must reduce the gap of every detected letter with contacts"
    print(f"pose library ok: {n} letters with contacts, all improved")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("ALL PASSED")
