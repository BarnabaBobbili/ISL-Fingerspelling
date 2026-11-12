"""
src/avatar/build_spec_poses.py
==============================
Replaces the detected pose of the chosen letters in app/static/data/poses.json by the authored pose from
src/avatar/spec_letters.py (hand shapes, palm directions and contacts written down, not measured).

  python -m src.avatar.build_spec_poses --letters GHJMPQRS      # only these letters
  python -m src.avatar.build_spec_poses --all                   # all 26

Every replaced letter gets "source": "spec".  Strokes (H, J) carry "keys": the poses the right hand passes through.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.avatar import spec_letters as sl  # noqa: E402
from src.avatar.handmodel import RADIUS, handedness  # noqa: E402

OUT = ROOT / "app" / "static" / "data" / "poses.json"


def r4(P):
    return np.round(np.asarray(P), 4).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if a.all else a.letters.upper()
    d = json.loads(OUT.read_text())
    for L in letters:
        r = sl.author(L)
        A, B = r["hands"]
        two = B is not None
        hands = [A, B] if two else [A]
        for h, c in zip(hands, r["chir"]):
            assert handedness(h) == c, (L, "handedness", handedness(h), c)
        contacts = []
        for c in r["contacts"]:
            if "pt" in c:
                contacts.append({"a": c["a"], "b": c["b"], "pt": r4(c["pt"]), "target": 0.0, "gap_raw": 0.0, "gap_touched": 0.0})
            else:
                gap = float(np.linalg.norm(A[c["a"]] - B[c["b"]]) - (RADIUS[c["a"]] + RADIUS[c["b"]]))
                contacts.append({"a": c["a"], "b": c["b"], "target": round(float(RADIUS[c["a"]] + RADIUS[c["b"]]), 4),
                                 "gap_raw": round(gap, 4), "gap_touched": round(gap, 4)})
        e = {"hands": 2 if two else 1, "raw": [r4(h) for h in hands], "touched": [r4(h) for h in hands],
             "contacts": contacts, "n_candidates": 0, "chir": r["chir"], "source": "spec"}
        if r["keys"]:
            e["keys"] = [[r4(x) for x in k] for k in r["keys"]]
        d["letters"][L] = e
        print(L, "ok", "keys" if r["keys"] else "", [c["gap_touched"] for c in contacts])
    OUT.write_text(json.dumps(d, separators=(",", ":")))
    print("saved", OUT)


if __name__ == "__main__":
    main()
