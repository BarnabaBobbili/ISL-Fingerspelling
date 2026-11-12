"""Contact sheet of the authored letters (skeleton view) for quick checking.  python -m src.avatar.preview_spec out.png [LETTERS]"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.avatar import spec_letters as sl  # noqa: E402
from src.avatar.handmodel import PARENT  # noqa: E402

COL = {0: "#c0392b", 1: "#1f6f78"}


def draw(ax, hands, view):
    for h, P in enumerate(hands):
        if P is None:
            continue
        for j in range(1, 21):
            p, q = P[PARENT[j]], P[j]
            ax.plot(*zip(*[_proj(p, view), _proj(q, view)]), color=COL[h], lw=1.6)
        ax.scatter(*zip(*[_proj(p, view) for p in P[[4, 8, 12, 16, 20]]]), color=COL[h], s=8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def _proj(p, view):
    x, y, z = p
    if view == "front":
        return (x, y)
    if view == "above":
        return (x, -z)
    return (z, y)             # side


def main():
    out = sys.argv[1]
    letters = sys.argv[2] if len(sys.argv) > 2 else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = len(letters)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 3 * 1.5, rows * 1.9))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        for v, view in enumerate(("front", "above", "side")):
            ax = axes[k // cols, (k % cols) * 3 + v]
            ax.axis("off")
        if k >= n:
            continue
        L = letters[k]
        try:
            r = sl.author(L)
        except Exception as e:  # noqa: BLE001
            print(L, "ERR", e)
            continue
        hands = r["keys"][len(r["keys"]) // 2] if r["keys"] else r["hands"]
        for v, view in enumerate(("front", "above", "side")):
            ax = axes[k // cols, (k % cols) * 3 + v]
            ax.axis("on")
            draw(ax, hands, view)
            if v == 0:
                ax.set_title(L, fontsize=11, loc="left", fontweight="bold")
    plt.tight_layout(pad=0.2)
    fig.savefig(out, dpi=95)


if __name__ == "__main__":
    main()
