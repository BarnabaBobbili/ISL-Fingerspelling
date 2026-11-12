"""
src/training/build_cont_features.py
===================================
Compute the 351-d per-frame features (src/pipeline/features.py) for every continuous video
from cache/cont_v2/*.npz  ->  cache/cont_feat.pt  { uid: float16 (T, 351) }

  python src/training/build_cont_features.py [--trunk checkpoints/dh_gat_v2_main.pt]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import features as F  # noqa: E402

torch.set_num_threads(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default=None)
    ap.add_argument("--out", default=str(ROOT / "cache" / "cont_feat.pt"))
    a = ap.parse_args()
    trunk, temp = F.load_trunk(a.trunk)
    files = sorted((ROOT / "cache" / "cont_v2").glob("*.npz"))
    feats, t0 = {}, time.time()
    for f in tqdm(files):
        d = np.load(f)
        px = F.to_px(d["lm"], d["wh"])
        feats[f.stem] = torch.from_numpy(F.sequence_features(trunk, temp, px, d["present"])).half()
    torch.save(feats, a.out)
    n = sum(len(v) for v in feats.values())
    print(f"{len(feats)} videos, {n} frames -> {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
