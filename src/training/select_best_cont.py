"""
src/training/select_best_cont.py
================================
Pick the continuous model with the lowest VALIDATION CER among checkpoints/cont_v2_*.pt,
copy it to checkpoints/cont_best.pt (what the app loads) and print a comparison table.

  python src/training/select_best_cont.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default=None, help="deploy this run instead of the lowest-validation-CER one "
                                            "(used for conformer_stream: chosen on the LIVE benchmark, see eval_streaming.py)")
a = ap.parse_args()
rows = []
for f in sorted((ROOT / "results").glob("cont_v2_*.json")):
    r = json.loads(f.read_text())
    rows.append(r)
if not rows:
    sys.exit("no results/cont_v2_*.json found - train first")
rows.sort(key=lambda r: r["val_cer"])
print(f"{'model':<18}{'features':<10}{'params':>10}{'val CER':>10}{'test CER':>10}")
for r in rows:
    print(f"{r['kind']:<18}{r['parts']:<10}{r['params']:>10,}{100*r['val_cer']:>9.2f}%{100*r['test_cer']:>9.2f}%")
best = next(r for r in rows if r["tag"] == a.tag) if a.tag else rows[0]
shutil.copy(ROOT / "checkpoints" / f"cont_v2_{best['tag']}.pt", ROOT / "checkpoints" / "cont_best.pt")
print(f"\n-> checkpoints/cont_best.pt  =  {best['tag']}  (val {100*best['val_cer']:.2f}%, test {100*best['test_cer']:.2f}%)")
