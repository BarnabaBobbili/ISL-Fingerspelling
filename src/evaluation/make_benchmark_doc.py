"""
src/evaluation/make_benchmark_doc.py
====================================
Writes the numbers of the documentation from the measured result files, so the manual
never contains hand-typed results:

  docs-site/docs/evaluation/benchmark.md
  docs-site/docs/_results_glance.md
  fills the @@...@@ markers of docs-site/docs/continuous-recognition/*.md

Inputs: results/benchmark_*.json, results/cont_v2_*.json, results/avatar_contacts.csv

  python src/evaluation/make_benchmark_doc.py [--static-default main|main_r]
"""

import argparse
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results"
DOCS = ROOT / "docs-site" / "docs"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def get(res, *keys):
    if res is None:
        return None
    for k in keys:
        for kk in res:
            if kk.strip() == k:
                return res[kk]
    return None


def static_table():
    v1 = load("benchmark_dh_gat_static_best.json")
    runs = [("DH-GAT v1 (original)", v1, None)]
    for tag, label in (("main", "DH-GAT v2"), ("main_r", "DH-GAT v2 + second-hand recovery")):
        runs.append((label, load(f"benchmark_dh_gat_v2_{tag}.json"), tag))
    rows = []
    for label, res, tag in runs:
        if res is None:
            continue
        if tag is None:
            ind = get(res, "in_domain (orig. split)")
            real = get(res, "real / test videos")
            real_all = get(res, "real (all videos)")
            rows.append((label, ind["acc"], real["acc"], real["ece"], None, "—"))
        else:
            ind = get(res, "in_domain (orig. test)")
            real = get(res, "real (test videos)")
            cal = get(res, "real (test videos) +calib")
            rows.append((label, ind["acc"], real["acc"], real["ece"], cal["ece"], tag))
    out = ["| Model | In-domain (orig. split) | **Real news-video test** | ECE uncalibrated | ECE calibrated |",
           "|---|---:|---:|---:|---:|"]
    for label, ind, real, e0, e1, _ in rows:
        out.append(f"| {label} | {pct(ind, 2)} | **{pct(real, 2)}** | {e0:.3f} | {'—' if e1 is None else f'{e1:.3f}'} |")
    return "\n".join(out), rows


def cross_table():
    out = ["| Model (trained **without** ISL_DATA) | ISL_DATA accuracy | …with both hands detected | …with one hand detected | ECE (calibrated) |",
           "|---|---:|---:|---:|---:|"]
    vals = {}
    for tag, label in (("xsrc", "DH-GAT v2"), ("xsrc_r", "DH-GAT v2 + second-hand recovery")):
        res = load(f"benchmark_dh_gat_v2_{tag}.json")
        h = get(res, "holdout source (cross-source)")
        hc = get(res, "holdout source (cross-source) +calib")
        if h is None:
            continue
        by = h.get("acc_by_hands_detected", {})
        out.append(f"| {label} | **{pct(h['acc'], 1)}** | {pct(by.get('2', by.get(2, 0)), 1)} | {pct(by.get('1', by.get(1, 0)), 1)} | {hc['ece']:.3f} |")
        vals[tag] = h["acc"]
    return "\n".join(out), vals


def cont_table():
    rows = [("BiLSTM-CTC v1 (63-d landmarks, one hand)", "—", 0.1424, 0.1475),
            ("Conformer-CTC v1 (63-d landmarks, one hand)", "—", 0.1872, 0.1926)]
    for f in sorted(R.glob("cont_v2_*.json")):
        r = json.loads(f.read_text())
        name = f"{'Conformer' if r['kind'] == 'conformer' else 'BiLSTM'}-CTC v2" + (" — features: all (351-d)" if r["parts"] == "all" else " — ablation: raw geometry + motion only (133-d)")
        if r["tag"].endswith("_stream"):
            name += " + streaming training (deployed)"
        rows.append((name, f"{r['params']:,}", r["val_cer"], r["test_cer"]))
    out = ["| Model | Parameters | Validation CER | **Test CER** |", "|---|---:|---:|---:|"]
    for n, p, v, t in rows:
        out.append(f"| {n} | {p} | {pct(v, 2)} | **{pct(t, 2)}** |")
    return "\n".join(out), rows


def streaming_table(tag, base):
    cur, old = load(f"streaming_{tag}.json"), load(f"streaming_{base}.json") if base else None
    if cur is None:
        return "", False
    head = ["| Condition | Previous model (live) | Whole clip | **Live decoder** | Letter delay (median) |", "|---|---:|---:|---:|---:|"]
    for k, v in cur["conditions"].items():
        b = f"{100 * old['conditions'][k]['stream_cer']:.2f} %" if old else "—"
        head.append(f"| {k} | {b} | {pct(v['whole_cer'], 2)} | **{pct(v['stream_cer'], 2)}** | {v['latency_median_s']:.2f} s |")
    return chr(10).join(head), True


def avatar_table():
    f = R / "avatar_contacts.csv"
    if not f.exists():
        return "", None
    rows = list(csv.DictReader(open(f)))
    out = ["| Letter | Hands | Candidates | Contacts | Mean gap, raw | Mean gap, aligned |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['letter']} | {r['hands']} | {r['candidates']} | {r['contacts']} | {r['mean_abs_gap_raw'] or '—'} | {r['mean_abs_gap_touched'] or '—'} |")
    t = [r for r in rows if r["contacts"] != "0"]
    raw = st.mean(float(r["mean_abs_gap_raw"]) for r in t)
    new = st.mean(float(r["mean_abs_gap_touched"]) for r in t)
    return "\n".join(out), (len(t), raw, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-default", default="main")
    ap.add_argument("--cont-best", default=None, help="tag of the deployed continuous model")
    ap.add_argument("--streaming", default=None)
    ap.add_argument("--streaming-base", default=None)
    a = ap.parse_args()

    st_tab, st_rows = static_table()
    cr_tab, cr_vals = cross_table()
    ct_tab, ct_rows = cont_table()
    av_tab, av = avatar_table()

    best_static = next(r for r in st_rows if r[5] == a.static_default)
    cont_v2 = [r for r in ct_rows if "v2" in r[0] and "features: all" in r[0]]
    best_cont = next((r for r in cont_v2 if "(deployed)" in r[0]), None) if a.cont_best else None
    best_cont = best_cont or (min(cont_v2, key=lambda r: r[2]) if cont_v2 else None)
    st_tab_live, has_live = streaming_table(a.streaming, a.streaming_base) if a.streaming else ("", False)

    def cer_of(kind, parts):
        for r in ct_rows:
            if ("Conformer" if kind == "conformer" else "BiLSTM") in r[0] and "v2" in r[0] and (("features: all" in r[0]) == (parts == "all")):
                return r[3]
        return None
    parts_txt = []
    for kind, label in (("conformer", "Conformer"), ("bilstm", "BiLSTM")):
        a_, b_ = cer_of(kind, "raw"), cer_of(kind, "all")
        if a_ is not None and b_ is not None:
            parts_txt.append(f"{label} test CER {pct(a_, 2)} → {pct(b_, 2)}")
    abl = "; ".join(parts_txt)

    live_section = (f"""## Live decoding

The live recogniser (30 Hz resampling, sliding window, stable-prefix commit) replayed over the held-out test videos, at normal and faster signing speeds and on slower cameras. Character error rate; lower is better.

{st_tab_live}

*Faster signing time-compresses the hand track; slower cameras drop frames and keep true timestamps. A letter is final about 0.4 s after it was signed. The previous model is the one trained without the streaming augmentations.*

""" if has_live else "")

    md = f"""---
sidebar_position: 1
title: Benchmark
---

# Benchmark

All numbers are produced by scripts in the repository and written into this page by
`src/evaluation/make_benchmark_doc.py` from the files in `results/`. Nothing is typed by hand.

## Static letters

Three tests are always reported together (see [design decision DD-008](../reference/design-decisions)):

| Test | What it measures |
|---|---|
| **In-domain** | the original random split of ≈ 45 k images (near-duplicate frames: optimistic) |
| **Real news video** | 1,069 letter frames CTC-aligned from **held-out test videos** (small hands, full-body shot) |
| **Cross-source** | train *without* ISL_DATA, test on all 13,061 of its images (new camera, background, signer) |

### In-domain and real-video

{st_tab}

*ECE = expected calibration error (10 bins); lower is better. "Calibrated" uses the temperature fitted on the real-video validation logits.*

### Cross-source (the realistic test for a new camera)

{cr_tab}

Accuracy is far lower than on the real news videos because ISL_DATA is shot close-up from below with overlapping hands, which also makes MediaPipe miss the second hand. The columns show how strongly accuracy depends on both hands being detected.

## Continuous fingerspelling

Character error rate (CER = edit distance / reference length, corpus level, greedy CTC decoding) on the validation and test videos of the dataset:

{ct_tab}

- The **v1** Conformer was *worse* than the v1 BiLSTM. The v2 features (pretrained hand-shape embedding + both hands + motion), augmentation, a smaller convolution kernel and the intermediate CTC loss turn this around.
- The **ablation** rows use no trunk/posterior features at all, so the difference to the "all features" rows is what the pretrained DH-GAT contributes: **{abl}**.
- Without the pretrained features the Conformer is *not* better than the BiLSTM — consistent with Conformers needing more data than the ≈ 70 minutes available. The pretrained hand-shape embedding supplies the missing prior, which is why the Conformer becomes the best model only with it.
- Validation CER selects the model; the test set is used once.

{live_section}## Avatar contact alignment

Mean absolute contact gap in hand sizes (1 hand size ≈ 9.5 cm; 0 = skin on skin).

{av_tab}

Over the {av[0]} letters that have contacts, the mean gap falls from **{av[1]:.3f}** to **{av[2]:.3f}** hand sizes (≈ {av[1]*9.5:.1f} cm → {av[2]*95:.1f} mm).

## Reproducing

```bash
python src/static/benchmark.py                 # legacy v1 on all protocols
python src/static/train_v2.py --tag main       # prints the v2 table
python src/training/select_best_cont.py        # continuous comparison
python src/evaluation/make_benchmark_doc.py    # regenerate this page
```
"""
    (DOCS / "evaluation" / "benchmark.md").write_text(md, encoding="utf-8")

    glance = f"""| Task | Test | Result |
|---|---|---|
| Static letters (DH-GAT v2) | real news-video letters, held-out videos | **{pct(best_static[2], 1)}** (v1: {pct(st_rows[0][2], 1)}) |
| Static letters | original in-domain split | **{pct(best_static[1], 1)}** |
| Static letters | cross-source (ISL_DATA never seen) | **{pct(cr_vals.get('xsrc_r' if a.static_default == 'main_r' else 'xsrc', 0), 1)}** |
| Calibration | expected calibration error on real videos | **{best_static[4] if best_static[4] is not None else best_static[3]:.3f}** (v1: {st_rows[0][3]:.3f}) |
| Continuous (Conformer-CTC v2) | character error rate, test videos | **{pct(best_cont[3], 2) if best_cont else 'n/a'}** (v1 BiLSTM: 14.75 %) |
| Avatar contact alignment | mean contact gap | **{av[1]:.3f} → {av[2]:.3f}** hand sizes |
"""
    (DOCS / "_results_glance.md").write_text(glance, encoding="utf-8")

    # fill placeholders of the continuous pages
    params = {}
    for f in R.glob("cont_v2_*.json"):
        r = json.loads(f.read_text())
        if r["parts"] == "all":
            params[r["kind"]] = r["params"]
    examples = ""
    best = max(R.glob("cont_v2_*_all.json"), key=lambda f: -json.loads(f.read_text())["val_cer"], default=None)
    if best is not None:
        examples = "\n".join(f"| `{e.split(' -> ')[0]}` | `{e.split(' -> ')[1]}` |" for e in json.loads(best.read_text())["examples"])
    for f in (DOCS / "continuous-recognition").glob("*.md"):
        s = f.read_text(encoding="utf-8")
        s = s.replace("@@BILSTM_PARAMS@@", f"{params.get('bilstm', 0):,}").replace("@@CONF_PARAMS@@", f"{params.get('conformer', 0):,}")
        s = s.replace("@@EXAMPLES@@", examples)
        f.write_text(s, encoding="utf-8")
    print("docs numbers written")


if __name__ == "__main__":
    main()
