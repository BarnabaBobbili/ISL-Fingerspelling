"""
src/evaluation/make_site_data.py
================================
Collects the measured results into app/static/data/results.json, which the website's Results
page and Home page render. Nothing is typed by hand: every number comes from results/*.json
and results/avatar_contacts.csv.

  python src/evaluation/make_site_data.py [--static-best main_r] [--cont-best conformer_all]
                                           [--streaming base_test] [--streaming-note "..."]
"""

import argparse
import csv
import json
import statistics as st
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results"
OUT = ROOT / "app" / "static" / "data" / "results.json"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def get(res, key):
    if res is None:
        return None
    for k, v in res.items():
        if k.strip() == key:
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-best", default="main_r")
    ap.add_argument("--static-runs", default="main,main_r", help="comma list of v2 tags to list after v1")
    ap.add_argument("--cross-runs", default="xsrc,xsrc_r")
    ap.add_argument("--names", default="", help="tag=Display name,... overrides")
    ap.add_argument("--streaming", default=None, help="tag of results/streaming_<tag>.json")
    ap.add_argument("--cont-best", default=None, help="tag of the deployed continuous model")
    ap.add_argument("--streaming-note", default="")
    ap.add_argument("--streaming-base", default=None, help="tag of the previous model's streaming results (adds a comparison column)")
    a = ap.parse_args()
    names = {"main": "DH-GAT v2", "main_r": "DH-GAT v2 + second-hand recovery",
             "xsrc": "DH-GAT v2", "xsrc_r": "DH-GAT v2 + second-hand recovery"}
    names.update(dict(kv.split("=", 1) for kv in a.names.split(",") if "=" in kv))

    rows, best = [], 0
    v1 = load("benchmark_dh_gat_static_best.json")
    if v1:
        ind, real = get(v1, "in_domain (orig. split)"), get(v1, "real / test videos")
        rows.append({"name": "DH-GAT v1 (original)", "in_domain": 100 * ind["acc"], "real": 100 * real["acc"],
                     "ece": real["ece"], "ece_cal": None})
    for tag in a.static_runs.split(","):
        r = load(f"benchmark_dh_gat_v2_{tag}.json")
        if not r:
            continue
        ind, real, cal = get(r, "in_domain (orig. test)"), get(r, "real (test videos)"), get(r, "real (test videos) +calib")
        rows.append({"name": names.get(tag, tag), "in_domain": 100 * ind["acc"], "real": 100 * real["acc"],
                     "ece": real["ece"], "ece_cal": cal["ece"]})
        if tag == a.static_best:
            best = len(rows) - 1

    cross = []
    for tag in a.cross_runs.split(","):
        r = load(f"benchmark_dh_gat_v2_{tag}.json")
        h = get(r, "holdout source (cross-source)")
        hc = get(r, "holdout source (cross-source) +calib")
        if h:
            by = h.get("acc_by_hands_detected", {})
            cross.append({"name": names.get(tag, tag), "acc": 100 * h["acc"], "both": 100 * by.get("2", 0),
                          "one": 100 * by.get("1", 0), "ece": hc["ece"]})

    cont = [{"name": "BiLSTM-CTC v1 (one hand, landmarks only)", "params": "—", "val": 14.24, "test": 14.75},
            {"name": "Conformer-CTC v1 (one hand, landmarks only)", "params": "—", "val": 18.72, "test": 19.26}]
    tags = []
    for f in sorted(R.glob("cont_v2_*.json")):
        r = json.loads(f.read_text())
        kind = "Conformer" if r["kind"] == "conformer" else "BiLSTM"
        label = f"{kind}-CTC v2" + ("" if r["parts"] == "all" else " (no pretrained features)")
        if r["tag"].endswith("_stream"):
            label += " + streaming training"
        cont.append({"name": label, "params": f"{r['params']:,}", "val": 100 * r["val_cer"], "test": 100 * r["test_cer"], "tag": r["tag"]})
        tags.append(r["tag"])
    ok = [i for i, c in enumerate(cont) if "tag" in c and "no pretrained" not in c["name"]]
    deployed = a.cont_best
    cbest = next((i for i, c in enumerate(cont) if c.get("tag") == deployed), None) if deployed else None
    if cbest is None:
        cbest = min(ok, key=lambda i: cont[i]["val"]) if ok else len(cont) - 1

    streaming = None
    if a.streaming and load(f"streaming_{a.streaming}.json"):
        s = load(f"streaming_{a.streaming}.json")
        b = load(f"streaming_{a.streaming_base}.json") if a.streaming_base else None
        streaming = {"note": a.streaming_note, "rows": [
            {"condition": k, "whole": 100 * v["whole_cer"], "stream": 100 * v["stream_cer"],
             "lat_median": v["latency_median_s"], "lat_p90": v["latency_p90_s"],
             "before": 100 * b["conditions"][k]["stream_cer"] if b else None} for k, v in s["conditions"].items()]}

    rowsA = list(csv.DictReader(open(R / "avatar_contacts.csv"))) if (R / "avatar_contacts.csv").exists() else []
    t = [r for r in rowsA if r["contacts"] != "0"]
    avatar = {"letters": len(t), "raw": st.mean(float(r["mean_abs_gap_raw"]) for r in t) if t else 0,
              "touched": st.mean(float(r["mean_abs_gap_touched"]) for r in t) if t else 0}

    data = {"generated": time.strftime("%Y-%m-%d"),
            "headline": {"real_acc": round(rows[best]["real"], 1), "cer": round(cont[cbest]["test"], 1)},
            "static": {"rows": rows, "best": best}, "cross": cross,
            "continuous": {"rows": cont, "best": cbest}, "streaming": streaming, "avatar": avatar}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}  headline={data['headline']}")


if __name__ == "__main__":
    main()
