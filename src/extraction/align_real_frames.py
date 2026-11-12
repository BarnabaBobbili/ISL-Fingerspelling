"""
src/extraction/align_real_frames.py
===================================
Builds a TRUSTWORTHY real-world static-sign benchmark from the continuous videos.

PROBLEM WITH THE OLD FRAMES (extract_static_frames_from_continuous.py)
  Each segment's time was split EVENLY across its letters and the frame with the
  highest hand-detection confidence was kept. Letters are not signed for equal
  durations, so many frames carry the wrong label.

THIS SCRIPT
  1. Runs the trained BiLSTM-CTC on the cached landmark sequence of a segment.
  2. CTC FORCED ALIGNMENT (Viterbi): finds the most probable frame span for each
     letter of the KNOWN transcript.
  3. Takes the frame where that letter's posterior peaks (the moment the model is
     most sure of the handshape) -> one frame per letter.
  4. Keeps only segments the model transcribes well (CER <= --max-cer) and letter
     peaks with posterior >= --min-post, so alignments are reliable.
  5. Re-detects BOTH hands on those frames (raw coords, same format as v2 cache).

OUTPUT
  cache/static_v2/real.npz   same fields as extract_v2 plus:
      uid, video, char_idx, post (peak posterior), split (train/val/test by VIDEO)
  data/real_frames_aligned/<LETTER>/<uid>_c<k>.jpg   (for visual inspection)

Run:  python src/extraction/align_real_frames.py
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.training.config import ANN_CSV, CACHE_DIR, CKPT_DIR, BLANK_IDX, CHAR_TO_IDX, IDX_TO_CHAR  # noqa: E402
from src.preprocessing.normalizer import normalize_sequence  # noqa: E402
from src.data.split_manager import build_final_split  # noqa: E402
from src.models.bilstm_ctc import build_bilstm  # noqa: E402

VIDEO_DIR = ROOT / "data" / "continious-Fingerspelling-dataset" / "videos"
OUT_NPZ = ROOT / "cache" / "static_v2" / "real.npz"
OUT_IMG = ROOT / "data" / "real_frames_aligned"


# ─────────────────────────────────────────────────────────────────────────────
def ctc_forced_align(log_probs: np.ndarray, target: list) -> list:
    """
    Viterbi forced alignment.

    log_probs : (T, C) log-softmax scores,  target : list of class ids (no blanks)
    Returns   : list (len == len(target)) of (start_frame, end_frame_inclusive)
                spans, or None if the target cannot fit in T frames.
    """
    T = log_probs.shape[0]
    L = len(target)
    S = 2 * L + 1
    if L == 0 or T < L:
        return None
    ext = np.full(S, BLANK_IDX, dtype=np.int64)
    ext[1::2] = target
    NEG = -1e30
    dp = np.full((T, S), NEG, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int8)           # 0 stay, 1 from s-1, 2 from s-2
    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], 1
            if s >= 2 and ext[s] != BLANK_IDX and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], 2
            dp[t, s] = best + log_probs[t, ext[s]]
            bp[t, s] = arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] <= NEG / 2:
        return None
    path = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = []
    for k in range(L):
        frames = np.where(path == 2 * k + 1)[0]
        if len(frames) == 0:
            return None
        spans.append((int(frames[0]), int(frames[-1])))
    return spans


def greedy_decode(log_probs: np.ndarray) -> str:
    ids = log_probs.argmax(-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK_IDX:
            out.append(IDX_TO_CHAR[int(i)])
        prev = i
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


# ─────────────────────────────────────────────────────────────────────────────
def load_bilstm(device="cpu"):
    ck = torch.load(CKPT_DIR / "bilstm_best.pt", map_location=device, weights_only=False)
    m = build_bilstm()
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval()


@torch.no_grad()
def posteriors(model, uid: str) -> np.ndarray:
    lm = np.load(CACHE_DIR / f"{uid}.npy")
    x = torch.from_numpy(normalize_sequence(lm)).unsqueeze(0).float()
    lp, _ = model(x, torch.tensor([x.shape[1]]))
    return lp[:, 0].numpy()                         # (T, 27)


def plan_frames(max_cer: float, min_post: float):
    """Step 1-4: returns list of dict(uid, video, k, letter, frame, post, split)."""
    model = load_bilstm()
    ann = pd.read_csv(ANN_CSV)
    split = build_final_split().set_index("uid")["split"].to_dict()
    picks, stats = [], dict(seg=0, ok=0, bad_cer=0, bad_align=0, letters=0)
    for _, row in tqdm(ann.iterrows(), total=len(ann), desc="aligning"):
        uid = row["uid"]
        if uid not in split or not (CACHE_DIR / f"{uid}.npy").exists():
            continue
        text = "".join(c for c in str(row["text"]).lower() if c in CHAR_TO_IDX)
        if not text:
            continue
        stats["seg"] += 1
        lp = posteriors(model, uid)
        cer = edit_distance(greedy_decode(lp), text) / len(text)
        if cer > max_cer:
            stats["bad_cer"] += 1
            continue
        target = [CHAR_TO_IDX[c] for c in text]
        spans = ctc_forced_align(lp, target)
        if spans is None:
            stats["bad_align"] += 1
            continue
        stats["ok"] += 1
        video = uid.rsplit("_seg", 1)[0]
        for k, ((a, b), ch) in enumerate(zip(spans, target)):
            seg = lp[a:b + 1, ch]
            f = a + int(seg.argmax())
            post = float(np.exp(seg.max()))
            if post < min_post:
                continue
            picks.append(dict(uid=uid, video=video, k=k, letter=text[k].upper(),
                              frame=f, post=post, split=split[uid]))
            stats["letters"] += 1
    print(stats)
    return picks


def grab_frames(picks):
    """Step 5: read frames from the mp4s, save JPEG, detect both hands."""
    from multiprocessing import Pool
    from src.static.extract_v2 import _init_worker, detect_hands
    import src.static.extract_v2 as ev2

    by_uid = {}
    for p in picks:
        by_uid.setdefault(p["uid"], []).append(p)
    jobs = []
    for uid, ps in by_uid.items():
        cap = cv2.VideoCapture(str(VIDEO_DIR / f"{uid}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {p["frame"]: p for p in ps}
        for fi in sorted(want):
            if fi >= n:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok:
                continue
            d = OUT_IMG / want[fi]["letter"]
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{uid}_c{want[fi]['k']}.jpg"
            cv2.imwrite(str(path), img)
            jobs.append((path, want[fi]))
        cap.release()
    print(f"saved {len(jobs)} frames, detecting hands...")
    return jobs, Pool, _init_worker, ev2


def _detect(path):
    img = cv2.imread(str(path))
    from src.static import extract_v2 as ev2
    return ev2.detect_hands(img, ev2._DET)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--min-post", type=float, default=0.6)
    a = ap.parse_args()
    t0 = time.time()
    picks = plan_frames(a.max_cer, a.min_post)
    jobs, Pool, init, _ = grab_frames(picks)
    with Pool(12, initializer=init) as pool:
        recs = list(tqdm(pool.imap(_detect, [j[0] for j in jobs], chunksize=16), total=len(jobs)))
    meta = [j[1] for j in jobs]
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        lm=np.stack([r["lm"] for r in recs]),
        present=np.stack([r["present"] for r in recs]),
        hand=np.stack([r["hand"] for r in recs]),
        score=np.stack([r["score"] for r in recs]),
        wh=np.stack([r["wh"] for r in recs]),
        label=np.array([ord(m["letter"]) - 65 for m in meta], np.int8),
        name=np.array([str(j[0].relative_to(ROOT / "data")) for j in jobs]),
        uid=np.array([m["uid"] for m in meta]),
        video=np.array([m["video"] for m in meta]),
        char_idx=np.array([m["k"] for m in meta], np.int16),
        post=np.array([m["post"] for m in meta], np.float32),
        split=np.array([m["split"] for m in meta]),
    )
    sp = pd.Series([m["split"] for m in meta]).value_counts().to_dict()
    nh = pd.Series([int(r["present"].sum()) for r in recs]).value_counts().sort_index().to_dict()
    print(f"-> {OUT_NPZ}\n splits={sp}  hands-detected-histogram={nh}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
