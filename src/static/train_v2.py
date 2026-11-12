"""
src/static/train_v2.py
======================
Train DH-GAT v2 on raw two-hand landmarks (cache/static_v2) and evaluate with the
Phase-1 benchmark protocols.

TRAIN DATA   static_isl + kaggle + isl_data (original train split)
             + real news-video letters from TRAIN videos (up-weighted: target domain)
MODEL SELECT real VAL videos accuracy (never the near-duplicate static val set)
TEST         real TEST videos | original in-domain test | isl_data (if --holdout isl_data)
CALIBRATION  temperature fitted on real VAL logits

  python src/static/train_v2.py --tag main
  python src/static/train_v2.py --tag xsrc --holdout isl_data     # cross-source test
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.models.dh_gat_v2 import DHGATv2, featurize  # noqa: E402
from src.static import benchmark as bm  # noqa: E402

import os
torch.set_num_threads(int(os.environ.get("ISL_THREADS", "10")))
CKPT_DIR = ROOT / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def to_px(d: dict):
    """raw normalised coords -> aspect-corrected tensors (lm, present, y)."""
    wh = d["wh"].astype(np.float32)
    scale = np.stack([wh[:, 0], wh[:, 1], wh[:, 0]], 1)[:, None, None, :]
    lm = torch.from_numpy((d["lm"] * scale).astype(np.float32))
    return lm, torch.from_numpy(d["present"]), torch.from_numpy(d["label"].astype(np.int64))


def build_sets(holdout: str = None):
    data = bm.load_all()
    tr, va_static, te_static = [], [], []
    for s in bm.SOURCES:
        if s == holdout:
            continue
        for split, bucket in (("train", tr), ("val", va_static), ("test", te_static)):
            bucket.append(bm.subset(data[s], bm.legacy_split_mask(data[s], s, split)))
    real = data["real"]
    sets = dict(
        train_static=bm.concat(tr),
        train_real=bm.subset(real, real["split"] == "train"),
        val_real=bm.subset(real, real["split"] == "val"),
        val_static=bm.concat(va_static),
        test_real=bm.subset(real, real["split"] == "test"),
        test_static=bm.concat(te_static),
    )
    if holdout:
        sets["test_holdout"] = data[holdout]
    return sets


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (batched, torch)
# ─────────────────────────────────────────────────────────────────────────────
def _rot(ax, ay, az):
    """batched rotation matrices (B,3,3) from angles (B,) in radians."""
    cx, sx, cy, sy, cz, sz = ax.cos(), ax.sin(), ay.cos(), ay.sin(), az.cos(), az.sin()
    o, z = torch.ones_like(ax), torch.zeros_like(ax)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).view(-1, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).view(-1, 3, 3)
    return Rz @ Ry @ Rx


def augment(lm: torch.Tensor, present: torch.Tensor, p_mirror=0.3, p_drop=0.04):
    B = lm.shape[0]
    lm, present = lm.clone(), present.clone()
    d2r = np.pi / 180
    pw = present.float()
    n = pw.sum(1).clamp(min=1)
    wrist = lm[:, :, 0]
    hs = (lm[:, :, 9] - wrist).norm(dim=-1).clamp(min=1e-6)
    hsize = (hs * pw).sum(1) / n                                          # (B,)
    center = (wrist * pw[..., None]).sum(1) / n[:, None]                  # (B,3)

    # per-hand independent: scale, small z-rotation about own wrist, translation
    s = 1 + (torch.rand(B, 2, 1, 1) - 0.5) * 0.2
    lm = wrist[:, :, None] + (lm - wrist[:, :, None]) * s
    rz = (torch.rand(B, 2) - 0.5) * 2 * 10 * d2r
    R = _rot(torch.zeros(B * 2), torch.zeros(B * 2), rz.reshape(-1)).view(B, 2, 3, 3)
    lm = wrist[:, :, None] + torch.einsum("bhij,bhnj->bhni", R, lm - wrist[:, :, None])
    lm = lm + (torch.randn(B, 2, 1, 3) * 0.12 * hsize[:, None, None, None])

    # global 3D rotation of the whole scene about its centre
    ax = (torch.rand(B) - 0.5) * 2 * 20 * d2r
    ay = (torch.rand(B) - 0.5) * 2 * 20 * d2r
    az = (torch.rand(B) - 0.5) * 2 * 30 * d2r
    Rg = _rot(ax, ay, az)
    lm = center[:, None, None] + torch.einsum("bij,bhnj->bhni", Rg, lm - center[:, None, None])

    # landmark noise (fingertips noisier, palm steadier)
    sig = torch.full((21,), 0.02)
    sig[[4, 8, 12, 16, 20]] = 0.03
    sig[[0, 1, 5, 9, 13, 17]] = 0.01
    lm = lm + torch.randn_like(lm) * sig[None, None, :, None] * hsize[:, None, None, None]

    # mirror (left-handed signer / mirrored webcam): flip x about centre, swap slots
    mir = torch.rand(B) < p_mirror
    if mir.any():
        lm[mir, :, :, 0] = 2 * center[mir, 0][:, None, None] - lm[mir, :, :, 0]
        lm[mir] = lm[mir].flip(1)
        present[mir] = present[mir].flip(1)

    # random hand dropout (simulates a missed MediaPipe detection)
    both = present.all(1)
    drop = both & (torch.rand(B) < p_drop)
    if drop.any():
        keep1 = torch.rand(B) < 0.5
        idx = torch.where(drop)[0]
        k = keep1[idx]                      # True -> keep slot 1, drop slot 0
        lm[idx[k], 0] = lm[idx[k], 1]
        present[idx, 0] = True
        present[idx, 1] = False
        lm[idx, 1] = 0
    # single hand must always sit in slot 0 (after mirror/drop)
    only1 = (~present[:, 0]) & present[:, 1]
    if only1.any():
        lm[only1] = lm[only1].flip(1)
        present[only1] = present[only1].flip(1)
    return lm, present


# ─────────────────────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def logits_of(model, lm, present, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(lm), bs):
        out.append(model.forward_raw(lm[i:i + bs], present[i:i + bs]))
    return torch.cat(out)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    t = torch.zeros(1, requires_grad=True)                    # log-temperature
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp())


class EMA:
    def __init__(self, model, decay=0.998):
        self.m = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                e.copy_(p)


def acc_of(model, lm, pr, y):
    return float((logits_of(model, lm, pr).argmax(1) == y).float().mean())


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--holdout", default=None, help="exclude this source from training and test on it")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epoch-size", type=int, default=32000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--real-weight", type=float, default=4.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    S = build_sets(a.holdout)
    T = {k: to_px(v) for k, v in S.items()}
    print({k: len(v["label"]) for k, v in S.items()})

    # training pool = static train + real train
    lm_tr = torch.cat([T["train_static"][0], T["train_real"][0]])
    pr_tr = torch.cat([T["train_static"][1], T["train_real"][1]])
    y_tr = torch.cat([T["train_static"][2], T["train_real"][2]])
    is_real = torch.cat([torch.zeros(len(T["train_static"][2])), torch.ones(len(T["train_real"][2]))])
    # sampling weight: domain weight / sqrt(class frequency within domain)
    w = torch.zeros(len(y_tr))
    for dom, dw in ((0, 1.0), (1, a.real_weight)):
        for c in range(26):
            m = (is_real == dom) & (y_tr == c)
            if m.any():
                w[m] = dw / np.sqrt(m.sum().item())
    w = w / w.sum()
    print(f"train pool {len(y_tr)}  (real {int(is_real.sum())});  real mass in sampler = {w[is_real == 1].sum():.2f}")

    model = DHGATv2()
    print("params", sum(p.numel() for p in model.parameters()))
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps_per_epoch = a.epoch_size // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.1)
    best, bad, t0 = -1, 0, time.time()
    CKPT_DIR.mkdir(exist_ok=True)
    ckpt_path = CKPT_DIR / f"dh_gat_v2_{a.tag}.pt"
    vr, vs = T["val_real"], T["val_static"]
    sub = torch.randperm(len(vs[2]))[:4000]

    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            idx = torch.multinomial(w, a.bs, replacement=True)
            lm, pr = augment(lm_tr[idx], pr_tr[idx])
            loss = F.cross_entropy(model.forward_raw(lm, pr), y_tr[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)
            tot += loss.item()
        ar = acc_of(ema.m, *vr)
        as_ = acc_of(ema.m, vs[0][sub], vs[1][sub], vs[2][sub])
        flag = ""
        if ar > best:
            best, bad, flag = ar, 0, " *"
            torch.save({"model_state": ema.m.state_dict(), "cfg": model.cfg, "epoch": ep,
                        "val_real_acc": ar}, ckpt_path)
        else:
            bad += 1
        print(f"ep {ep:02d} loss {tot/steps_per_epoch:.3f} | val_real {100*ar:.1f}% | val_static {100*as_:.1f}% "
              f"| {time.time()-t0:.0f}s{flag}", flush=True)
        if bad >= a.patience:
            print("early stop")
            break

    # ── final evaluation with the best checkpoint ──────────────────────────
    ck = torch.load(ckpt_path, weights_only=False)
    model = DHGATv2(**ck["cfg"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    temp = fit_temperature(logits_of(model, *vr[:2]), vr[2])
    print(f"\ntemperature (fit on real val) = {temp:.3f}")
    ck["temperature"] = temp
    torch.save(ck, ckpt_path)

    res = {}
    for name, key in (("real (test videos)", "test_real"), ("in_domain (orig. test)", "test_static"),
                      ("holdout source (cross-source)", "test_holdout")):
        if key not in T:
            continue
        lm, pr, y = T[key]
        lg = logits_of(model, lm, pr)
        for tag, tt in (("", 1.0), (" +calib", temp)):
            r = bm.summarize(torch.softmax(lg / tt, 1).numpy(), y.numpy(), pr.sum(1).numpy())
            res[name + tag] = r
    bm.print_table(f"DH-GAT v2 [{a.tag}]", res)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"benchmark_dh_gat_v2_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"\nsaved {ckpt_path.name} and results/benchmark_dh_gat_v2_{a.tag}.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
