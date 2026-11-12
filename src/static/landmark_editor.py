"""
src/static/landmark_editor.py
==============================
Interactive two-hand landmark editor for ISL sign avatar poses.

Both right and left hands are shown simultaneously.
Drag interactions:
  - Click on a landmark dot       → move that single landmark
  - Click on the wrist dot (0)    → drag the whole hand
  - Click anywhere on the hand body (within 55 px of any landmark,
    not on a specific dot)        → drag the whole hand

Press S to save both hand poses for the current letter.
text_to_sign_avatar.py will automatically prefer curated poses.

Controls:
  Left-click + drag  : individual landmark OR whole-hand (see above)
  LEFT / RIGHT       : previous / next letter
  A / D              : same as arrows
  S                  : save both hands for this letter
  R                  : reset current letter to dataset medoid
  Ctrl+Z             : undo last edit
  Q / ESC            : quit

Run:
  python src/static/landmark_editor.py
  python src/static/landmark_editor.py --letter D
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.config import STATIC_LETTERS, STATIC_CACHE_DIR, HAND_EDGES, ROOT

CURATED_DIR = ROOT / "cache" / "static_curated"
CURATED_DIR.mkdir(parents=True, exist_ok=True)

# ── Canvas ─────────────────────────────────────────────────────────────────────
CANVAS_W  = 1100
CANVAS_H  = 680
BG        = (18, 18, 28)
SCALE     = 250          # pixels per normalized unit

# Starting wrist anchor positions (canvas px) — can be dragged around
INIT_R_ANCHOR = (CANVAS_W // 4,         CANVAS_H - 100)
INIT_L_ANCHOR = (3 * CANVAS_W // 4,     CANVAS_H - 100)

# Interaction radii
HIT_R      = 13    # px — grab a single landmark
HAND_GRAB_R = 55   # px — grab whole hand (click on body, not on a dot)

MAX_UNDO = 30

# ── Visual style ───────────────────────────────────────────────────────────────
FINGER_COLS = [
    (160, 160, 180),   # 0 palm edges / wrist
    (60,  160, 255),   # 1 thumb   — blue
    (255, 160,  40),   # 2 index   — orange
    (60,  220,  80),   # 3 middle  — green
    (200,  80, 255),   # 4 ring    — purple
    (255,  80, 120),   # 5 pinky   — pink
]
LM_FINGER = [
    0, 1,1,1,1,
    2,2,2,2,
    3,3,3,3,
    4,4,4,4,
    5,5,5,5,
]
LM_NAMES = [
    "Wrist",
    "Thumb CMC","Thumb MCP","Thumb IP","Thumb Tip",
    "Index MCP","Index PIP","Index DIP","Index Tip",
    "Middle MCP","Middle PIP","Middle DIP","Middle Tip",
    "Ring MCP","Ring PIP","Ring DIP","Ring Tip",
    "Pinky MCP","Pinky PIP","Pinky DIP","Pinky Tip",
]
PALM_EDGES = frozenset([
    frozenset((0,1)), frozenset((0,5)),  frozenset((0,9)),
    frozenset((0,13)), frozenset((0,17)),
    frozenset((5,9)), frozenset((9,13)), frozenset((13,17)),
])
TIPS = {4, 8, 12, 16, 20}

OPEN_HAND = np.array([
    [ 0.00,  0.00, 0.0],
    [-0.20, -0.22, 0.0], [-0.36, -0.46, 0.0], [-0.47, -0.65, 0.0], [-0.55, -0.82, 0.0],
    [-0.10, -0.72, 0.0], [-0.10, -1.02, 0.0], [-0.10, -1.22, 0.0], [-0.10, -1.42, 0.0],
    [ 0.05, -0.76, 0.0], [ 0.05, -1.07, 0.0], [ 0.05, -1.27, 0.0], [ 0.05, -1.45, 0.0],
    [ 0.20, -0.72, 0.0], [ 0.20, -1.00, 0.0], [ 0.20, -1.19, 0.0], [ 0.20, -1.36, 0.0],
    [ 0.35, -0.62, 0.0], [ 0.35, -0.87, 0.0], [ 0.35, -1.03, 0.0], [ 0.35, -1.18, 0.0],
], dtype=np.float32)


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def _to_pts(lm: np.ndarray, ax: int, ay: int) -> list:
    """Normalized (21,3) → pixel list given anchor (ax,ay) for wrist."""
    return [(int(ax + lm[i,0]*SCALE), int(ay + lm[i,1]*SCALE)) for i in range(21)]


def _pts_to_lm(pts: list, ax: int, ay: int, lm_z: np.ndarray) -> np.ndarray:
    """Pixel list → normalized (21,3) given anchor; z preserved from lm_z."""
    out = np.zeros((21, 3), dtype=np.float32)
    for i, (px, py) in enumerate(pts):
        out[i, 0] = (px - ax) / SCALE
        out[i, 1] = (py - ay) / SCALE
        out[i, 2] = lm_z[i, 2] if lm_z is not None else 0.0
    return out


# ── Data ───────────────────────────────────────────────────────────────────────

def _medoid(samples):
    arr = np.stack(samples)
    c   = arr.mean(axis=0)
    d   = np.linalg.norm((arr - c).reshape(len(arr), -1), axis=1)
    return arr[np.argmin(d)]


def load_pose(letter: str, hand: str):
    """Return (lm (21,3), is_curated bool). Priority: curated > medoid > default."""
    p = CURATED_DIR / f"{letter}_{hand}.npz"
    if p.exists():
        try:
            return np.load(str(p))["lm"].astype(np.float32), True
        except Exception:
            pass

    ld = STATIC_CACHE_DIR / letter
    if not ld.exists():
        return OPEN_HAND.copy(), False

    key_lm, key_pr = f"{hand}_lm", f"{hand}_present"
    samples = []
    for fp in sorted(ld.glob("*.npz"))[:400]:
        try:
            d = np.load(str(fp))
            if bool(d[key_pr]):
                lm = d[key_lm].astype(np.float32)
                if lm.shape == (21,3) and not np.all(lm==0):
                    if 0.4 < float(np.linalg.norm(lm[9])) < 2.5:
                        samples.append(lm)
        except Exception:
            continue
    return (_medoid(samples), False) if samples else (OPEN_HAND.copy(), False)


def save_pose(letter: str, hand: str, lm: np.ndarray):
    p = CURATED_DIR / f"{letter}_{hand}.npz"
    np.savez_compressed(str(p), lm=lm.astype(np.float32))
    print(f"[editor] Saved {letter} {hand} → {p}")


# ── Drawing ─────────────────────────────────────────────────────────────────────

def _draw_hand(canvas, pts, label, hover_hand, hover_lm, drag_hand, drag_lm, hand_id):
    """Draw one hand skeleton with interactive highlights.
    hover_hand / drag_hand: which hand is being hovered / dragged ("right"|"left"|None)
    hover_lm  / drag_lm  : landmark index (-1 = whole-hand mode)
    hand_id: "right" or "left"
    """
    is_active = (drag_hand == hand_id) or (hover_hand == hand_id and drag_hand is None)

    # Edges
    for a, b in HAND_EDGES:
        key = frozenset((a, b))
        if key in PALM_EDGES:
            col, thick = (80, 80, 100), 2
        else:
            fi = LM_FINGER[b]
            col = FINGER_COLS[fi]
            thick = 3 if is_active else 2
        cv2.line(canvas, pts[a], pts[b], col, thick, cv2.LINE_AA)

    # Whole-hand drag indicator ring around wrist
    if drag_hand == hand_id and drag_lm == -1:
        cv2.circle(canvas, pts[0], 28, (255, 255, 100), 1, cv2.LINE_AA)

    # Dots
    for i, pt in enumerate(pts):
        fi = LM_FINGER[i]
        base = FINGER_COLS[fi]

        is_sel = (drag_hand == hand_id and drag_lm == i)
        is_hov = (hover_hand == hand_id and hover_lm == i and drag_hand is None)

        if i == 0:   # wrist — whole-hand grab handle
            wrist_col = (100, 230, 160) if (is_active and
                         ((hover_lm == 0 and hover_hand == hand_id) or
                          (drag_hand == hand_id and drag_lm in (-1, 0)))) else (60, 60, 80)
            cv2.circle(canvas, pt, 12, wrist_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 12, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, "0", (pt[0]+14, pt[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140,140,160), 1)
            continue

        if is_sel:
            r, ring_col = 13, (255, 255, 255)
        elif is_hov:
            r, ring_col = 11, (210, 210, 210)
        else:
            r, ring_col = 7 if not is_active else 8, base

        cv2.circle(canvas, pt, r+3, ring_col, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt, r, base, -1, cv2.LINE_AA)
        if i in TIPS:
            cv2.circle(canvas, pt, r-3, (255,255,255), 1, cv2.LINE_AA)

        if is_sel or is_hov:
            cv2.putText(canvas, str(i), (pt[0]+12, pt[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230,230,230), 1)

    # Hand label below wrist
    w_col = (0, 220, 100) if hand_id == "right" else (100, 180, 255)
    cv2.putText(canvas, label, (pts[0][0]-30, pts[0][1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, w_col, 1)


def _draw_hud(canvas, letter_idx, hover_hand, hover_lm, drag_hand, drag_lm,
              lm_r, lm_l, saved, undo_depth):
    h, w = canvas.shape[:2]

    # ── Top: A–Z strip ──
    bar_h = 38
    ov = canvas.copy()
    cv2.rectangle(ov, (0,0), (w, bar_h), (22, 22, 38), -1)
    canvas[:] = cv2.addWeighted(ov, 0.9, canvas, 0.1, 0)

    step = w // 26
    for i, lt in enumerate(STATIC_LETTERS):
        x = i * step + step//2 - 7
        if i == letter_idx:
            cv2.rectangle(canvas, (i*step, 2), (i*step+step-1, bar_h-2),
                          (0, 160, 70), -1)
            col, thick = (255, 255, 255), 2
        else:
            col, thick = (80, 80, 95), 1
        cv2.putText(canvas, lt, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, thick)

    # ── Big letter ──
    lt = STATIC_LETTERS[letter_idx]
    cv2.putText(canvas, lt, (w//2 - 40, h - 80),
                cv2.FONT_HERSHEY_DUPLEX, 4.5, (0, 235, 115), 5, cv2.LINE_AA)

    # ── Save status ──
    s_col = (0, 200, 90) if saved else (180, 120, 50)
    s_txt = "SAVED" if saved else "unsaved — press S to save"
    cv2.putText(canvas, s_txt, (10, h - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, s_col, 1)

    # ── Hovered / dragged info ──
    idx  = drag_lm  if drag_hand  else hover_lm
    hand = drag_hand if drag_hand else hover_hand
    lm   = lm_r if hand == "right" else lm_l
    if idx is not None and idx >= 0 and lm is not None:
        fi   = LM_FINGER[idx]
        col  = FINGER_COLS[fi]
        name = LM_NAMES[idx]
        xv, yv, zv = lm[idx]
        info = f"[{idx:02d}] {name}   x={xv:+.3f}  y={yv:+.3f}"
        cv2.putText(canvas, info, (10, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
    elif drag_hand and drag_lm == -1:
        cv2.putText(canvas, f"Dragging {drag_hand} hand", (10, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 100), 1)

    # ── Controls (top-right) ──
    keys = [
        "Drag dot   — move landmark",
        "Drag body  — move whole hand",
        "L / R  or  A / D   — letter",
        "S — save    R — reset",
        "Ctrl+Z — undo    Q — quit",
    ]
    for ki, txt in enumerate(keys):
        cv2.putText(canvas, txt, (w - 310, 60 + ki * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (85, 85, 100), 1)

    # ── Undo depth ──
    if undo_depth:
        cv2.putText(canvas, f"undo: {undo_depth}", (w - 100, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 110), 1)

    # ── Finger legend (bottom-right) ──
    fx = w - 100
    fy = h - 160
    for fi, (fn, fc) in enumerate(
            zip(["Thumb","Index","Middle","Ring","Pinky"], FINGER_COLS[1:])):
        cv2.circle(canvas, (fx, fy + fi*22), 5, fc, -1)
        cv2.putText(canvas, fn, (fx+10, fy+fi*22+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, fc, 1)


# ── Editor state ───────────────────────────────────────────────────────────────

class _ES:
    letter_idx  : int   = 0
    # Per-hand data
    lm_r        : np.ndarray = None    # (21,3) right hand normalized
    lm_l        : np.ndarray = None    # (21,3) left  hand normalized
    lm_r_orig   : np.ndarray = None    # medoid for R-reset
    lm_l_orig   : np.ndarray = None
    pts_r       : list  = None         # pixel pts, right
    pts_l       : list  = None         # pixel pts, left
    ax_r        : int   = INIT_R_ANCHOR[0]
    ay_r        : int   = INIT_R_ANCHOR[1]
    ax_l        : int   = INIT_L_ANCHOR[0]
    ay_l        : int   = INIT_L_ANCHOR[1]
    # Drag state
    drag_hand   = None             # "right" | "left" | None
    drag_lm     : int   = None     # landmark index, or -1 for whole-hand
    drag_start  = None             # (mx, my) at drag start
    pts_r_start : list  = None     # copy of pts_r at drag start
    pts_l_start : list  = None
    ax_r_start  : int   = 0
    ay_r_start  : int   = 0
    ax_l_start  : int   = 0
    ay_l_start  : int   = 0
    # Hover state
    hover_hand  = None
    hover_lm    : int   = -1
    # Misc
    undo_stack  : list  = None
    saved       : bool  = False

_s = _ES()
_s.undo_stack = []


def _refresh_pts():
    _s.pts_r = _to_pts(_s.lm_r, _s.ax_r, _s.ay_r)
    _s.pts_l = _to_pts(_s.lm_l, _s.ax_l, _s.ay_l)


def _load_letter():
    lt = STATIC_LETTERS[_s.letter_idx]
    lr, _ = load_pose(lt, "right")
    ll, _ = load_pose(lt, "left")
    # Load fresh medoid (ignoring curated) for R-reset
    _s.lm_r_orig = _dataset_medoid(lt, "right")
    _s.lm_l_orig = _dataset_medoid(lt, "left")
    _s.lm_r = lr.copy()
    _s.lm_l = ll.copy()
    _s.ax_r, _s.ay_r = INIT_R_ANCHOR
    _s.ax_l, _s.ay_l = INIT_L_ANCHOR
    _refresh_pts()
    _s.undo_stack.clear()
    _s.drag_hand = None
    _s.drag_lm   = None
    _s.hover_hand = None
    _s.hover_lm   = -1
    _s.saved      = (CURATED_DIR / f"{lt}_right.npz").exists() or \
                    (CURATED_DIR / f"{lt}_left.npz").exists()


def _dataset_medoid(letter: str, hand: str) -> np.ndarray:
    ld = STATIC_CACHE_DIR / letter
    if not ld.exists():
        return OPEN_HAND.copy()
    key_lm, key_pr = f"{hand}_lm", f"{hand}_present"
    samples = []
    for fp in sorted(ld.glob("*.npz"))[:400]:
        try:
            d = np.load(str(fp))
            if bool(d[key_pr]):
                lm = d[key_lm].astype(np.float32)
                if lm.shape==(21,3) and not np.all(lm==0):
                    if 0.4 < float(np.linalg.norm(lm[9])) < 2.5:
                        samples.append(lm)
        except Exception:
            continue
    return _medoid(samples) if samples else OPEN_HAND.copy()


def _nearest_lm(pts, mx, my):
    """Return (best_idx, best_dist) for nearest landmark to (mx,my)."""
    best_d, best_i = float("inf"), -1
    for i, (px, py) in enumerate(pts):
        d = ((px-mx)**2 + (py-my)**2)**0.5
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d


def _push_undo():
    _s.undo_stack.append((
        _s.lm_r.copy(), _s.lm_l.copy(),
        _s.ax_r, _s.ay_r, _s.ax_l, _s.ay_l,
    ))
    if len(_s.undo_stack) > MAX_UNDO:
        _s.undo_stack.pop(0)


# ── Mouse callback ─────────────────────────────────────────────────────────────

def _mouse(event, mx, my, flags, param):
    if _s.pts_r is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        _push_undo()
        _s.saved = False

        # Test right hand first, then left
        for hand_id, pts in [("right", _s.pts_r), ("left", _s.pts_l)]:
            idx, dist = _nearest_lm(pts, mx, my)
            if dist <= HIT_R:
                # Individual landmark (wrist = idx 0 → whole-hand)
                _s.drag_hand = hand_id
                _s.drag_lm   = idx if idx != 0 else -1
                _s.drag_start = (mx, my)
                _s.pts_r_start = [p for p in _s.pts_r]
                _s.pts_l_start = [p for p in _s.pts_l]
                _s.ax_r_start, _s.ay_r_start = _s.ax_r, _s.ay_r
                _s.ax_l_start, _s.ay_l_start = _s.ax_l, _s.ay_l
                return
            elif dist <= HAND_GRAB_R:
                # Whole-hand drag
                _s.drag_hand = hand_id
                _s.drag_lm   = -1
                _s.drag_start = (mx, my)
                _s.pts_r_start = [p for p in _s.pts_r]
                _s.pts_l_start = [p for p in _s.pts_l]
                _s.ax_r_start, _s.ay_r_start = _s.ax_r, _s.ay_r
                _s.ax_l_start, _s.ay_l_start = _s.ax_l, _s.ay_l
                return

        # Click landed on nothing — pop the undo we just pushed
        _s.undo_stack.pop()
        _s.saved = _s.saved  # keep

    elif event == cv2.EVENT_MOUSEMOVE:
        if _s.drag_hand is not None:
            dx = mx - _s.drag_start[0]
            dy = my - _s.drag_start[1]

            if _s.drag_lm == -1:
                # Whole-hand drag: shift anchor
                if _s.drag_hand == "right":
                    _s.ax_r = _s.ax_r_start + dx
                    _s.ay_r = _s.ay_r_start + dy
                    _s.pts_r = _to_pts(_s.lm_r, _s.ax_r, _s.ay_r)
                else:
                    _s.ax_l = _s.ax_l_start + dx
                    _s.ay_l = _s.ay_l_start + dy
                    _s.pts_l = _to_pts(_s.lm_l, _s.ax_l, _s.ay_l)
            else:
                # Individual landmark drag
                if _s.drag_hand == "right":
                    ox, oy = _s.pts_r_start[_s.drag_lm]
                    nx = int(np.clip(ox + dx, 5, CANVAS_W - 5))
                    ny = int(np.clip(oy + dy, 5, CANVAS_H - 5))
                    _s.pts_r[_s.drag_lm] = (nx, ny)
                    _s.lm_r = _pts_to_lm(_s.pts_r, _s.ax_r, _s.ay_r, _s.lm_r)
                else:
                    ox, oy = _s.pts_l_start[_s.drag_lm]
                    nx = int(np.clip(ox + dx, 5, CANVAS_W - 5))
                    ny = int(np.clip(oy + dy, 5, CANVAS_H - 5))
                    _s.pts_l[_s.drag_lm] = (nx, ny)
                    _s.lm_l = _pts_to_lm(_s.pts_l, _s.ax_l, _s.ay_l, _s.lm_l)

        else:
            # Update hover
            best_hand, best_lm, best_d = None, -1, float("inf")
            for hand_id, pts in [("right", _s.pts_r), ("left", _s.pts_l)]:
                idx, d = _nearest_lm(pts, mx, my)
                if d < HIT_R and d < best_d:
                    best_hand, best_lm, best_d = hand_id, idx, d
            _s.hover_hand = best_hand
            _s.hover_lm   = best_lm

    elif event == cv2.EVENT_LBUTTONUP:
        _s.drag_hand = None
        _s.drag_lm   = None


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_editor(start_letter: str = "A"):
    _s.letter_idx = STATIC_LETTERS.index(start_letter.upper())
    _load_letter()

    WIN = "ISL Landmark Editor  —  Two-Hand"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WIN, _mouse)

    print(f"[editor] Letter {start_letter.upper()} loaded.")
    print("[editor] Click dot=landmark drag | click body=whole-hand drag | S=save | Q=quit")

    while True:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)

        lt = STATIC_LETTERS[_s.letter_idx]
        _draw_hand(canvas, _s.pts_r, f"Right ({lt})",
                   _s.hover_hand, _s.hover_lm,
                   _s.drag_hand,  _s.drag_lm,  "right")
        _draw_hand(canvas, _s.pts_l, f"Left ({lt})",
                   _s.hover_hand, _s.hover_lm,
                   _s.drag_hand,  _s.drag_lm,  "left")
        _draw_hud(canvas, _s.letter_idx,
                  _s.hover_hand, _s.hover_lm,
                  _s.drag_hand,  _s.drag_lm,
                  _s.lm_r, _s.lm_l,
                  _s.saved, len(_s.undo_stack))

        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(16) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key in (ord("d"), 83):    # D or RIGHT arrow
            _s.letter_idx = (_s.letter_idx + 1) % 26
            _load_letter()

        elif key in (ord("a"), 81):    # A or LEFT arrow
            _s.letter_idx = (_s.letter_idx - 1) % 26
            _load_letter()

        elif key in (ord("s"), ord("S")):
            save_pose(lt, "right", _s.lm_r)
            save_pose(lt, "left",  _s.lm_l)
            _s.saved = True

        elif key in (ord("r"), ord("R")):
            _s.lm_r = _s.lm_r_orig.copy()
            _s.lm_l = _s.lm_l_orig.copy()
            _s.ax_r, _s.ay_r = INIT_R_ANCHOR
            _s.ax_l, _s.ay_l = INIT_L_ANCHOR
            _refresh_pts()
            _s.undo_stack.clear()
            _s.saved = False
            print(f"[editor] Reset {lt} to dataset medoid")

        elif key == 26:                # Ctrl+Z (ASCII 26)
            if _s.undo_stack:
                _s.lm_r, _s.lm_l, _s.ax_r, _s.ay_r, _s.ax_l, _s.ay_l = \
                    _s.undo_stack.pop()
                _refresh_pts()
                _s.saved = False

    cv2.destroyAllWindows()
    print("[editor] Closed.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Two-Hand Landmark Editor")
    parser.add_argument("--letter", type=str, default="A",
                        help="Starting letter (A-Z)")
    args = parser.parse_args()
    run_editor(start_letter=args.letter)
