"""
tests/test_streaming.py
=======================
Checks of the live recogniser (src/pipeline/streaming.py):

  * Resampler: any camera rate -> exactly the 30 Hz grid, linear interpolation, hand-set changes
  * StreamingSession on a real held-out video: the committed text equals the whole-clip decode,
    letters are committed with a bounded delay, and the pause commits the last letter
  * the same video at HALF the camera rate still reads correctly (the point of the resampler)

  python tests/test_streaming.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.streaming import Resampler  # noqa: E402


def test_resampler_rate_and_interpolation():
    for cam_fps in (10, 15, 24, 30, 60):
        r = Resampler(30.0)
        n_out, last = 0, None
        for i in range(cam_fps * 2 + 1):                       # 2 seconds of camera frames
            px = np.zeros((2, 21, 3), np.float32)
            px[0, 0, 0] = i / cam_fps * 10.0                   # x grows linearly with time (10 units per second)
            for q, pr in r.push(i / cam_fps, px, np.array([True, False])):
                n_out += 1
                last = q
        assert abs(n_out - 60) <= 2, (cam_fps, n_out)
        assert abs(last[0, 0, 0] - 20.0) < 0.5, (cam_fps, last[0, 0, 0])       # interpolated value follows the line


def test_resampler_bridges_nothing_across_long_gaps():
    r = Resampler(30.0, max_gap=0.25)
    z = np.zeros((2, 21, 3), np.float32)
    r.push(0.0, z, np.array([True, False]))
    out = r.push(2.0, z, np.array([True, False]))              # a 2 s stall must not be filled with fake frames
    assert len(out) == 1


def test_resampler_hand_set_change_uses_nearest():
    r = Resampler(30.0)
    a = np.zeros((2, 21, 3), np.float32)
    b = np.ones((2, 21, 3), np.float32)
    r.push(0.0, a, np.array([True, False]))
    out = r.push(0.1, b, np.array([True, True]))               # second hand appears: no blending of 1 hand with 2
    assert all(pr.tolist() in ([True, False], [True, True]) for _, pr in out)
    assert any(pr.all() for _, pr in out) and any(not pr.all() for _, pr in out)


def _load():
    ck = ROOT / "checkpoints" / "cont_best.pt"
    npz = ROOT / "cache" / "cont_v2" / "2KoaQCJSzuI_seg001.npz"
    if not ck.exists() or not npz.exists():
        print("skip: model or cache not available")
        return None
    from src.pipeline.continuous import ContinuousModel
    return ContinuousModel.load(), np.load(npz)


def _stream(model, d, step=1):
    from src.pipeline.streaming import StreamingSession
    s = StreamingSession(model, commit_lag=12)
    fps = 29.97
    for i in range(0, len(d["lm"]), step):
        s.step({"lm": d["lm"][i], "present": d["present"][i], "wh": d["wh"]}, 100.0 + i / fps, i)
    return s


def test_streaming_matches_truth_and_commits_with_bounded_delay():
    got = _load()
    if got is None:
        return
    model, d = got
    s = _stream(model, d)
    text = s.flush().replace(" ", "").lower()
    assert text == "devendrafadnavis", text
    assert s.latencies and max(s.latencies) <= 20, max(s.latencies)             # <= ~0.67 s at 30 Hz


def test_streaming_half_camera_rate():
    got = _load()
    if got is None:
        return
    model, d = got
    text = _stream(model, d, step=2).flush().replace(" ", "").lower()           # 15 fps camera
    from src.pipeline import ctc
    assert ctc.cer([text], ["devendrafadnavis"]) <= 0.07, text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("ALL PASSED")
