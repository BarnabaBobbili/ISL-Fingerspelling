"""
app/server.py
=============
Web back end for the ISL fingerspelling demo (Starlette + uvicorn).

  python app/server.py                 # http://127.0.0.1:8000
  python app/server.py --port 8080 --host 0.0.0.0

Endpoints
  GET  /                      single-page app (app/static/index.html)
  WS   /ws/live?mode=static|continuous
                              webcam frames in (binary JPEG), recognition state out (JSON)
  POST /api/video             upload a video (multipart: file, mode) -> {job}
  GET  /api/job/{id}          progress / result of an upload job
  GET  /api/info              model + system information, supported characters
  POST /api/text2sign         {"text": "..."} -> letters to sign (A-Z, spaces kept as pauses)
  GET  /jobs/{id}/...         files of a finished job (original video, keyframes, annotated video)
"""

import argparse
import json
import mimetypes
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline.hands import HandTracker  # noqa: E402
from src.pipeline.classifier import StaticClassifier  # noqa: E402
from src.pipeline.hold import Recognizer  # noqa: E402
from src.pipeline.video import analyze_static  # noqa: E402

# Windows registries often map these wrongly; browsers refuse ES modules / WASM with a wrong type
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")

STATIC = Path(__file__).parent / "static"
JOBS_DIR = ROOT / "results" / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = 300 * 1024 * 1024
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def jsonable(o):
    """recursively convert numpy scalars/arrays (and tuples) to plain JSON types."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def err(msg: str, code: int = 400):
    return JSONResponse({"detail": msg}, status_code=code)


# ─────────────────────────────────────────────────────────────────────────────
class Engine:
    """Lazily loaded, shared models (one MediaPipe tracker guarded by a lock)."""
    _lock = threading.Lock()
    tracker = None          # plain tracker (continuous mode: its features were built without recovery)
    tracker_static = None   # tracker matching the static classifier (recovery on for *_r models)
    clf = None
    cont = None
    cont_error = None

    @classmethod
    def load(cls):
        with cls._lock:
            if cls.tracker is None:
                cls.clf = StaticClassifier()
                cls.tracker_static = HandTracker(recover=cls.clf.recover)
                cls.tracker = cls.tracker_static if not cls.clf.recover else HandTracker()
            return cls

    @classmethod
    def continuous(cls):
        """continuous (CTC) model, loaded on first use; None if no checkpoint is available."""
        with cls._lock:
            if cls.cont is None and cls.cont_error is None:
                try:
                    from src.pipeline.continuous import ContinuousModel
                    cls.cont = ContinuousModel.load()
                except Exception as e:  # noqa: BLE001
                    cls.cont_error = f"{type(e).__name__}: {e}"
            return cls.cont


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=Engine.load, daemon=True).start()     # warm the models in the background
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Live webcam
# ─────────────────────────────────────────────────────────────────────────────
def _rec_from_packet(msg: dict) -> dict:
    """landmark packet from the in-browser tracker -> the same record HandTracker produces."""
    rec = {"lm": np.zeros((2, 21, 3), np.float32), "present": np.zeros(2, bool),
           "wh": np.array([int(msg.get("w", 640)), int(msg.get("h", 480))], np.int32), "score": np.zeros(2, np.float32)}
    hands = sorted((np.asarray(h, np.float32) for h in msg.get("hands", [])[:2]), key=lambda a: a[0, 0])
    for i, arr in enumerate(hands):
        if arr.shape == (21, 3):
            rec["lm"][i] = arr
            rec["present"][i] = True
    return rec


def _process_rec(rec, rcg, session, mode, t, n, t_start=None):
    """recognition step for one frame's landmarks (from the server tracker or the browser)."""
    eng = Engine.load()
    t0 = time.time()
    hands = [[[round(float(x), 4), round(float(y), 4)] for x, y, _ in rec["lm"][i]]
             for i in range(2) if rec["present"][i]]
    if mode == "continuous" and session is not None:
        out = session.step(rec, t, n)
    else:
        probs = eng.clf.predict(rec)
        out = rcg.step(rec, probs, t, n)
    out["hands"] = hands
    out["ms"] = round((time.time() - t0) * 1000, 1)
    out["mode"] = mode
    out["seq"] = n
    return out


def _process_live(frame, rcg, session, mode, t, n):
    """JPEG fallback path: detect hands on the server, then recognise."""
    eng = Engine.load()
    t0 = time.time()
    rec = (eng.tracker if mode == "continuous" else eng.tracker_static)(frame)
    out = _process_rec(rec, rcg, session, mode, t, n)
    out["ms"] = round((time.time() - t0) * 1000, 1)
    return out


async def ws_live(ws: WebSocket):
    await ws.accept()
    mode = ws.query_params.get("mode", "static")
    session = None
    if mode == "continuous":
        model = await run_in_threadpool(Engine.continuous)
        if model is None:
            await ws.send_json({"error": f"continuous model unavailable ({Engine.cont_error})"})
            await ws.close()
            return
        from src.pipeline.continuous import ContinuousSession
        session = ContinuousSession(model)
    rcg = Recognizer()
    t0, n = time.time(), 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                frame = cv2.imdecode(np.frombuffer(msg["bytes"], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                out = await run_in_threadpool(_process_live, frame, rcg, session, mode, time.time() - t0, n)
                n += 1
                await ws.send_json(jsonable(out))
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "lm":                     # landmarks from the in-browser tracker
                    rec = _rec_from_packet(data)
                    out = await run_in_threadpool(_process_rec, rec, rcg, session, mode, float(data.get("t", 0.0)), n)
                    n += 1
                    await ws.send_json(jsonable(out))
                    continue
                cmd = data.get("cmd")
                target = session if session is not None else rcg
                if cmd == "reset":
                    target.reset()
                elif cmd == "space":
                    target.text += " "
                elif cmd == "backspace":
                    target.text = target.text[:-1]
    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Video upload jobs
# ─────────────────────────────────────────────────────────────────────────────
JOBS = {}
POOL = ThreadPoolExecutor(max_workers=1)


def _run_job(job_id: str, path: Path, mode: str):
    job = JOBS[job_id]
    out = JOBS_DIR / job_id
    try:
        eng = Engine.load()
        job["status"] = "running"

        def cb(p):
            job["progress"] = round(float(p), 3)

        if mode == "continuous":
            from src.pipeline.continuous import analyze_continuous
            model = Engine.continuous()
            if model is None:
                raise RuntimeError(f"continuous model unavailable ({Engine.cont_error})")
            res = analyze_continuous(path, eng.tracker, out, model=model, progress_cb=cb)
        else:
            res = analyze_static(path, eng.tracker_static, eng.clf, out, progress_cb=cb)
        res["video"] = f"/jobs/{job_id}/{path.name}"
        res["base"] = f"/jobs/{job_id}/"
        (out / "result.json").write_text(json.dumps(res))
        job.update(status="done", progress=1.0, result=res)
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=f"{type(e).__name__}: {e}")


async def upload_video(request: Request):
    form = await request.form()
    up, mode = form.get("file"), form.get("mode", "static")
    if up is None or not hasattr(up, "read"):
        return err("no file")
    if mode not in ("static", "continuous"):
        return err("mode must be static or continuous")
    ext = Path(up.filename or "video.mp4").suffix.lower()
    if ext not in VIDEO_EXT:
        return err("unsupported video type")
    job_id = uuid.uuid4().hex[:10]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    path = d / f"input{ext}"
    size = 0
    with open(path, "wb") as f:
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                return err("file too large (max 300 MB)", 413)
            f.write(chunk)
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "mode": mode}
    POOL.submit(_run_job, job_id, path, mode)
    return JSONResponse({"job": job_id})


async def job_status(request: Request):
    job_id = request.path_params["job_id"]
    job = JOBS.get(job_id)
    if job is None:
        f = JOBS_DIR / re.sub(r"[^a-f0-9]", "", job_id) / "result.json"
        if f.exists():
            return JSONResponse({"status": "done", "progress": 1.0, "result": json.loads(f.read_text())})
        return err("unknown job", 404)
    return JSONResponse(job)


# ─────────────────────────────────────────────────────────────────────────────
# Misc API
# ─────────────────────────────────────────────────────────────────────────────
async def info(request: Request):
    cont = Engine.cont is not None
    return JSONResponse({
        "name": "ISL Fingerspelling",
        "characters": [chr(65 + i) for i in range(26)],
        "one_handed": list("CILOUV"),
        "static_model": "DH-GAT",
        "static_recover": bool(getattr(Engine.clf, "recover", False)),
        "continuous_model": getattr(Engine.cont, "name", None) if cont else None,
        "continuous_available": cont or (ROOT / "checkpoints" / "cont_best.pt").exists(),
        "ready": Engine.tracker is not None,
    })


async def text2sign(request: Request):
    body = await request.json()
    text = str(body.get("text", ""))[:500]
    clean = re.sub(r"[^A-Za-z ]", "", text).upper()
    return JSONResponse({"text": clean, "letters": [c for c in clean if c != " "], "words": clean.split()})


async def model_file(request: Request):
    """the MediaPipe hand model, shared by the server tracker and the in-browser tracker."""
    return FileResponse(ROOT / "hand_landmarker.task", media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=86400"})


async def index(request: Request):
    return FileResponse(STATIC / "index.html")


class RevalidatedFiles(StaticFiles):
    """static files that the browser must revalidate on every load (the pose library changes between versions)."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


DOCS_BUILD = Path(__file__).parent / "docs_build"          # Docusaurus output (npm run build:app in docs-site)
_docs_routes = [Mount("/docs", StaticFiles(directory=DOCS_BUILD, html=True), name="docs")] if DOCS_BUILD.exists() else []

app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/", index),
        Route("/api/info", info),
        Route("/model/hand_landmarker.task", model_file),
        Route("/api/text2sign", text2sign, methods=["POST"]),
        Route("/api/video", upload_video, methods=["POST"]),
        Route("/api/job/{job_id}", job_status),
        WebSocketRoute("/ws/live", ws_live),
        *_docs_routes,
        Mount("/static", RevalidatedFiles(directory=STATIC), name="static"),
        Mount("/jobs", StaticFiles(directory=JOBS_DIR), name="jobs"),
    ],
)


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
