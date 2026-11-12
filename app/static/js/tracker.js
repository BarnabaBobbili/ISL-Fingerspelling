/* In-browser two-hand tracker (MediaPipe Tasks Vision, same hand_landmarker.task as the server).
 *
 * Running the tracker in the browser removes the JPEG upload + server-side detection from the
 * critical path: the server only receives ~1 KB of landmarks per frame, so live recognition runs
 * at the camera's frame rate (≈30 fps) instead of ≈10 fps, which is what the continuous model
 * was trained on.
 *
 * Output format is identical to src/pipeline/hands.py: hands ordered by wrist-x (image-left
 * first), normalised image coordinates, a lone hand in slot 0.
 */
import { FilesetResolver, HandLandmarker } from '/static/vendor/mediapipe/vision_bundle.mjs';

const WASM = '/static/vendor/mediapipe/wasm';
const MODEL = '/model/hand_landmarker.task';

const hullArea = pts => { let a = 0; for (let i = 0; i < pts.length; i++) { const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length]; a += x1 * y2 - x2 * y1; } return a / 2; };

function convexHull(points) {                       // Andrew's monotone chain, points = [[x,y],...]
    const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = []; for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    const up = []; for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    lo.pop(); up.pop(); return lo.concat(up);
}

export class BrowserTracker {
    /** mode: 'video' (tracking between frames, fastest) | 'image' (full detection each frame, matches training exactly) */
    constructor({ mode = 'video', recover = false, delegate = 'GPU', minConf = 0.4 } = {}) {
        this.mode = mode; this.recover = recover; this.delegate = delegate; this.minConf = minConf; this.ready = false;
    }

    async init() {
        const fileset = await FilesetResolver.forVisionTasks(WASM);
        const make = async (runningMode, delegate) => HandLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: MODEL, delegate }, runningMode, numHands: 2,
            minHandDetectionConfidence: this.minConf, minHandPresenceConfidence: this.minConf, minTrackingConfidence: this.minConf });
        const rm = this.mode === 'image' ? 'IMAGE' : 'VIDEO';
        try { this.lm = await make(rm, this.delegate); this.usedDelegate = this.delegate; }
        catch (e) { this.lm = await make(rm, 'CPU'); this.usedDelegate = 'CPU'; }
        if (this.recover) {                                        // second-hand recovery needs per-image detection
            this.lmImg = rm === 'IMAGE' ? this.lm : await make('IMAGE', this.usedDelegate);
            this.cv = document.createElement('canvas'); this.cx = this.cv.getContext('2d', { willReadFrequently: false });
        }
        this.lastTs = -1; this.ready = true; return this;
    }

    static order(dets) {                                           // dets: [[[x,y,z]*21], ...]
        const d = dets.slice().sort((a, b) => a[0][0] - b[0][0]);
        return d.slice(0, 2);
    }

    /** Blank out the hand that was found and look for a second one (see src/static/extract_v2.py). */
    secondHand(video, first) {
        const w = video.videoWidth, h = video.videoHeight;
        if (this.cv.width !== w || this.cv.height !== h) { this.cv.width = w; this.cv.height = h; }
        const cx = this.cx; cx.drawImage(video, 0, 0, w, h);
        const pts = first.map(p => [p[0] * w, p[1] * h]);
        let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
        const pad = 0.12 * Math.max(x1 - x0, y1 - y0), hull = convexHull(pts);
        const s = cx.getImageData(Math.floor(w * 0.45), Math.floor(h * 0.05), 8, 8).data;       // background colour sample
        cx.fillStyle = `rgb(${s[0]},${s[1]},${s[2]})`; cx.strokeStyle = cx.fillStyle; cx.lineJoin = 'round'; cx.lineWidth = 2 * pad;
        cx.beginPath(); hull.forEach(([x, y], i) => (i ? cx.lineTo(x, y) : cx.moveTo(x, y))); cx.closePath(); cx.fill(); cx.stroke();
        const r = this.lmImg.detect(this.cv);
        for (const l of r.landmarks) {
            const wx = l[0].x * w, wy = l[0].y * h;
            // accept only if the new wrist lies outside the blanked area (distance to hull edges > pad)
            let inside = true, minD = 1e9;
            for (let i = 0; i < hull.length; i++) {
                const [ax, ay] = hull[i], [bx, by] = hull[(i + 1) % hull.length];
                const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1;
                const side = (dx * (wy - ay) - dy * (wx - ax)) / L; if (side < 0) inside = false;
                const t = Math.max(0, Math.min(1, ((wx - ax) * dx + (wy - ay) * dy) / (L * L)));
                minD = Math.min(minD, Math.hypot(wx - (ax + t * dx), wy - (ay + t * dy)));
            }
            if (!inside || minD > pad) return l.map(p => [p.x, p.y, p.z]);
        }
        return null;
    }

    /** video: HTMLVideoElement, tMs: monotonically increasing timestamp. Returns {hands, ms} */
    detect(video, tMs) {
        const t0 = performance.now();
        let res;
        if (this.mode === 'image') res = this.lm.detect(video);
        else { if (tMs <= this.lastTs) tMs = this.lastTs + 1; this.lastTs = tMs; res = this.lm.detectForVideo(video, tMs); }
        let dets = res.landmarks.map(l => l.map(p => [p.x, p.y, p.z]));
        if (this.recover && dets.length === 1) { const extra = this.secondHand(video, dets[0]); if (extra) dets.push(extra); }
        return { hands: BrowserTracker.order(dets), ms: performance.now() - t0 };
    }
}
