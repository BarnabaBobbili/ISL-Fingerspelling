/* Live webcam.
 *
 * Fast path : camera -> in-browser MediaPipe tracker (tracker.js) -> ~1 KB landmark packet per
 *             frame over a WebSocket -> recognition state -> HUD on top of the video.
 *             Runs at the camera frame rate (~30 fps); the skeleton is drawn from the LOCAL
 *             tracker result, so it never lags behind the picture.
 * Fallback  : if the browser tracker cannot start, JPEG frames are sent and the server tracks.
 */
import { drawHud, drawSkeleton, containRect } from '/static/js/hud.js';
import { BrowserTracker } from '/static/js/tracker.js';

const MAX_IN_FLIGHT = 4;

export class Live {
    constructor({ canvas, msg, onText, onMetrics }) {
        this.c = canvas; this.ctx = canvas.getContext('2d'); this.msg = msg;
        this.onText = onText; this.onMetrics = onMetrics;
        this.video = document.createElement('video'); this.video.playsInline = true; this.video.muted = true;
        this.grab = document.createElement('canvas');
        this.state = { status: 'no_hand', text: '' }; this.hands = [];
        this.mirror = true; this.skeleton = true; this.running = false; this.mode = 'static';
        this.reset();
    }

    reset() {
        this.fps = 0; this.frames = 0; this.t0 = performance.now(); this.busy = false;
        this.sentAt = new Map(); this.rtt = 0; this.trackMs = 0; this.n = 0; this.inFlight = 0;
    }

    async start(mode, opts = {}) {
        this.mode = mode; this.reset();
        this.msg.textContent = 'Opening camera…'; this.msg.classList.remove('hidden');
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 }, facingMode: 'user' }, audio: false });
        } catch (e) { this.msg.textContent = 'Camera unavailable: ' + e.message; throw e; }
        this.video.srcObject = this.stream; await this.video.play();
        this.tracker = null; this.path = 'jpeg';
        if (opts.browserTracking !== false) {
            try {
                this.msg.textContent = 'Loading hand tracker…';
                const info = await (await fetch('/api/info')).json();
                this.tracker = await new BrowserTracker({ mode: opts.trackerMode || 'video', recover: mode === 'static' && !!info.static_recover }).init();
                this.path = 'browser';
            } catch (e) { console.warn('browser tracker unavailable, falling back to server tracking', e); this.tracker = null; }
        }
        const vw = this.video.videoWidth, vh = this.video.videoHeight, sc = Math.min(1, 960 / vw);
        this.grab.width = Math.round(vw * sc); this.grab.height = Math.round(vh * sc);
        await this.connect();
        this.running = true; this.msg.classList.add('hidden');
        this.draw();
        if (this.tracker) this.trackLoop(); else this.sendJpeg();
        return this.path;
    }

    connect() {
        return new Promise((res, rej) => {
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            this.ws = new WebSocket(`${proto}://${location.host}/ws/live?mode=${this.mode}`);
            this.ws.binaryType = 'arraybuffer';
            this.ws.onopen = () => res();
            this.ws.onerror = () => rej(new Error('websocket error'));
            this.ws.onclose = () => { if (this.running) { this.msg.classList.remove('hidden'); this.msg.textContent = 'Connection closed'; } };
            this.ws.onmessage = ev => {
                const d = JSON.parse(ev.data);
                if (d.error) { this.msg.classList.remove('hidden'); this.msg.textContent = d.error; this.stop(); return; }
                this.state = d; this.inFlight = Math.max(0, this.inFlight - 1); this.busy = false;
                if (!this.tracker) this.hands = d.hands || [];
                const sent = this.sentAt.get(d.seq); if (sent !== undefined) { this.rtt = 0.8 * this.rtt + 0.2 * (performance.now() - sent) || (performance.now() - sent); this.sentAt.delete(d.seq); }
                this.onText && this.onText(d.text || '', d.tentative || '');
                this.frames++;
                const now = performance.now();
                if (now - this.t0 > 1000) { this.fps = this.frames * 1000 / (now - this.t0); this.frames = 0; this.t0 = now; }
                this.onMetrics && this.onMetrics({ fps: this.fps, ms: this.tracker ? this.rtt : d.ms, hands: this.hands.length, path: this.path, trackMs: this.trackMs });
                if (!this.tracker) this.sendJpeg();
            };
        });
    }

    /* ---------------------------------------------------------- browser-tracking path */
    trackLoop() {
        const v = this.video;
        const step = (now) => {
            if (!this.running) return;
            const r = this.tracker.detect(v, now);
            this.hands = r.hands.map(h => h.map(p => [p[0], p[1]]));
            this.trackMs = 0.85 * this.trackMs + 0.15 * r.ms;
            if (this.ws.readyState === 1 && this.inFlight < MAX_IN_FLIGHT) {
                this.sentAt.set(this.n, performance.now()); this.inFlight++;
                this.ws.send(JSON.stringify({ type: 'lm', t: now / 1000, w: v.videoWidth, h: v.videoHeight,
                    hands: r.hands.map(h => h.map(p => [+p[0].toFixed(5), +p[1].toFixed(5), +p[2].toFixed(5)])) }));
                this.n++;
            }
            if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step); else requestAnimationFrame(step);
        };
        if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step); else requestAnimationFrame(step);
    }

    /* ---------------------------------------------------------- JPEG fallback path */
    sendJpeg() {                               // pull-based: next frame only after the previous reply
        if (!this.running || this.busy || !this.ws || this.ws.readyState !== 1) return;
        this.busy = true;
        const g = this.grab.getContext('2d'); g.drawImage(this.video, 0, 0, this.grab.width, this.grab.height);
        this.grab.toBlob(b => b && b.arrayBuffer().then(buf => this.ws.readyState === 1 && this.ws.send(buf)), 'image/jpeg', 0.82);
    }

    cmd(name) { this.ws && this.ws.readyState === 1 && this.ws.send(JSON.stringify({ cmd: name })); }

    draw() {
        if (!this.running) return;
        const { ctx, c, video } = this, W = c.width, H = c.height;
        ctx.fillStyle = '#05070f'; ctx.fillRect(0, 0, W, H);
        const r = containRect(video.videoWidth || 16, video.videoHeight || 9, W, H);
        ctx.save();
        if (this.mirror) { ctx.translate(r.x * 2 + r.w, 0); ctx.scale(-1, 1); }
        ctx.drawImage(video, r.x, r.y, r.w, r.h);
        ctx.restore();
        if (this.skeleton) drawSkeleton(ctx, this.hands, r, this.mirror);
        drawHud(ctx, W, H, this.state);
        requestAnimationFrame(() => this.draw());
    }

    stop() {
        this.running = false;
        this.stream && this.stream.getTracks().forEach(t => t.stop());
        this.ws && this.ws.close();
        this.msg.classList.remove('hidden'); if (!this.msg.textContent.startsWith('Camera') && !this.msg.textContent.includes('unavailable')) this.msg.innerHTML = 'Camera stopped. Press <b>Start camera</b> to begin';
    }
}
