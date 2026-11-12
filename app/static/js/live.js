/* Live webcam: camera -> JPEG -> WebSocket -> recognition state -> HUD on top of the video. */
import { drawHud, drawSkeleton, containRect } from '/static/js/hud.js';

export class Live {
    constructor({ canvas, msg, onText, onMetrics }) {
        this.c = canvas; this.ctx = canvas.getContext('2d'); this.msg = msg;
        this.onText = onText; this.onMetrics = onMetrics;
        this.video = document.createElement('video'); this.video.playsInline = true; this.video.muted = true;
        this.grab = document.createElement('canvas');
        this.state = { status: 'no_hand', text: '' }; this.hands = [];
        this.mirror = true; this.skeleton = true; this.running = false; this.mode = 'static';
        this.fps = 0; this.frames = 0; this.t0 = performance.now(); this.busy = false;
    }

    async start(mode) {
        this.mode = mode;
        this.msg.textContent = 'Opening camera…'; this.msg.classList.remove('hidden');
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
        } catch (e) { this.msg.textContent = 'Camera unavailable: ' + e.message; throw e; }
        this.video.srcObject = this.stream; await this.video.play();
        const vw = this.video.videoWidth, vh = this.video.videoHeight, sc = Math.min(1, 960 / vw);
        this.grab.width = Math.round(vw * sc); this.grab.height = Math.round(vh * sc);
        await this.connect();
        this.running = true; this.msg.classList.add('hidden');
        this.draw(); this.send();
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
                this.state = d; this.hands = d.hands || []; this.busy = false; this.frames++;
                this.onText && this.onText(d.text || '');
                const now = performance.now();
                if (now - this.t0 > 1000) { this.fps = this.frames * 1000 / (now - this.t0); this.frames = 0; this.t0 = now; }
                this.onMetrics && this.onMetrics({ fps: this.fps, ms: d.ms, hands: this.hands.length });
                this.send();
            };
        });
    }

    send() {                                   // pull-based: next frame only after the previous reply
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
