/* 3D ISL avatar: solid two-hand model driven by the pose library (poses.json).
 *
 *  - HandMesh      tapered fingers, rounded palm, forearm + sleeve (not a skeleton)
 *  - blendPose     joint interpolation followed by bone-length re-projection (FK), so
 *                  fingers never shrink or stretch while moving between letters
 *  - Avatar        scene, lighting, orbit camera, letter timeline, contact markers
 */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { RoomEnvironment } from '/static/vendor/RoomEnvironment.js';

const PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19];
const PERIM = [0, 1, 2, 5, 9, 13, 17];                 // palm outline (wrist -> thumb -> pinky)
const SKINS = { light: 0xf0c6a4, medium: 0xd29a72, deep: 0x8a5738 };
const V = (x, y, z) => new THREE.Vector3(x, y, z);

const ease = t => t * t * (3 - 2 * t);
const flat = pts => Float32Array.from(pts.flat());

/* ---------------------------------------------------------------- pose maths */
export function blendPose(A, B, t) {
    const P = new Float32Array(63);
    for (let i = 0; i < 63; i++) P[i] = A[i] + (B[i] - A[i]) * t;
    const out = new Float32Array(63);
    out[0] = P[0]; out[1] = P[1]; out[2] = P[2];
    for (let j = 1; j < 21; j++) {
        const p = PARENT[j];
        let dx = P[3 * j] - P[3 * p], dy = P[3 * j + 1] - P[3 * p + 1], dz = P[3 * j + 2] - P[3 * p + 2];
        let n = Math.hypot(dx, dy, dz);
        if (n < 1e-6) { dx = A[3 * j] - A[3 * p]; dy = A[3 * j + 1] - A[3 * p + 1]; dz = A[3 * j + 2] - A[3 * p + 2]; n = Math.hypot(dx, dy, dz) || 1; }
        const la = Math.hypot(A[3 * j] - A[3 * p], A[3 * j + 1] - A[3 * p + 1], A[3 * j + 2] - A[3 * p + 2]);
        const lb = Math.hypot(B[3 * j] - B[3 * p], B[3 * j + 1] - B[3 * p + 1], B[3 * j + 2] - B[3 * p + 2]);
        const len = la + (lb - la) * t;
        out[3 * j] = out[3 * p] + dx / n * len;
        out[3 * j + 1] = out[3 * p + 1] + dy / n * len;
        out[3 * j + 2] = out[3 * p + 2] + dz / n * len;
    }
    return out;
}
const shift = (P, o) => { const Q = new Float32Array(63); for (let i = 0; i < 63; i++) Q[i] = P[i] + o[i % 3]; return Q; };

/* ------------------------------------------------------------------ HandMesh */
class HandMesh {
    constructor(scene, skin, sleeve, radius) {
        this.group = new THREE.Group(); scene.add(this.group);
        const sph = new THREE.SphereGeometry(1, 28, 18);
        this.joints = radius.map((r, j) => {
            const m = new THREE.Mesh(sph, skin);
            m.scale.setScalar(j === 0 ? 0.24 : r * 1.12); m.castShadow = m.receiveShadow = true;
            this.group.add(m); return m;
        });
        this.bones = PARENT.map((p, j) => {
            if (j === 0 || p === 0) return null;                 // wrist->MCP spans are covered by the palm
            const m = new THREE.Mesh(new THREE.CylinderGeometry(radius[j] * 1.12, radius[p] * 1.12, 1, 24, 1, true), skin);
            m.castShadow = m.receiveShadow = true; this.group.add(m); return m;
        });
        // palm: two rings (top / bottom) + two domed centres, shared vertices -> rounded edges
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(16 * 3), 3));
        const idx = [], n = PERIM.length;
        for (let k = 0; k < n; k++) {
            const a = k, b = (k + 1) % n, c = n + (k + 1) % n, d = n + k;
            idx.push(14, a, b, 15, c, d, a, d, c, a, c, b);
        }
        g.setIndex(idx);
        this.palmGeo = g;
        this.palm = new THREE.Mesh(g, skin);
        this.palm.castShadow = this.palm.receiveShadow = true; this.palm.frustumCulled = false;
        this.group.add(this.palm);
        // forearm (skin) + sleeve (cloth) + cuff
        const cyl = (r0, r1, mat) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, 1, 32, 1), mat); m.castShadow = true; m.receiveShadow = true; this.group.add(m); return m; };
        this.arm = cyl(0.33, 0.37, skin);
        this.sleeve = cyl(0.52, 0.66, sleeve);
        this.cuff = new THREE.Mesh(new THREE.TorusGeometry(0.52, 0.06, 14, 40), sleeve);
        this.cuff.castShadow = true; this.group.add(this.cuff);
        this.up = V(0, 1, 0);
    }

    setSkin(skin) { this.joints.forEach(m => m.material = skin); this.bones.forEach(m => m && (m.material = skin)); this.palm.material = skin; this.arm.material = skin; }

    update(P) {
        const p = j => V(P[3 * j], P[3 * j + 1], P[3 * j + 2]);
        for (let j = 0; j < 21; j++) this.joints[j].position.set(P[3 * j], P[3 * j + 1], P[3 * j + 2]);
        for (let j = 1; j < 21; j++) {
            const m = this.bones[j]; if (!m) continue;
            const a = p(PARENT[j]), b = p(j), d = b.clone().sub(a), len = d.length() || 1e-4;
            m.position.copy(a).addScaledVector(d, 0.5);
            m.quaternion.setFromUnitVectors(this.up, d.divideScalar(len));   // +Y end = child (top radius)
            m.scale.set(1, len, 1);
        }
        // palm
        const c = V(0, 0, 0); PERIM.forEach(j => c.add(p(j))); c.divideScalar(PERIM.length);
        const n = new THREE.Vector3().crossVectors(p(5).sub(p(0)), p(17).sub(p(0))).normalize();
        const pos = this.palmGeo.attributes.position, T = 0.17;
        PERIM.forEach((j, k) => {
            const q = p(j); q.addScaledVector(q.clone().sub(c), 0.16);   // push the outline outward a little
            const t = q.clone().addScaledVector(n, T), b = q.clone().addScaledVector(n, -T);
            pos.setXYZ(k, t.x, t.y, t.z); pos.setXYZ(PERIM.length + k, b.x, b.y, b.z);
        });
        const tc = c.clone().addScaledVector(n, T * 1.35), bc = c.clone().addScaledVector(n, -T * 1.35);
        pos.setXYZ(14, tc.x, tc.y, tc.z); pos.setXYZ(15, bc.x, bc.y, bc.z);
        pos.needsUpdate = true; this.palmGeo.computeVertexNormals();
        // forearm points away from the fingers
        const mc = V(0, 0, 0); [5, 9, 13, 17].forEach(j => mc.add(p(j))); mc.divideScalar(4);
        const w = p(0), dir = w.clone().sub(mc).normalize();
        const place = (mesh, from, to) => {
            mesh.position.copy(w).addScaledVector(dir, (from + to) / 2);
            mesh.quaternion.setFromUnitVectors(this.up, dir); mesh.scale.set(1, to - from, 1);
        };
        place(this.arm, 0.0, 0.95);
        place(this.sleeve, 0.85, 9.0);
        this.cuff.position.copy(w).addScaledVector(dir, 0.86);
        this.cuff.quaternion.setFromUnitVectors(V(0, 0, 1), dir);
    }
}

/* -------------------------------------------------------------------- Avatar */
export class Avatar {
    constructor(canvas, opts = {}) {
        this.canvas = canvas; this.opts = opts;
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: !opts.background, preserveDrawingBuffer: !!opts.preserve });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.05;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.scene = new THREE.Scene();
        if (opts.background) this.scene.background = new THREE.Color(opts.background);
        const pm = new THREE.PMREMGenerator(this.renderer);
        this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
        this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
        this.camera.position.set(0, 0.5, 8.4);
        const key = new THREE.DirectionalLight(0xfff2e6, 2.6); key.position.set(4, 7, 8); key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048); key.shadow.radius = 5; key.shadow.bias = -0.0004;
        Object.assign(key.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 1, far: 30 });
        this.scene.add(key);
        const rim = new THREE.DirectionalLight(0x8fb4ff, 1.3); rim.position.set(-6, 3, -5); this.scene.add(rim);
        this.scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x241a2e, 0.55));
        if (!opts.noControls) {
            this.controls = new OrbitControls(this.camera, canvas);
            this.controls.target.set(0, -0.7, 0); this.controls.enableDamping = true; this.controls.minDistance = 4; this.controls.maxDistance = 22;
            this.controls.maxPolarAngle = Math.PI * 0.62; this.controls.update();
        } else this.camera.lookAt(0, -0.7, 0);

        this.skin = new THREE.MeshStandardMaterial({ color: SKINS.medium, roughness: 0.52, metalness: 0.0, envMapIntensity: 0.9 });
        this.sleeve = new THREE.MeshStandardMaterial({ color: 0x2c3a6b, roughness: 0.85, metalness: 0.0 });
        this.useTouched = true; this.showSkeleton = false; this.showMarkers = true;
        this.cur = null; this.seq = null; this.t0 = 0; this.speed = 1;
        this.marks = [];
        this.running = false;
    }

    async load(url = '/static/data/poses.json') {
        const d = await (await fetch(url)).json();
        this.poses = d.letters; this.meta = d.meta;
        this.hands = [0, 1].map(() => new HandMesh(this.scene, this.skin, this.sleeve, this.meta.radius));
        // skeleton overlay
        const mk = () => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(40 * 6), 3));
            return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x5eead4, depthTest: false, transparent: true })); };
        this.skel = [mk(), mk()]; this.skel.forEach(s => { s.visible = false; s.renderOrder = 10; s.frustumCulled = false; this.scene.add(s); });
        const mg = new THREE.SphereGeometry(0.075, 16, 12);
        for (let i = 0; i < 4; i++) {
            const m = new THREE.Mesh(mg, new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.95, depthTest: false }));
            m.visible = false; m.renderOrder = 11; this.scene.add(m); this.marks.push(m);
        }
        this.cur = this.restPose();
        this.apply(this.cur);
        return this;
    }

    setSkin(name) { this.skin.color.setHex(SKINS[name] || SKINS.medium); }

    /* --- poses --- */
    restPose() {
        const r = this.poses._rest.raw;
        return [shift(flat(r[0]), [-2.6, -2.9, 0]), shift(flat(r[1]), [2.6, -2.9, 0])];
    }
    info(letter) { return this.poses[letter]; }
    poseFor(letter) {
        const L = this.poses[letter]; if (!L) return this.restPose();
        const arr = this.useTouched ? L.touched : L.raw;
        if (L.hands === 2) return [shift(flat(arr[0]), [0, -0.4, 0]), shift(flat(arr[1]), [0, -0.4, 0])];
        return [this.restPose()[0], shift(flat(arr[0]), [1.0, -0.6, 0])];      // one-handed: dominant hand on the right
    }
    contactsFor(letter, pose) {
        const L = this.poses[letter]; if (!L || L.hands !== 2 || !this.useTouched) return [];
        return L.contacts.map(c => V(
            (pose[0][3 * c.a] + pose[1][3 * c.b]) / 2, (pose[0][3 * c.a + 1] + pose[1][3 * c.b + 1]) / 2,
            (pose[0][3 * c.a + 2] + pose[1][3 * c.b + 2]) / 2));
    }

    apply(pose, marks = []) {
        this.cur = pose;
        this.hands.forEach((h, i) => h.update(pose[i]));
        this.skel.forEach((s, i) => {
            s.visible = this.showSkeleton; if (!s.visible) return;
            const a = s.geometry.attributes.position; let k = 0;
            for (let j = 1; j < 21; j++) { const p = PARENT[j]; a.setXYZ(k++, pose[i][3 * p], pose[i][3 * p + 1], pose[i][3 * p + 2]); a.setXYZ(k++, pose[i][3 * j], pose[i][3 * j + 1], pose[i][3 * j + 2]); }
            a.needsUpdate = true; s.geometry.setDrawRange(0, 40);
        });
        this.marks.forEach((m, i) => { const v = this.showMarkers ? marks[i] : null; m.visible = !!v; if (v) m.position.copy(v); });
    }

    /* --- timeline --- */
    play(text, { speed = 1, onItem = () => {}, onDone = () => {} } = {}) {
        const rest = this.restPose();
        const clean = text.toUpperCase().replace(/[^A-Z ]/g, '');
        const items = []; let prev = null;
        [...clean].forEach((ch, i) => {
            if (ch === ' ') { items.push({ ch, pose: rest, trans: 0.55, hold: 0.35, idx: i }); prev = null; return; }
            const target = this.poseFor(ch);
            if (prev === ch) {        // repeated letter: relax slightly toward rest so the repeat is visible
                const lift = [blendPose(target[0], rest[0], 0.28), blendPose(target[1], rest[1], 0.28)];
                items.push({ ch: null, pose: lift, trans: 0.28, hold: 0.0, idx: i, lift: true });
            }
            items.push({ ch, pose: target, trans: 0.5, hold: 0.85, idx: i });
            prev = ch;
        });
        items.push({ ch: null, pose: rest, trans: 0.7, hold: 0, idx: clean.length, end: true });
        let t = 0; items.forEach(it => { it.start = t; t += (it.trans + it.hold) / speed; it.end_t = t; });
        this.seq = { items, from: this.cur, total: t, speed, onItem, onDone, fired: -1, startTime: performance.now() / 1000 };
        this.start();
    }
    stop(toRest = true) { this.seq = null; if (toRest && this.poses) this.apply(this.restPose()); }

    showLetter(ch) { this.seq = null; const p = this.poseFor(ch); this.apply(p, this.contactsFor(ch, p)); }

    tick(now) {
        const s = this.seq;
        if (s) {
            const el = now - s.startTime;
            if (el >= s.total) { const last = s.items[s.items.length - 1]; this.apply(last.pose); const d = s.onDone; this.seq = null; d(); }
            else {
                let k = s.items.findIndex(it => el < it.end_t);
                const it = s.items[k], prevPose = k === 0 ? s.from : s.items[k - 1].pose;
                const lt = (el - it.start) * s.speed, a = Math.min(1, lt / it.trans);
                const pose = [blendPose(prevPose[0], it.pose[0], ease(a)), blendPose(prevPose[1], it.pose[1], ease(a))];
                const holding = a >= 1 && it.ch && it.ch !== ' ';
                this.apply(pose, holding ? this.contactsFor(it.ch, it.pose) : []);
                if (k !== s.fired && !it.lift) { s.fired = k; s.onItem(it, k); }
            }
        }
        // gentle idle float so the hands feel alive
        const f = Math.sin(now * 1.4) * 0.02;
        this.hands.forEach(h => { h.group.position.y = f; });
        const pulse = 0.85 + 0.25 * Math.sin(now * 6);
        this.marks.forEach(m => m.scale.setScalar(pulse));
    }

    resize() {
        const w = this.canvas.clientWidth || 640, h = this.canvas.clientHeight || 480;
        if (this.canvas.width !== Math.floor(w * this.renderer.getPixelRatio()) || this.canvas.height !== Math.floor(h * this.renderer.getPixelRatio())) {
            this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
        }
    }
    render() { this.resize(); this.controls && this.controls.update(); this.renderer.render(this.scene, this.camera); }
    start() { if (this.running) return; this.running = true; const loop = () => { if (!this.running) return; this.tick(performance.now() / 1000); this.render(); requestAnimationFrame(loop); }; requestAnimationFrame(loop); }
    pause() { this.running = false; }
}

/* one shared off-screen avatar renders the alphabet thumbnails */
export async function makeThumbnails(letters, skin = 'medium') {
    const c = document.createElement('canvas'); c.width = c.height = 320; c.style.cssText = 'width:320px;height:320px;position:fixed;left:-9999px';
    document.body.appendChild(c);
    const av = new Avatar(c, { background: 0x0e1530, noControls: true, preserve: true });
    av.renderer.setPixelRatio(1); av.renderer.setSize(320, 320, false);
    await av.load(); av.setSkin(skin); av.camera.aspect = 1; av.camera.position.set(0, 0.2, 7.6); av.camera.lookAt(0, -0.75, 0); av.camera.updateProjectionMatrix();
    const out = {};
    for (const L of letters) { av.showLetter(L); av.renderer.render(av.scene, av.camera); out[L] = c.toDataURL('image/jpeg', 0.85); }
    av.renderer.dispose(); c.remove();
    return out;
}
