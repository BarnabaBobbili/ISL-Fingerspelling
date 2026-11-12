/* The 3D signer: two skinned hands with cloth sleeves.
 *
 *  - HandRig (signer/rig.js)     a skinned hand mesh driven by the 21 landmarks of each hand
 *  - Sleeves (signer/sleeves.js) a cloth forearm behind each hand
 *  - blendPose                   joint interpolation + bone-length re-projection (forward kinematics),
 *                                so fingers never shrink or stretch while moving between letters
 *  - Avatar                      scene, lighting, camera presets, the letter timeline, contact markers
 *
 * Each pose is shown exactly as the landmarks give it: the left- or right-hand mesh is chosen from the
 * hand's real geometry (pose.chir), nothing is mirrored. A hand that is not needed rests BELOW the
 * frame and rises when the next letter needs it.
 */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { RoomEnvironment } from '/static/vendor/RoomEnvironment.js';
import { HandRig, loadHandModel } from '/static/js/signer/rig.js';
import { Sleeves } from '/static/js/signer/sleeves.js';

const PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19];
const SKINS = { light: 0xf0c6a4, medium: 0xcf9870, deep: 0x8a5738 };
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const ease = t => t * t * (3 - 2 * t);
const flat = pts => Float32Array.from(pts.flat());

/* where the signing space sits, and where a resting hand waits (below the frame) */
const SPACE_TWO = [0, -3.3, 2.0];
const SPACE_ONE = [0.1, -4.1, 2.0];
const WAIT = [[-2.2, -11.5, 0.8], [2.2, -11.5, 0.8]];

const VIEWS = {   // position, target, fov
    front: { p: [0, -3.0, 12.2], t: [0, -3.3, 2.0], fov: 36 },
    close: { p: [0, -3.1, 8.8], t: [0, -3.3, 2.0], fov: 36 },
    above: { p: [0, 5.2, 9.5], t: [0, -3.6, 2.2], fov: 36 },
    thumb: { p: [-0.1, -3.0, 11.2], t: [-0.1, -3.2, 2.0], fov: 36 },
};

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
const place = (P, wrist) => {                         // translate so the wrist sits at `wrist`
    const Q = new Float32Array(63);
    for (let j = 0; j < 21; j++) { Q[3 * j] = P[3 * j] - P[0] + wrist[0]; Q[3 * j + 1] = P[3 * j + 1] - P[1] + wrist[1]; Q[3 * j + 2] = P[3 * j + 2] - P[2] + wrist[2]; }
    return Q;
};
const withChir = (pose, chir) => { pose.chir = chir; return pose; };

/* -------------------------------------------------------------------- Avatar */
export class Avatar {
    constructor(canvas, opts = {}) {
        this.canvas = canvas; this.opts = opts;
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: !opts.background, preserveDrawingBuffer: !!opts.preserve });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.0;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.scene = new THREE.Scene();
        if (opts.background) this.scene.background = new THREE.Color(opts.background);
        const pm = new THREE.PMREMGenerator(this.renderer);
        this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
        this.camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
        const key = new THREE.DirectionalLight(0xfff1e0, 2.4); key.position.set(5, 8, 10); key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048); key.shadow.radius = 5; key.shadow.bias = -0.0004;
        Object.assign(key.shadow.camera, { left: -9, right: 9, top: 9, bottom: -9, near: 1, far: 40 });
        this.scene.add(key);
        const rim = new THREE.DirectionalLight(0x9fc4ff, 1.1); rim.position.set(-7, 4, -6); this.scene.add(rim);
        this.scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x2a2030, 0.5));
        if (!opts.noControls) {
            this.controls = new OrbitControls(this.camera, canvas);
            this.controls.enableDamping = true; this.controls.minDistance = 3; this.controls.maxDistance = 26;
            this.controls.maxPolarAngle = Math.PI * 0.62; this.controls.enablePan = false;
        }
        this.mats = {
            skin: new THREE.MeshStandardMaterial({ color: SKINS.medium, roughness: 0.5, metalness: 0.0, envMapIntensity: 0.9 }),
            cloth: new THREE.MeshStandardMaterial({ color: 0x0e5a5f, roughness: 0.85, metalness: 0.0 }),
            trim: new THREE.MeshStandardMaterial({ color: 0xd9962a, roughness: 0.6, metalness: 0.1 }),
            hair: new THREE.MeshStandardMaterial({ color: 0x15110f, roughness: 0.55, metalness: 0.0 }),
        };
        this.useTouched = true; this.showSkeleton = false; this.showMarkers = true;
        this.cur = null; this.seq = null; this.speed = 1; this.marks = []; this.running = false;
        this.view = 'front';
    }

    async load(url = '/static/data/poses.json') {
        const [d, ...models] = await Promise.all([fetch(url).then(r => r.json()), loadHandModel('left'), loadHandModel('right'), loadHandModel('left'), loadHandModel('right')]);
        this.poses = d.letters; this.meta = d.meta;
        // every slot owns a right-type and a left-type hand; the one matching the pose's real geometry is shown
        const type = rg => (rg.handed > 0 ? 1 : -1);
        this.rigs = [0, 1].map(i => {
            const pair = [new HandRig(models[2 * i], this.mats.skin), new HandRig(models[2 * i + 1], this.mats.skin)];
            const byType = {}; pair.forEach(rg => { byType[type(rg)] = rg; this.scene.add(rg.root); });
            return byType;
        });
        this.sleeves = new Sleeves(this.scene, this.mats);
        const mk = () => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(40 * 6), 3));
            return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x62beb5, depthTest: false, transparent: true })); };
        this.skel = [mk(), mk()]; this.skel.forEach(s => { s.visible = false; s.renderOrder = 10; s.frustumCulled = false; this.scene.add(s); });
        const mg = new THREE.SphereGeometry(0.075, 16, 12);
        for (let i = 0; i < 4; i++) {
            const m = new THREE.Mesh(mg, new THREE.MeshBasicMaterial({ color: 0xefb03a, transparent: true, opacity: 0.95, depthTest: false }));
            m.visible = false; m.renderOrder = 11; this.scene.add(m); this.marks.push(m);
        }
        this.setView('front');
        this.cur = this.restPose();
        this.apply(this.cur);
        return this;
    }

    setSkin(name) { this.mats.skin.color.setHex(SKINS[name] || SKINS.medium); }

    /** camera presets: 'front' | 'close' | 'above' */
    setView(name) {
        const v = VIEWS[name] || VIEWS.front; this.view = name;
        this.camera.position.set(...v.p); this.camera.fov = v.fov; this.camera.updateProjectionMatrix();
        if (this.controls) { this.controls.target.set(...v.t); this.controls.update(); } else this.camera.lookAt(...v.t);
    }

    /* --- poses (world coordinates; slot 0 = the hand on the viewer's left) --- */
    restPose() {
        const r = this.poses._rest.raw;                         // [left-type, right-type] relative to the wrist
        return withChir([place(flat(r[1]), WAIT[0]), place(flat(r[0]), WAIT[1])], [1, -1]);      // waiting below the frame
    }
    info(letter) { return this.poses[letter]; }
    poseFor(letter) {
        const L = this.poses[letter]; if (!L) return this.restPose();
        if (L.keys && L.keys.length) return this.keysFor(letter)[L.keys.length >> 1];            // a stroke: show its middle frame
        const arr = this.useTouched ? L.touched : L.raw;
        if (L.hands === 2) return withChir([shift(flat(arr[0]), SPACE_TWO), shift(flat(arr[1]), SPACE_TWO)], L.chir);
        return withChir([shift(flat(arr[0]), SPACE_ONE), this.restPose()[1]], [L.chir[0], -1]);   // one hand signs, the other waits below
    }
    /** letters that are a movement (H, J): the poses the hands pass through, or null */
    keysFor(letter) {
        const L = this.poses[letter]; if (!L || !L.keys || !L.keys.length) return null;
        return L.keys.map(k => withChir([shift(flat(k[0]), SPACE_TWO), shift(flat(k[1]), SPACE_TWO)], L.chir));
    }
    strokeAt(keys, u) {                                  // piecewise-linear position along the key poses, u in [0, 1]
        const x = Math.min(1, Math.max(0, u)) * (keys.length - 1), i = Math.min(keys.length - 2, Math.floor(x)), f = x - i;
        return withChir([blendPose(keys[i][0], keys[i + 1][0], f), blendPose(keys[i][1], keys[i + 1][1], f)], keys[0].chir);
    }
    contactsFor(letter, pose) {
        const L = this.poses[letter]; if (!L || L.hands !== 2 || !this.useTouched) return [];
        return L.contacts.map(c => c.pt ? V(c.pt[0] + SPACE_TWO[0], c.pt[1] + SPACE_TWO[1], c.pt[2] + SPACE_TWO[2]) : V(
            (pose[0][3 * c.a] + pose[1][3 * c.b]) / 2, (pose[0][3 * c.a + 1] + pose[1][3 * c.b + 1]) / 2,
            (pose[0][3 * c.a + 2] + pose[1][3 * c.b + 2]) / 2));
    }

    apply(pose, marks = []) {
        this.cur = pose;
        const chir = pose.chir || [1, -1];
        pose.forEach((P, i) => {
            const on = chir[i] > 0 ? 1 : -1;
            this.rigs[i][on].root.visible = true; this.rigs[i][-on].root.visible = false;
            this.rigs[i][on].pose(P);
        });
        this.sleeves.update(pose);
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
            if (ch === ' ') { items.push({ ch, pose: rest, trans: 0.6, hold: 0.3, idx: i }); prev = null; return; }
            const target = this.poseFor(ch);
            if (prev === ch) {        // repeated letter: relax slightly toward rest so the repeat is visible
                const lift = withChir([blendPose(target[0], rest[0], 0.28), blendPose(target[1], rest[1], 0.28)], target.chir);
                items.push({ ch: null, pose: lift, trans: 0.28, hold: 0.0, idx: i, lift: true });
            }
            const keys = this.keysFor(ch);
            if (keys) items.push({ ch, pose: keys[0], out: keys[keys.length - 1], keys, trans: 0.55, hold: 1.5, idx: i });   // a stroke
            else items.push({ ch, pose: target, trans: 0.55, hold: 0.85, idx: i });
            prev = ch;
        });
        items.push({ ch: null, pose: rest, trans: 0.8, hold: 0, idx: clean.length, end: true });
        let t = 0; items.forEach(it => { it.start = t; t += (it.trans + it.hold) / speed; it.end_t = t; });
        this.seq = { items, from: this.cur, total: t, speed, onItem, onDone, fired: -1, startTime: performance.now() / 1000 };
        this.start();
    }
    stop(toRest = true) { this.seq = null; this.demo = null; if (toRest && this.poses) this.apply(this.restPose()); }

    showLetter(ch) {
        this.seq = null; this.demo = null;
        const p = this.poseFor(ch); this.apply(p, this.contactsFor(ch, p));
        const keys = this.keysFor(ch);
        if (keys && !this.still) { this.demo = { keys, t0: performance.now() / 1000 }; this.start(); }    // a movement repeats until another letter is chosen
    }

    tick(now) {
        const s = this.seq;
        if (this.demo && !s) {
            const ph = (((now - this.demo.t0) % 3.0) / 3.0), u = Math.min(1, ph / 0.75);
            this.apply(this.strokeAt(this.demo.keys, u));
        }
        if (s) {
            const el = now - s.startTime;
            if (el >= s.total) { const last = s.items[s.items.length - 1]; this.apply(last.pose); const d = s.onDone; this.seq = null; d(); }
            else {
                const k = s.items.findIndex(it => el < it.end_t);
                const it = s.items[k], prevPose = k === 0 ? s.from : (s.items[k - 1].out || s.items[k - 1].pose);
                const lt = (el - it.start) * s.speed, a = Math.min(1, lt / it.trans);
                let pose = withChir([blendPose(prevPose[0], it.pose[0], ease(a)), blendPose(prevPose[1], it.pose[1], ease(a))],
                    a < 0.5 ? (prevPose.chir || [1, -1]) : (it.pose.chir || [1, -1]));
                const holding = a >= 1 && it.ch && it.ch !== ' ';
                if (holding && it.keys) pose = this.strokeAt(it.keys, Math.min(1, (lt - it.trans) / (it.hold * 0.8)));      // the movement itself
                this.apply(pose, holding && !it.keys ? this.contactsFor(it.ch, it.pose) : []);
                if (k !== s.fired && !it.lift) { s.fired = k; s.onItem(it, k); }
            }
        }
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
    const av = new Avatar(c, { background: 0x1a2429, noControls: true, preserve: true });
    av.still = true; av.renderer.setPixelRatio(1); av.renderer.setSize(320, 320, false);
    await av.load(); av.setSkin(skin); av.camera.aspect = 1; av.setView('thumb'); av.camera.aspect = 1; av.camera.updateProjectionMatrix();
    const out = {};
    for (const L of letters) { av.showLetter(L); av.renderer.render(av.scene, av.camera); out[L] = c.toDataURL('image/jpeg', 0.85); }
    av.renderer.dispose(); c.remove();
    return out;
}
