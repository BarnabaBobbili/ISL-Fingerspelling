/* The 3D signer.
 *
 *  - Body (signer/body.js)   upper-body figure in a kurta, arms solved by two-bone IK
 *  - HandRig (signer/rig.js) a skinned hand mesh driven by the 21 landmarks of each hand
 *  - blendPose               joint interpolation + bone-length re-projection (forward kinematics),
 *                            so fingers never shrink or stretch while moving between letters
 *  - Avatar                  scene, lighting, camera presets, the letter timeline, contact markers
 *
 * Slot 0 is the signer's RIGHT hand (the viewer's left); slot 1 the left hand. One-handed letters are
 * signed with the right hand; the other arm hangs relaxed at the side.
 */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';
import { RoomEnvironment } from '/static/vendor/RoomEnvironment.js';
import { HandRig, loadHandModel } from '/static/js/signer/rig.js';
import { Body } from '/static/js/signer/body.js';

const PARENT = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19];
const SKINS = { light: 0xf0c6a4, medium: 0xcf9870, deep: 0x8a5738 };
const V = (x, y, z) => new THREE.Vector3(x, y, z);
const ease = t => t * t * (3 - 2 * t);
const flat = pts => Float32Array.from(pts.flat());

/* where the signing space sits in front of the chest, and where a resting hand hangs */
const SPACE_TWO = [0, -3.2, 2.5];
const SPACE_ONE = [-1.6, -2.7, 2.7];
const HANG = [[-2.55, -5.9, 0.9], [2.55, -5.9, 0.9]];

const VIEWS = {   // position, target, fov, head visible
    front: { p: [0, -0.8, 19.6], t: [0, -1.7, 0], fov: 36, head: true },
    hands: { p: [0, -2.6, 11.0], t: [0, -3.4, 1.8], fov: 36, head: true },
    thumb: { p: [-0.5, -2.6, 12.2], t: [-0.5, -3.2, 1.8], fov: 36, head: true },
    signer: { p: [0, 2.5, 1.8], t: [0, -3.4, 3.0], fov: 60, head: false },
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
const hang = (P, wrist) => {                          // rotate 180° about z (fingers point down), put the wrist at `wrist`
    const Q = new Float32Array(63);
    for (let j = 0; j < 21; j++) { Q[3 * j] = -(P[3 * j] - P[0]) + wrist[0]; Q[3 * j + 1] = -(P[3 * j + 1] - P[1]) + wrist[1]; Q[3 * j + 2] = P[3 * j + 2] - P[2] + wrist[2]; }
    return Q;
};

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
        const [d, left, right] = await Promise.all([fetch(url).then(r => r.json()), loadHandModel('left'), loadHandModel('right')]);
        this.poses = d.letters; this.meta = d.meta;
        const rigs = [new HandRig(left, this.mats.skin), new HandRig(right, this.mats.skin)];
        const rightType = rigs.find(r => r.handed > 0) || rigs[1], leftType = rigs.find(r => r !== rightType);
        this.rigs = [rightType, leftType];                       // slot 0 = right-type hand, slot 1 = left-type hand
        this.rigs.forEach(r => this.scene.add(r.root));
        this.body = new Body(this.scene, this.mats);
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

    /** camera presets: 'front' (whole signer) | 'hands' (close-up) | 'signer' (the signer's own point of view) */
    setView(name) {
        const v = VIEWS[name] || VIEWS.front; this.view = name;
        this.camera.position.set(...v.p); this.camera.fov = v.fov; this.camera.updateProjectionMatrix();
        if (this.controls) { this.controls.target.set(...v.t); this.controls.update(); } else this.camera.lookAt(...v.t);
        if (this.body) this.body.setHeadVisible(v.head);
    }

    /* --- poses (world coordinates; slot 0 = signer's right hand) --- */
    restPose() {
        const r = this.poses._rest.raw;                         // [left-type, right-type] relative to the wrist
        return [hang(flat(r[1]), HANG[0]), hang(flat(r[0]), HANG[1])];
    }
    info(letter) { return this.poses[letter]; }
    poseFor(letter) {
        const L = this.poses[letter]; if (!L) return this.restPose();
        const arr = this.useTouched ? L.touched : L.raw;
        if (L.hands === 2) return [shift(flat(arr[0]), SPACE_TWO), shift(flat(arr[1]), SPACE_TWO)];
        return [shift(flat(arr[0]), SPACE_ONE), this.restPose()[1]];       // one hand signs, the other hangs at the side
    }
    contactsFor(letter, pose) {
        const L = this.poses[letter]; if (!L || L.hands !== 2 || !this.useTouched) return [];
        return L.contacts.map(c => V(
            (pose[0][3 * c.a] + pose[1][3 * c.b]) / 2, (pose[0][3 * c.a + 1] + pose[1][3 * c.b + 1]) / 2,
            (pose[0][3 * c.a + 2] + pose[1][3 * c.b + 2]) / 2));
    }

    apply(pose, marks = []) {
        this.cur = pose;
        const wr = [], fw = [];
        pose.forEach((P, i) => {
            this.rigs[i].pose(P);
            wr.push(V(P[0], P[1], P[2]));
            fw.push(V(P[27] - P[0], P[28] - P[1], P[29] - P[2]).normalize());
        });
        this.body.update(wr, fw);
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
                const lift = [blendPose(target[0], rest[0], 0.28), blendPose(target[1], rest[1], 0.28)];
                items.push({ ch: null, pose: lift, trans: 0.28, hold: 0.0, idx: i, lift: true });
            }
            items.push({ ch, pose: target, trans: 0.55, hold: 0.85, idx: i });
            prev = ch;
        });
        items.push({ ch: null, pose: rest, trans: 0.8, hold: 0, idx: clean.length, end: true });
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
                const k = s.items.findIndex(it => el < it.end_t);
                const it = s.items[k], prevPose = k === 0 ? s.from : s.items[k - 1].pose;
                const lt = (el - it.start) * s.speed, a = Math.min(1, lt / it.trans);
                const pose = [blendPose(prevPose[0], it.pose[0], ease(a)), blendPose(prevPose[1], it.pose[1], ease(a))];
                const holding = a >= 1 && it.ch && it.ch !== ' ';
                this.apply(pose, holding ? this.contactsFor(it.ch, it.pose) : []);
                if (k !== s.fired && !it.lift) { s.fired = k; s.onItem(it, k); }
            }
        }
        if (this.body) this.body.head.rotation.z = Math.sin(now * 0.9) * 0.025;      // gentle idle
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
    av.renderer.setPixelRatio(1); av.renderer.setSize(320, 320, false);
    await av.load(); av.setSkin(skin); av.camera.aspect = 1; av.setView('thumb'); av.camera.aspect = 1; av.camera.updateProjectionMatrix();
    const out = {};
    for (const L of letters) { av.showLetter(L); av.renderer.render(av.scene, av.camera); out[L] = c.toDataURL('image/jpeg', 0.85); }
    av.renderer.dispose(); c.remove();
    return out;
}
