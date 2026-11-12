/* HandRig — drives a skinned hand mesh (WebXR generic hand, MIT, 25-joint skeleton) from the 21
 * MediaPipe landmarks of one hand.
 *
 * Every bone is placed at its landmark (world position) and oriented so that its axis points at the
 * next landmark (shortest-arc from the orientation it would have under its parent's new pose); the
 * wrist takes its full orientation from the palm frame. Skinning then deforms the mesh, so the
 * fingers really touch where the landmarks say they touch.
 *
 * landmark index : bone
 *   0 wrist | 1..4 thumb CMC,MCP,IP,tip | 5..8 index MCP,PIP,DIP,tip | 9..12 middle | 13..16 ring | 17..20 pinky
 * WebXR also has a metacarpal bone per finger (no landmark): it is put on the wrist→MCP line.
 */
import * as THREE from 'three';
import { GLTFLoader } from '/static/vendor/loaders/GLTFLoader.js';

const FINGERS = [['index-finger', 5], ['middle-finger', 9], ['ring-finger', 13], ['pinky-finger', 17]];

/** bone chains, root -> tip. from/to are landmark indices; 'mc' = virtual metacarpal point. */
const CHAINS = (() => {
    const c = [[
        { name: 'thumb-metacarpal', from: 1, to: 2 }, { name: 'thumb-phalanx-proximal', from: 2, to: 3 },
        { name: 'thumb-phalanx-distal', from: 3, to: 4 }, { name: 'thumb-tip', from: 4, to: null }]];
    for (const [f, m] of FINGERS) c.push([
        { name: `${f}-metacarpal`, from: 'mc', to: m, mcOf: m }, { name: `${f}-phalanx-proximal`, from: m, to: m + 1 },
        { name: `${f}-phalanx-intermediate`, from: m + 1, to: m + 2 }, { name: `${f}-phalanx-distal`, from: m + 2, to: m + 3 },
        { name: `${f}-tip`, from: m + 3, to: null }]);
    return c;
})();

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const _loader = new GLTFLoader();
export function loadHandModel(side) {      // fresh copy per call (each avatar owns its skeleton); the browser caches the file
    return new Promise((res, rej) => _loader.load(`/static/models/hand-${side}.glb`, g => res(g.scene), undefined, rej));
}

function basis(f, n) {                                        // orthonormal frame from a forward and a normal hint
    const e1 = f.clone().normalize();
    const e2 = n.clone().sub(e1.clone().multiplyScalar(n.dot(e1))).normalize();
    return new THREE.Matrix4().makeBasis(e1, e2, e1.clone().cross(e2));
}

export class HandRig {
    /** scene: a fresh copy of a hand GLB scene. NOTE: the skeleton in this model is FLAT (every bone is a
     *  direct child of one Armature node), so bind data is taken in WORLD space and the finger chains
     *  below carry each bone's rotation to the next one. */
    constructor(scene, material) {
        this.root = scene;
        this.bones = {};
        scene.traverse(o => {
            if (o.isBone) this.bones[o.name] = o;
            if (o.isSkinnedMesh) { this.mesh = o; o.frustumCulled = false; if (material) o.material = material; o.castShadow = o.receiveShadow = true; }
        });
        scene.updateMatrixWorld(true);
        this.bindW = {};
        for (const n in this.bones) this.bindW[n] = { p: this.bones[n].getWorldPosition(V()), q: this.bones[n].getWorldQuaternion(new THREE.Quaternion()) };
        const bp = n => this.bindW[n].p;
        const w = bp('wrist');
        this.k = 1 / bp('middle-finger-phalanx-proximal').distanceTo(w);          // metres -> hand-size units
        this.dirBind = {};                                                       // bind world direction bone -> next bone in its chain
        for (const ch of CHAINS) for (let i = 0; i < ch.length - 1; i++) this.dirBind[ch[i].name] = bp(ch[i + 1].name).clone().sub(bp(ch[i].name)).normalize();
        // bind palm frame and handedness signature (same formula as the landmark side)
        const pI = bp('index-finger-phalanx-proximal').clone().sub(w), pP = bp('pinky-finger-phalanx-proximal').clone().sub(w), pM = bp('middle-finger-phalanx-proximal').clone().sub(w);
        this.bindWristM = basis(pM, pI.clone().cross(pP));
        this.handed = Math.sign(pI.clone().cross(pP).dot(bp('thumb-phalanx-proximal').clone().sub(w))) || 1;   // +1 right-type, -1 left-type
        this._s = V(this.k, this.k, this.k);
    }

    _place(bone, pos, q) {
        const desired = new THREE.Matrix4().compose(pos, q, this._s);
        const local = bone.parent.matrixWorld.clone().invert().multiply(desired);
        local.decompose(bone.position, bone.quaternion, bone.scale);
        bone.updateMatrixWorld(true);
    }

    /** L: Float32Array(63) landmark positions in world units (y up) */
    pose(L) {
        const P = i => V(L[3 * i], L[3 * i + 1], L[3 * i + 2]);
        const w = P(0);
        // wrist: full orientation from the palm frame; dW is the rotation applied to the whole hand
        const f = P(9).sub(w), n = P(5).sub(w).cross(P(17).sub(w));
        const dW = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().multiplyMatrices(basis(f, n), this.bindWristM.clone().transpose()));
        this._place(this.bones['wrist'], w, dW.clone().multiply(this.bindW['wrist'].q));
        for (const ch of CHAINS) {
            let dPar = dW;                                           // rotation carried down the chain
            for (let i = 0; i < ch.length; i++) {
                const e = ch[i], bone = this.bones[e.name], bw = this.bindW[e.name];
                let pos, dir = null;
                if (e.from === 'mc') {                               // metacarpal joint rides on the wrist
                    pos = w.clone().add(bw.p.clone().sub(this.bindW['wrist'].p).multiplyScalar(this.k).applyQuaternion(dW));
                    dir = P(e.mcOf).sub(pos);
                } else { pos = P(e.from); if (e.to != null) dir = P(e.to).sub(pos); }
                let d = dPar;
                if (dir && dir.lengthSq() > 1e-10) {
                    const cur = this.dirBind[e.name].clone().applyQuaternion(dPar);
                    d = new THREE.Quaternion().setFromUnitVectors(cur, dir.normalize()).multiply(dPar);
                }
                this._place(bone, pos, d.clone().multiply(bw.q));
                dPar = d;
            }
        }
    }
}
