/* Sleeves — a cloth forearm behind each hand (no body, no head).
 * Each sleeve follows its hand's own axis (from the knuckles back to the wrist), is blended a little
 * toward "down" so it never points at the camera, and runs out of the frame. A short skin wrist and
 * a trim ring join it to the hand. Units: 1 = one hand size (≈ 9.5 cm).
 */
import * as THREE from 'three';

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const Y = V(0, 1, 0);
const LEN = 16;

function limb(mesh, a, b) {                                    // cylinder (height 1 along +y) between two points
    const d = b.clone().sub(a), len = d.length() || 1e-4;
    mesh.position.copy(a).addScaledVector(d, 0.5);
    mesh.quaternion.setFromUnitVectors(Y, d.divideScalar(len));
    mesh.scale.set(1, len, 1);
}

export class Sleeves {
    constructor(scene, mats) {
        this.group = new THREE.Group(); scene.add(this.group);
        const add = (geo, mat) => { const m = new THREE.Mesh(geo, mat); m.castShadow = m.receiveShadow = true; this.group.add(m); return m; };
        this.arms = [0, 1].map(() => ({
            sleeve: add(new THREE.CylinderGeometry(0.95, 0.46, 1, 40, 1, true), mats.cloth),     // top = far end (wider)
            cap: add(new THREE.RingGeometry(0.3, 0.46, 40), mats.cloth),
            cuff: add(new THREE.TorusGeometry(0.43, 0.07, 14, 40), mats.trim),
            wrist: add(new THREE.CylinderGeometry(0.27, 0.31, 1, 24), mats.skin),
        }));
        this.arms.forEach(a => { a.sleeve.material = mats.cloth.clone(); a.sleeve.material.side = THREE.DoubleSide; a.cap.visible = false; });
    }

    /** pose: [Float32Array(63), Float32Array(63)] world landmark positions (slot 0, slot 1) */
    update(pose) {
        this.arms.forEach((a, i) => {
            const P = pose[i];
            const W = V(P[0], P[1], P[2]);
            const pc = V(0, 0, 0);
            [5, 9, 13, 17].forEach(j => pc.add(V(P[3 * j], P[3 * j + 1], P[3 * j + 2])));
            pc.divideScalar(4);
            const back = W.clone().sub(pc).normalize().multiplyScalar(0.72).add(V(0, -0.5, -0.1)).normalize();   // away from the fingers
            const cuffAt = W.clone().addScaledVector(back, 0.42);
            limb(a.wrist, W, cuffAt);
            a.cuff.position.copy(cuffAt); a.cuff.quaternion.setFromUnitVectors(V(0, 0, 1), back);
            limb(a.sleeve, cuffAt, cuffAt.clone().addScaledVector(back, LEN));
        });
    }
}
