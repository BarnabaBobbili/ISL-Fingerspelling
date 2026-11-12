/* Body — a stylised upper-body signer: head, neck, torso in a kurta, shoulders, sleeved arms.
 * The arms are solved with two-bone inverse kinematics from each shoulder to the pose's wrist.
 * Units: 1 = one hand size (wrist → middle-finger knuckle, ≈ 9.5 cm). y = 0 is the shoulder line,
 * the signer faces +z (towards the viewer), slot 0 = the signer's right arm = the viewer's left.
 */
import * as THREE from 'three';

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
export const SHOULDER = 2.05;          // half shoulder width
export const UPPER = 3.15, FORE = 2.75; // upper arm / forearm lengths

export function solveArm(S, T, a, b, pole) {
    const d = T.clone().sub(S);
    let L = d.length(); const u = d.clone().divideScalar(L || 1);
    L = Math.min(Math.max(L, Math.abs(a - b) + 1e-3), a + b - 1e-3);
    const cosA = (a * a + L * L - b * b) / (2 * a * L), sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
    let p = pole.clone().sub(u.clone().multiplyScalar(pole.dot(u)));
    if (p.lengthSq() < 1e-6) p = V(0, -1, 0).sub(u.clone().multiplyScalar(-u.y));
    p.normalize();
    const E = S.clone().add(u.clone().multiplyScalar(a * cosA)).add(p.multiplyScalar(a * sinA));
    // the wrist can only reach distance L: if the target is farther, the wrist stops short of it
    return { E, W: S.clone().add(u.clone().multiplyScalar(L)) };
}

const Y = V(0, 1, 0);
function limb(mesh, a, b) {                                    // cylinder (height 1 along +y) between two points
    const d = b.clone().sub(a), len = d.length() || 1e-4;
    mesh.position.copy(a).addScaledVector(d, 0.5);
    mesh.quaternion.setFromUnitVectors(Y, d.divideScalar(len));
    mesh.scale.set(1, len, 1);
}

export class Body {
    constructor(scene, mats) {
        this.mats = mats;
        this.group = new THREE.Group(); scene.add(this.group);
        const { skin, cloth, trim, hair } = mats;
        const add = (geo, mat, x = 0, y = 0, z = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = m.receiveShadow = true; this.group.add(m); return m; };

        // torso: lathe profile (radius by height), squashed front-to-back; a kurta
        const prof = [[0.55, 0.45], [0.95, 0.15], [1.9, -0.12], [2.25, -0.55], [2.2, -1.6], [2.05, -3.0], [1.9, -4.6], [1.95, -6.4], [2.0, -8.5]];
        const lathe = new THREE.LatheGeometry(prof.map(([r, y]) => V(r, y)), 56);
        this.torso = add(lathe, cloth); this.torso.scale.z = 0.62;
        this.torso.position.z = -0.05;
        // collar + placket in the accent colour
        const collar = add(new THREE.TorusGeometry(0.62, 0.09, 16, 48), trim, 0, 0.42, 0.05); collar.rotation.x = Math.PI / 2; collar.scale.y = 0.9;
        const placket = add(new THREE.BoxGeometry(0.16, 3.0, 0.05), trim, 0, -1.4, 1.43); placket.rotation.x = -0.02;
        // neck, head, hair
        add(new THREE.CylinderGeometry(0.46, 0.52, 1.25, 28), skin, 0, 0.85, 0);
        this.head = new THREE.Group(); this.group.add(this.head); this.head.position.set(0, 2.55, 0.05);
        const hd = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), skin); hd.scale.set(0.84, 1.18, 0.96); hd.castShadow = true; this.head.add(hd);
        const nose = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), skin); nose.position.set(0, -0.28, 0.93); nose.scale.set(1, 1.1, 1.2); this.head.add(nose);
        const hr = new THREE.Mesh(new THREE.SphereGeometry(1.02, 40, 24, 0, Math.PI * 2, 0, Math.PI * 0.56), hair);
        hr.scale.set(0.9, 1.22, 1.06); hr.position.set(0, 0.04, -0.1); hr.rotation.x = 0.16; hr.castShadow = true; this.head.add(hr);
        for (const sx of [-1, 1]) { const ear = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 12), skin); ear.scale.set(0.55, 1, 0.8); ear.position.set(sx * 0.82, -0.05, 0.0); ear.castShadow = true; this.head.add(ear); }
        this.headParts = [this.head];
        // shoulders and arms
        this.arms = [];
        for (const side of [-1, 1]) {                           // side -1 = signer's right = viewer's left (slot 0)
            const sh = add(new THREE.SphereGeometry(0.6, 28, 20), cloth, side * SHOULDER, -0.2, 0);
            const up = add(new THREE.CylinderGeometry(0.42, 0.5, 1, 28), cloth);
            const elb = add(new THREE.SphereGeometry(0.44, 24, 16), cloth);
            const fo = add(new THREE.CylinderGeometry(0.29, 0.42, 1, 28), cloth);
            const cuff = add(new THREE.TorusGeometry(0.3, 0.065, 12, 32), trim);
            const wr = add(new THREE.CylinderGeometry(0.25, 0.29, 1, 20), skin);
            const wj = add(new THREE.SphereGeometry(0.31, 20, 14), skin);
            this.arms.push({ side, shoulder: V(side * SHOULDER, -0.2, 0), sh, up, elb, fo, cuff, wr, wj });
        }
    }

    setSkin(mat) { this.mats.skin = mat; }

    /** place both arms. wrists: [Vector3, Vector3] world wrist targets; fwd: [Vector3,...] hand directions (wrist → fingers) */
    update(wrists, fwd) {
        this.arms.forEach((a, i) => {
            const S = a.shoulder, T = wrists[i];
            const outward = V(a.side, -1, 0.15).normalize();
            const ideal = T.clone().addScaledVector(fwd[i], -FORE * 0.9).sub(S).normalize();   // elbow behind the hand's axis
            const pole = ideal.multiplyScalar(0.55).add(outward.multiplyScalar(0.45));
            const { E, W } = solveArm(S, T, UPPER, FORE, pole);
            limb(a.up, S, E);
            a.elb.position.copy(E);
            const dir = W.clone().sub(E).normalize();
            const sleeveEnd = W.clone().addScaledVector(dir, -0.28);
            limb(a.fo, E, sleeveEnd);
            a.cuff.position.copy(sleeveEnd); a.cuff.quaternion.setFromUnitVectors(V(0, 0, 1), dir);
            limb(a.wr, sleeveEnd, W.clone().addScaledVector(dir, 0.1));
            a.wj.position.copy(W);
        });
    }

    setHeadVisible(v) { this.head.visible = v; }
}
