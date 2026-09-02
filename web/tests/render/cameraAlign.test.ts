/**
 * Iso rays (dirMatrix) must hit the same world points the box projects to.
 */
import * as THREE from "three";
import {
  ndcToDirMatrix,
  offsetDirMatrix,
  perspectiveDirScale,
} from "../../src/render/camera.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function applyProjectionOffset(camera: THREE.PerspectiveCamera, ox: number, oy: number) {
  camera.updateProjectionMatrix();
  const e = camera.projectionMatrix.elements;
  if (Math.abs(ox) > 1e-12) {
    for (let c = 0; c < 4; c++) e[c * 4 + 0] += ox * e[c * 4 + 3];
  }
  if (Math.abs(oy) > 1e-12) {
    for (let c = 0; c < 4; c++) e[c * 4 + 1] += oy * e[c * 4 + 3];
  }
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

function projectNdc(camera: THREE.Camera, p: THREE.Vector3): THREE.Vector3 {
  const v = p.clone().applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
  return v;
}

function rayDir(M: ArrayLike<number>, ndcX: number, ndcY: number): THREE.Vector3 {
  const xy1 = [ndcX, ndcY, 1];
  return new THREE.Vector3(
    M[0] * xy1[0] + M[1] * xy1[1] + M[2] * xy1[2],
    M[3] * xy1[0] + M[4] * xy1[1] + M[5] * xy1[2],
    M[6] * xy1[0] + M[7] * xy1[1] + M[8] * xy1[2],
  );
}

function rayMiss(ro: THREE.Vector3, rd: THREE.Vector3, target: THREE.Vector3): number {
  const to = target.clone().sub(ro);
  const t = to.dot(rd) / rd.lengthSq();
  const closest = ro.clone().add(rd.clone().multiplyScalar(t));
  return closest.distanceTo(target);
}

export async function run() {
  return runSuite("render / camera-align", [
    {
      name: "dirMatrix ray through projected NDC hits the world point (no offset)",
      fn: () => {
        const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 100);
        camera.up.set(0, 0, 1);
        camera.position.set(5.2, 6.8, 4.0);
        camera.lookAt(0, 0, 0);
        camera.updateMatrixWorld(true);
        camera.updateProjectionMatrix();

        const { sx, sy } = perspectiveDirScale(camera);
        const M = ndcToDirMatrix(camera, sx, sy);
        const ro = camera.position.clone();
        const target = new THREE.Vector3(1.2, -0.4, 0.8);
        const clip = projectNdc(camera, target);
        const miss = rayMiss(ro, rayDir(M, clip.x, clip.y), target);
        assert(miss < 1e-4, `ray missed projected point by ${miss}`);
      },
    },
    {
      name: "dirMatrix+offset matches projection offset for a world point",
      fn: () => {
        const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 100);
        camera.up.set(0, 0, 1);
        camera.position.set(5.2, 6.8, 4.0);
        camera.lookAt(0, 0, 0);
        camera.updateMatrixWorld(true);
        const ox = 0.28;
        const oy = -0.12;
        applyProjectionOffset(camera, ox, oy);

        const { sx, sy } = perspectiveDirScale(camera);
        const M = offsetDirMatrix(ndcToDirMatrix(camera, sx, sy), ox, oy);
        const ro = camera.position.clone();
        const target = new THREE.Vector3(1.2, -0.4, 0.8);
        const clip = projectNdc(camera, target);
        const miss = rayMiss(ro, rayDir(M, clip.x, clip.y), target);
        assert(
          miss < 1e-4,
          `offset ray missed projected point by ${miss} (ndc=${clip.x.toFixed(4)},${clip.y.toFixed(4)})`,
        );
      },
    },
    {
      name: "look-at origin projects to NDC offset after composition hack",
      fn: () => {
        const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.05, 100);
        camera.up.set(0, 0, 1);
        camera.position.set(5.2, 6.8, 4.0);
        camera.lookAt(0, 0, 0);
        camera.updateMatrixWorld(true);
        const ox = 0.28;
        applyProjectionOffset(camera, ox, 0);
        const clip = projectNdc(camera, new THREE.Vector3(0, 0, 0));
        assertNear(clip.x, ox, 1e-4, "origin ndc.x");
        assertNear(clip.y, 0, 1e-4, "origin ndc.y");
      },
    },
  ]);
}
