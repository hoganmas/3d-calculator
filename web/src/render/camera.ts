/**
 * Camera / NDC helpers for the volume march.
 */
import { Vector4, type Camera, type PerspectiveCamera } from "three";

export type DirMatrix = Float64Array | Float32Array | number[];

const _clip = new Vector4();

/**
 * Map NDC (x, y, 1) → world dir_raw = R · (sx x, sy y, −1).
 * Returns 3×3 row-major: d = M · (x, y, 1).
 */
export function ndcToDirMatrix(camera: Camera, sx: number, sy: number): Float64Array {
  const e = camera.matrixWorld.elements;
  const M = new Float64Array(9);
  M[0] = e[0] * sx;
  M[1] = e[4] * sy;
  M[2] = -e[8];
  M[3] = e[1] * sx;
  M[4] = e[5] * sy;
  M[5] = -e[9];
  M[6] = e[2] * sx;
  M[7] = e[6] * sy;
  M[8] = -e[10];
  return M;
}

export function perspectiveDirScale(camera: PerspectiveCamera): { sx: number; sy: number } {
  const tan = Math.tan((camera.fov * Math.PI) / 180 / 2);
  return { sx: tan * camera.aspect, sy: tan };
}

/**
 * Build the same M · (ndcX, ndcY, 1) → world dir_raw map from an arbitrary
 * projection (including asymmetric XR eye frusta). Uses the near-plane
 * unprojection so stereo views stay correct under Three.js WebXR.
 */
export function dirMatrixFromProjection(camera: Camera): Float64Array {
  camera.updateMatrixWorld(true);
  const invP = camera.projectionMatrixInverse;
  const mw = camera.matrixWorld.elements;

  function worldDir(ndcX: number, ndcY: number): [number, number, number] {
    _clip.set(ndcX, ndcY, -1, 1).applyMatrix4(invP);
    const iw = 1 / (_clip.w || 1e-8);
    const x = _clip.x * iw;
    const y = _clip.y * iw;
    const z = _clip.z * iw;
    return [
      mw[0] * x + mw[4] * y + mw[8] * z,
      mw[1] * x + mw[5] * y + mw[9] * z,
      mw[2] * x + mw[6] * y + mw[10] * z,
    ];
  }

  const c = worldDir(0, 0);
  const r = worldDir(1, 0);
  const u = worldDir(0, 1);
  const M = new Float64Array(9);
  M[0] = r[0] - c[0];
  M[1] = u[0] - c[0];
  M[2] = c[0];
  M[3] = r[1] - c[1];
  M[4] = u[1] - c[1];
  M[5] = c[1];
  M[6] = r[2] - c[2];
  M[7] = u[2] - c[2];
  M[8] = c[2];
  return M;
}

/**
 * Shift ray matrix so NDC (0,0) aims through the composition center
 * (region not covered by an overlay panel).
 */
export function offsetDirMatrix(
  M: DirMatrix,
  ndcOffsetX = 0,
  ndcOffsetY = 0,
): DirMatrix {
  const ox = Number(ndcOffsetX) || 0;
  const oy = Number(ndcOffsetY) || 0;
  if (!(Math.abs(ox) > 1e-12) && !(Math.abs(oy) > 1e-12)) return M;
  const out = M instanceof Float64Array ? new Float64Array(M) : new Float32Array(M);
  if (Math.abs(ox) > 1e-12) {
    out[2] -= ox * out[0];
    out[5] -= ox * out[3];
    out[8] -= ox * out[6];
  }
  if (Math.abs(oy) > 1e-12) {
    out[2] -= oy * out[1];
    out[5] -= oy * out[4];
    out[8] -= oy * out[7];
  }
  return out;
}
