/**
 * Camera / NDC helpers for the volume march.
 */
import type { Camera, PerspectiveCamera } from "three";

type DirMatrix = Float64Array | Float32Array | number[];

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
 * Shift ray matrix so NDC x=0 aims through the composition center
 * (e.g. middle of the canvas region not covered by an overlay panel).
 */
export function offsetDirMatrix(M: DirMatrix, ndcOffsetX: number): DirMatrix {
  const o = Number(ndcOffsetX) || 0;
  if (!(Math.abs(o) > 1e-12)) return M;
  const out = M instanceof Float64Array ? new Float64Array(M) : new Float32Array(M);
  out[2] -= o * out[0];
  out[5] -= o * out[3];
  out[8] -= o * out[6];
  return out;
}
