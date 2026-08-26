/**
 * Camera / NDC helpers for the volume march.
 * Soft degree cap for fit UI; volume grid is M = deg + 1.
 */

/** Soft cap for fit / UI. Volume grid is M = deg + 1. */
export const MAX_DEG = 128;

/**
 * Map NDC (x, y, 1) → world dir_raw = R · (sx x, sy y, −1).
 * Returns 3×3 row-major: d = M · (x, y, 1).
 */
export function ndcToDirMatrix(camera, sx, sy) {
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

export function perspectiveDirScale(camera) {
  const tan = Math.tan((camera.fov * Math.PI) / 180 / 2);
  return { sx: tan * camera.aspect, sy: tan };
}
