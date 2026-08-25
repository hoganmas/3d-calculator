/**
 * Clip/NDC-grid fiber pipeline (see research/poly/notes/clip-space-babbage.md).
 *
 * Per atlas pixel: intersect the primary ray with the bbox, then build γ(u) for
 * f(P0 + u·Du) on u∈[-1,1] by the same nested Horner composition as the LOS
 * shader (world monomials). Do NOT form α(s) and binomial-reexpand — that basis
 * is catastrophically unstable for deg ≳ 5 and paints corner “blobs”.
 */

export const MAX_DEG = 8;

function clear(arr) {
  arr.fill(0);
}

export function horner1d(a, deg, x) {
  let s = 0;
  for (let i = deg; i >= 0; i--) s = s * x + (a[i] || 0);
  return s;
}

/**
 * Map NDC (x,y,1) → world dir_raw = R · (sx x, sy y, −1).
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

function intersectRayBox(ro, rd, h) {
  let t0 = -1e30;
  let t1 = 1e30;
  for (let a = 0; a < 3; a++) {
    const d = rd[a];
    const inv = Math.abs(d) < 1e-15 ? 1e15 : 1 / d;
    const tA = (-h - ro[a]) * inv;
    const tB = (h - ro[a]) * inv;
    const lo = Math.min(tA, tB);
    const hi = Math.max(tA, tB);
    if (lo > t0) t0 = lo;
    if (hi < t1) t1 = hi;
  }
  if (t0 < 0) t0 = 0;
  return t1 > t0 ? [t0, t1] : null;
}

function mulLinear(poly, a0, a1, max1d, out) {
  clear(out);
  for (let t = 0; t <= max1d; t++) {
    const v = poly[t];
    if (v === 0) continue;
    out[t] += v * a0;
    if (t + 1 <= max1d) out[t + 1] += v * a1;
  }
}

/**
 * Exact LOS-style γ(u) = f(P0 + u·Du) via nested Horner (float64).
 * Matches web/poly-cloud/src/shaders.js composition.
 */
export function composeGammaNested(worldMono, deg, P0, Du, gamma) {
  const n = deg + 1;
  const max1d = 3 * deg;
  const nAlpha = max1d + 1;
  clear(gamma);

  const zPow = new Array(n);
  for (let k = 0; k < n; k++) zPow[k] = new Float64Array(nAlpha);
  const pk = new Float64Array(nAlpha);
  const tmp = new Float64Array(nAlpha);
  pk[0] = 1;
  for (let k = 0; k <= deg; k++) {
    zPow[k].set(pk);
    if (k === deg) break;
    mulLinear(pk, P0[2], Du[2], max1d, tmp);
    pk.set(tmp);
  }

  const si = new Float64Array(nAlpha);
  const row = new Float64Array(nAlpha);
  for (let i = deg; i >= 0; i--) {
    clear(si);
    for (let j = deg; j >= 0; j--) {
      clear(row);
      for (let k = 0; k <= deg; k++) {
        const c = worldMono[i + j * n + k * n * n];
        if (Math.abs(c) < 1e-20) continue;
        const zp = zPow[k];
        for (let m = 0; m < nAlpha; m++) row[m] += c * zp[m];
      }
      if (j < deg) {
        mulLinear(si, P0[1], Du[1], max1d, tmp);
        si.set(tmp);
      }
      for (let m = 0; m < nAlpha; m++) si[m] += row[m];
    }
    if (i < deg) {
      mulLinear(gamma, P0[0], Du[0], max1d, tmp);
      gamma.set(tmp);
    }
    for (let m = 0; m < nAlpha; m++) gamma[m] += si[m];
  }
  return max1d;
}

/**
 * Pack γ_k(u) into atlas: width×(height*nAlpha), block k at rows [k*H,(k+1)*H).
 */
export function fillGammaGrid(worldMono, deg, width, height, x0, dx, y0, dy, ro, half, M) {
  const max1d = 3 * deg;
  const nAlpha = max1d + 1;
  const out = new Float32Array(width * height * nAlpha);
  const gamma = new Float64Array(nAlpha);
  const rd = new Float64Array(3);
  const P0 = new Float64Array(3);
  const Du = new Float64Array(3);

  for (let py = 0; py < height; py++) {
    const y = y0 + dy * (py + 0.5);
    for (let px = 0; px < width; px++) {
      const x = x0 + dx * (px + 0.5);
      rd[0] = M[0] * x + M[1] * y + M[2];
      rd[1] = M[3] * x + M[4] * y + M[5];
      rd[2] = M[6] * x + M[7] * y + M[8];
      const hit = intersectRayBox(ro, rd, half);
      if (!hit) continue;
      const [tEnter, tExit] = hit;
      const tMid = 0.5 * (tEnter + tExit);
      const tHw = 0.5 * (tExit - tEnter);
      if (!(tHw > 1e-10)) continue;

      P0[0] = ro[0] + tMid * rd[0];
      P0[1] = ro[1] + tMid * rd[1];
      P0[2] = ro[2] + tMid * rd[2];
      Du[0] = tHw * rd[0];
      Du[1] = tHw * rd[1];
      Du[2] = tHw * rd[2];

      composeGammaNested(worldMono, deg, P0, Du, gamma);
      for (let k = 0; k <= max1d; k++) {
        out[(k * height + py) * width + px] = gamma[k];
      }
    }
  }
  return { data: out, width, height, nAlpha, max1d };
}

export function bakeClipGridFibers(worldMono, deg, camera, width, height, half) {
  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const M = ndcToDirMatrix(camera, sx, sy);
  const dx = 2 / width;
  const dy = 2 / height;
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;
  // World monomials + nested compose (no camera-frame α / reexpand).
  const grid = fillGammaGrid(worldMono, deg, width, height, -1, dx, -1, dy, ro, h, M);
  return { ...grid, sx, sy, deg, M };
}
