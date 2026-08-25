import { evalMonomial3D } from "./fit.js";

/**
 * Clip/NDC-grid fiber bake — golden path dens atlas for poly-cloud.
 * See research/poly/notes/clip-space-babbage.md.
 *
 * Atlas stores dens at fixed ray-t Chebyshev-root nodes t_j (view-fixed window).
 * Each dens_j(x_ndc,y_ndc)=f(ro+t_j·rd) is filled with coarse-step middle-out
 * Babbage + Newton (f64 on CPU; GPU port in clipBakeGpu.js). Outside-box samples
 * are zeroed so f32/bilinear aren't poisoned by exterior Runge spikes.
 * March is Beer–Lambert raymarch (not Path C); see research/poly/notes/path-c.md.
 */

export const MAX_DEG = 16;
/** Default tile width for middle-out coarse lattice (pixels). */
export const BABBAGE_TILE = 256;

function clear(arr) {
  arr.fill(0);
}

export function horner1d(a, deg, x) {
  let s = 0;
  for (let i = deg; i >= 0; i--) s = s * x + (a[i] || 0);
  return s;
}

/** Chebyshev roots (1st kind) on (-1,1) — avoids Lobatto endpoints where the
 *  world poly explodes outside the fit box and wrecks f32 / barycentric. */
export function chebRootNodes(nAlpha) {
  const M = Math.max(1, nAlpha | 0);
  const u = new Float64Array(M);
  for (let j = 0; j < M; j++) {
    u[j] = Math.cos((Math.PI * (j + 0.5)) / M);
  }
  return u;
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

/**
 * View-fixed fiber window. Prefer the *center-ray* box slab (padded): a sphere
 * window evaluates the world poly far outside the fit domain → O(1e15) dens
 * samples that destroy f32 and high-order Δ.
 */
export function viewFiberWindow(ro, half, M) {
  const h = half;
  if (M) {
    const rd = [M[2], M[5], M[8]]; // NDC (0,0) → dir
    const hit = intersectRayBox(ro, rd, h);
    if (hit) {
      let [t0, t1] = hit;
      const pad = Math.max(0.05 * (t1 - t0), 1e-4);
      t0 = Math.max(0, t0 - pad);
      t1 = t1 + pad;
      // Slight shrink so u∈(-1,1) Chebyshev roots stay near the box.
      const mid = 0.5 * (t0 + t1);
      const hw = 0.5 * (t1 - t0) * 0.98;
      return { tMid: mid, tHw: Math.max(hw, 1e-3) };
    }
  }
  const r = h * Math.sqrt(3);
  const oc = Math.hypot(ro[0], ro[1], ro[2]);
  const t0 = Math.max(0, oc - r);
  const t1 = oc + r + 1e-3;
  return { tMid: 0.5 * (t0 + t1), tHw: Math.max(0.5 * (t1 - t0), 1e-3) };
}

/** Build forward-difference table: d[k] = Δ^k f[0] from samples f[0..D] (step h). */
function buildForwardDiffs(samples, D, dOut) {
  for (let i = 0; i <= D; i++) dOut[i] = samples[i];
  for (let k = 1; k <= D; k++) {
    for (let i = D; i >= k; i--) dOut[i] -= dOut[i - 1];
  }
}

/**
 * Newton forward series: f(x0 + t·h) = Σ binom(t,k) Δ^k.
 * t is in units of the coarse step h (t=0 at first seed). Negative t is valid
 * (middle-out: stencil centered in the tile, evaluate both sides).
 */
function evalNewtonForward(d, D, t) {
  let acc = d[0];
  let binom = 1;
  for (let k = 1; k <= D; k++) {
    binom *= (t - (k - 1)) / k;
    acc += binom * d[k];
  }
  return acc;
}

/**
 * Uniform step h with D·h ≥ span. Prefer power-of-two ≥ ceil(span/D)
 * (middle-out / dyadic friendly). Never default to 1px when coarser fits.
 */
function coarseStep(span, D) {
  if (span <= D) return 1;
  const hNeed = Math.max(1, Math.ceil(span / D));
  let h = 1;
  while (h < hNeed) h *= 2;
  return h;
}

/**
 * Bake dens(t_j) sample planes via f64 coarse-step Babbage + Newton subdivision.
 */
export function fillGammaGrid(worldMono, deg, width, height, x0, dx, y0, dy, ro, half, M, tilePx = BABBAGE_TILE) {
  const max1d = 3 * deg;
  const nAlpha = max1d + 1;
  const D = max1d;
  const out = new Float32Array(width * height * nAlpha);
  const { tMid, tHw } = viewFiberWindow(ro, half, M);
  const tile = Math.max(D + 1, tilePx | 0);
  const uNodes = chebRootNodes(nAlpha);
  const tNodes = new Float64Array(nAlpha);
  for (let j = 0; j < nAlpha; j++) tNodes[j] = tMid + tHw * uNodes[j];

  const dens = new Float64Array(nAlpha);
  const rd = new Float64Array(3);
  const p = new Float64Array(3);

  const diff = new Array(nAlpha);
  for (let k = 0; k < nAlpha; k++) diff[k] = new Float64Array(D + 1);
  const seedSamples = new Array(nAlpha);
  for (let k = 0; k < nAlpha; k++) seedSamples[k] = new Float64Array(D + 1);

  function exactDensAt(px, py) {
    const x = x0 + dx * (px + 0.5);
    const y = y0 + dy * (py + 0.5);
    rd[0] = M[0] * x + M[1] * y + M[2];
    rd[1] = M[3] * x + M[4] * y + M[5];
    rd[2] = M[6] * x + M[7] * y + M[8];
    // Direct world eval (stable); avoid Horner on raw α(t) with |t|≳1, deg∼3N.
    for (let j = 0; j < nAlpha; j++) {
      const t = tNodes[j];
      p[0] = ro[0] + t * rd[0];
      p[1] = ro[1] + t * rd[1];
      p[2] = ro[2] + t * rd[2];
      dens[j] = evalMonomial3D(worldMono, deg, p[0], p[1], p[2]);
    }
  }

  function writePixel(px, py, samples) {
    // Recompute rd for outside-box masking at this pixel.
    const x = x0 + dx * (px + 0.5);
    const y = y0 + dy * (py + 0.5);
    rd[0] = M[0] * x + M[1] * y + M[2];
    rd[1] = M[3] * x + M[4] * y + M[5];
    rd[2] = M[6] * x + M[7] * y + M[8];
    // Soft cap: exterior Runge spikes and rare Newton glitches must not enter f32
    // (they turn into white fireflies under any high-order u-interp).
    const CAP = 8;
    for (let j = 0; j < nAlpha; j++) {
      const t = tNodes[j];
      p[0] = ro[0] + t * rd[0];
      p[1] = ro[1] + t * rd[1];
      p[2] = ro[2] + t * rd[2];
      const inside =
        Math.abs(p[0]) <= half && Math.abs(p[1]) <= half && Math.abs(p[2]) <= half;
      let v = inside ? samples[j] : 0;
      if (!Number.isFinite(v)) v = 0;
      else if (v > CAP) v = CAP;
      else if (v < -CAP) v = -CAP;
      out[(j * height + py) * width + px] = v;
    }
  }

  function fillTileExact(xBase, xEnd, py) {
    for (let px = xBase; px < xEnd; px++) {
      exactDensAt(px, py);
      writePixel(px, py, dens);
    }
  }

  function fillTileBabbage(xBase, xEnd, py) {
    const span = xEnd - xBase - 1;
    if (span <= 0) {
      exactDensAt(xBase, py);
      writePixel(xBase, py, dens);
      return;
    }
    if (span < D) {
      fillTileExact(xBase, xEnd, py);
      return;
    }

    const h = coarseStep(span, D);
    // Middle-out: center the D·h coarse stencil in the tile so Newton |t|
    // (and fp error) is symmetric about the tile mid, not piled on the right.
    const cover = D * h;
    const xOrigin = xBase + Math.floor((span - cover) / 2);
    for (let i = 0; i <= D; i++) {
      exactDensAt(xOrigin + i * h, py);
      for (let j = 0; j < nAlpha; j++) seedSamples[j][i] = dens[j];
    }
    for (let j = 0; j < nAlpha; j++) {
      buildForwardDiffs(seedSamples[j], D, diff[j]);
    }

    for (let px = xBase; px < xEnd; px++) {
      const t = (px - xOrigin) / h; // may be < 0 on the left half
      for (let j = 0; j < nAlpha; j++) {
        dens[j] = evalNewtonForward(diff[j], D, t);
      }
      writePixel(px, py, dens);
    }
  }

  // Center the tile partition on the atlas so leftover strips split L/R (and T/B).
  const nTilesX = Math.max(1, Math.ceil(width / tile));
  const nTilesY = Math.max(1, Math.ceil(height / tile));
  const xGrid0 = -Math.floor((nTilesX * tile - width) / 2);
  const yGrid0 = -Math.floor((nTilesY * tile - height) / 2);

  for (let ty = 0; ty < nTilesY; ty++) {
    const yBase = Math.max(0, yGrid0 + ty * tile);
    const yEnd = Math.min(height, yGrid0 + (ty + 1) * tile);
    if (yBase >= yEnd) continue;
    for (let py = yBase; py < yEnd; py++) {
      for (let tx = 0; tx < nTilesX; tx++) {
        const xBase = Math.max(0, xGrid0 + tx * tile);
        const xEnd = Math.min(width, xGrid0 + (tx + 1) * tile);
        if (xBase >= xEnd) continue;
        fillTileBabbage(xBase, xEnd, py);
      }
    }
  }

  return {
    data: out,
    width,
    height,
    nAlpha,
    max1d,
    tMid,
    tHw,
    tile,
    uNodes: new Float32Array(uNodes),
    method: "babbage-newton-t-mid",
  };
}

export function bakeClipGridFibers(worldMono, deg, camera, width, height, half) {
  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const M = ndcToDirMatrix(camera, sx, sy);
  const dx = 2 / width;
  const dy = 2 / height;
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;
  const grid = fillGammaGrid(worldMono, deg, width, height, -1, dx, -1, dy, ro, h, M);
  return { ...grid, sx, sy, deg, M, ro, half: h };
}
