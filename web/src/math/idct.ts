/**
 * Separable Chebyshev IDCT-III: coeffs c_ijk → dens on Chebyshev-root grid.
 * Convention matches fit.js (eval f = Σ c_ijk T_i T_j T_k; analysis used α in DCT).
 * See research/poly/notes/cheb-idct-volume.md.
 */

import type {
  Idct3DResult,
  IdctCurl3DResult,
  IdctGrad3DResult,
} from "../types/models.js";
import type { ChebAxis } from "./calcOps.js";

/**
 * cos(i·(m+½)·π/M) basis for the length-n → length-M Chebyshev IDCT-III,
 * packed [m*n+i]. Every expression sharing a (n, M) pair (typically all
 * layers at the current UI degree) reuses the same basis instead of each
 * paying its own O(n·M) transcendental cost.
 */
const chebBasisCache = new Map<number, Float64Array>();

function getChebBasis(n: number, M: number): Float64Array {
  const key = n * 100000 + M;
  const cached = chebBasisCache.get(key);
  if (cached) return cached;
  const basis = new Float64Array(M * n);
  const invM = Math.PI / M;
  for (let m = 0; m < M; m++) {
    const phase = (m + 0.5) * invM;
    const base = m * n;
    for (let i = 0; i < n; i++) basis[base + i] = Math.cos(i * phase);
  }
  chebBasisCache.set(key, basis);
  return basis;
}

/** Univariate IDCT at M Chebyshev roots: v_m = Σ_{i=0}^{n-1} c_i T_i(ξ_m). */
function idctCheb1D(coeff: ArrayLike<number>, M: number): Float64Array {
  const n = coeff.length;
  const basis = getChebBasis(n, M);
  const out = new Float64Array(M);
  for (let m = 0; m < M; m++) {
    const base = m * n;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const c = coeff[i];
      if (c !== 0) s += c * basis[base + i];
    }
    out[m] = s;
  }
  return out;
}

/** Same as idctCheb1D but writes straight into dst (stride dstStride, from dstBase) — avoids a per-row allocation when a 3D pass runs this over many rows. */
function applyChebBasisRow(
  basis: Float64Array,
  n: number,
  M: number,
  coeff: ArrayLike<number>,
  dst: Float64Array | Float32Array,
  dstBase: number,
  dstStride: number,
): void {
  for (let m = 0; m < M; m++) {
    const base = m * n;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const c = coeff[i];
      if (c !== 0) s += c * basis[base + i];
    }
    dst[dstBase + m * dstStride] = s;
  }
}

/**
 * Chebyshev → derivative Chebyshev (same length, last mode 0).
 * If s = Σ_{k=0}^{n-1} c_k T_k, returns d with s' = Σ d_k T_k.
 */
function chebDiff1D(c: ArrayLike<number>): Float64Array {
  const n = c.length;
  const d = new Float64Array(n);
  if (n < 2) return d;
  d[n - 1] = 0;
  d[n - 2] = 2 * (n - 1) * (c[n - 1] || 0);
  for (let k = n - 3; k >= 0; k--) {
    d[k] = d[k + 2] + 2 * (k + 1) * (c[k + 1] || 0);
  }
  d[0] *= 0.5;
  return d;
}

/**
 * 3D separable IDCT. Input cheb packed i + j*n + k*n*n, length n³ (n = N+1).
 * Output dens packed ix + iy*M + iz*M*M at Chebyshev roots in [-1,1]³.
 */
export function idctCheb3D(
  cheb: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): Idct3DResult {
  const n = deg + 1;
  const M = Math.max(n, (gridM ?? n) | 0 || n);
  const basis = getChebBasis(n, M);
  const tmp1 = new Float64Array(M * n * n);
  const row = new Float64Array(n);

  for (let j = 0; j < n; j++) {
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < n; i++) row[i] = cheb[i + j * n + k * n * n] || 0;
      applyChebBasisRow(basis, n, M, row, tmp1, j * M + k * M * n, 1);
    }
  }

  const tmp2 = new Float64Array(M * M * n);
  const rowY = new Float64Array(n);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) rowY[j] = tmp1[m + j * M + k * M * n];
      applyChebBasisRow(basis, n, M, rowY, tmp2, m + k * M * M, M);
    }
  }

  const dens = new Float32Array(M * M * M);
  const rowZ = new Float64Array(n);
  for (let m = 0; m < M; m++) {
    for (let p = 0; p < M; p++) {
      for (let k = 0; k < n; k++) rowZ[k] = tmp2[m + p * M + k * M * M];
      applyChebBasisRow(basis, n, M, rowZ, dens, m + p * M, M * M);
    }
  }

  return { dens, M, deg, n };
}

/** Resumable separable Chebyshev-root IDCT (Gauss grid). */
export interface IdctCheb3DJob {
  cheb: Float32Array | Float64Array;
  deg: number;
  M: number;
  n: number;
  pass: 0 | 1 | 2;
  cursor: number;
  tmp1: Float64Array;
  tmp2: Float64Array;
  dens: Float32Array;
  row: Float64Array;
}

export function beginIdctCheb3D(
  cheb: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): IdctCheb3DJob {
  const n = deg + 1;
  const M = Math.max(n, (gridM ?? n) | 0 || n);
  return {
    cheb,
    deg,
    M,
    n,
    pass: 0,
    cursor: 0,
    tmp1: new Float64Array(M * n * n),
    tmp2: new Float64Array(M * M * n),
    dens: new Float32Array(M * M * M),
    row: new Float64Array(n),
  };
}

export function stepIdctCheb3D(
  job: IdctCheb3DJob,
  budgetMs: number,
): { job: IdctCheb3DJob | null; done: boolean; dens?: Float32Array } {
  const t0 = performance.now();
  const { cheb, n, M } = job;
  const basis = getChebBasis(n, M);

  while (performance.now() - t0 < budgetMs) {
    if (job.pass === 0) {
      const total = n * n;
      if (job.cursor >= total) {
        job.pass = 1;
        job.cursor = 0;
        continue;
      }
      const j = (job.cursor / n) | 0;
      const k = job.cursor % n;
      for (let i = 0; i < n; i++) job.row[i] = cheb[i + j * n + k * n * n] || 0;
      applyChebBasisRow(basis, n, M, job.row, job.tmp1, j * M + k * M * n, 1);
      job.cursor++;
      continue;
    }

    if (job.pass === 1) {
      const total = M * n;
      if (job.cursor >= total) {
        job.pass = 2;
        job.cursor = 0;
        continue;
      }
      const m = (job.cursor / n) | 0;
      const k = job.cursor % n;
      for (let j = 0; j < n; j++) job.row[j] = job.tmp1[m + j * M + k * M * n]!;
      applyChebBasisRow(basis, n, M, job.row, job.tmp2, m + k * M * M, M);
      job.cursor++;
      continue;
    }

    const total = M * M;
    if (job.cursor >= total) return { job: null, done: true, dens: job.dens };
    const m = (job.cursor / M) | 0;
    const p = job.cursor % M;
    for (let k = 0; k < n; k++) job.row[k] = job.tmp2[m + p * M + k * M * M]!;
    applyChebBasisRow(basis, n, M, job.row, job.dens, m + p * M, M * M);
    job.cursor++;
  }

  return { job, done: false };
}

export function finishIdctCheb3D(job: IdctCheb3DJob): Float32Array {
  let live = job;
  while (true) {
    const step = stepIdctCheb3D(live, Infinity);
    if (step.done) return step.dens ?? live.dens;
    if (!step.job) throw new Error("IDCT stalled");
    live = step.job;
  }
}

/** One axis of ∂f/∂ξ: coeff pass then separable IDCT. */
export interface GradAxisJob {
  series: Float32Array | Float64Array;
  deg: number;
  axis: ChebAxis;
  phase: "coeff" | "idct";
  cursor: number;
  coeff: Float64Array;
  row: Float64Array;
  idct: IdctCheb3DJob | null;
}

export function beginGradAxisJob(
  series: Float32Array | Float64Array,
  deg: number,
  axis: ChebAxis,
): GradAxisJob {
  const n = deg + 1;
  return {
    series,
    deg,
    axis,
    phase: "coeff",
    cursor: 0,
    coeff: new Float64Array(n * n * n),
    row: new Float64Array(n),
    idct: null,
  };
}

export function stepGradAxisJob(
  job: GradAxisJob,
  budgetMs: number,
): { job: GradAxisJob | null; done: boolean; grad?: Float32Array } {
  const t0 = performance.now();
  const n = job.deg + 1;

  if (job.phase === "coeff") {
    while (performance.now() - t0 < budgetMs) {
      const total = n * n;
      if (job.cursor >= total) {
        job.phase = "idct";
        job.cursor = 0;
        job.idct = beginIdctCheb3D(job.coeff, job.deg, n);
        break;
      }
      const a = (job.cursor / n) | 0;
      const b = job.cursor % n;
      if (job.axis === 0) {
        for (let i = 0; i < n; i++) job.row[i] = job.series[i + a * n + b * n * n] || 0;
        const d = chebDiff1D(job.row);
        for (let i = 0; i < n; i++) job.coeff[i + a * n + b * n * n] = d[i]!;
      } else if (job.axis === 1) {
        for (let j = 0; j < n; j++) job.row[j] = job.series[a + j * n + b * n * n] || 0;
        const d = chebDiff1D(job.row);
        for (let j = 0; j < n; j++) job.coeff[a + j * n + b * n * n] = d[j]!;
      } else {
        for (let k = 0; k < n; k++) job.row[k] = job.series[a + b * n + k * n * n] || 0;
        const d = chebDiff1D(job.row);
        for (let k = 0; k < n; k++) job.coeff[a + b * n + k * n * n] = d[k]!;
      }
      job.cursor++;
    }
    if (job.phase === "coeff") return { job, done: false };
  }

  if (!job.idct) {
    job.idct = beginIdctCheb3D(job.coeff, job.deg, n);
  }
  const remaining = Math.max(0, budgetMs - (performance.now() - t0));
  const idctJob = job.idct;
  const idctStep = stepIdctCheb3D(idctJob, remaining);
  if (idctStep.done) {
    return { job: null, done: true, grad: idctStep.dens ?? idctJob.dens };
  }
  job.idct = idctStep.job;
  return { job, done: false };
}
export function idctChebGrad3D(
  cheb: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): IdctGrad3DResult {
  const n = deg + 1;
  const M = Math.max(n, (gridM ?? n) | 0 || n);
  const n3 = n * n * n;
  const cx = new Float64Array(n3);
  const cy = new Float64Array(n3);
  const cz = new Float64Array(n3);
  const row = new Float64Array(n);

  for (let j = 0; j < n; j++) {
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < n; i++) row[i] = cheb[i + j * n + k * n * n] || 0;
      const d = chebDiff1D(row);
      for (let i = 0; i < n; i++) cx[i + j * n + k * n * n] = d[i];
    }
  }
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) row[j] = cheb[i + j * n + k * n * n] || 0;
      const d = chebDiff1D(row);
      for (let j = 0; j < n; j++) cy[i + j * n + k * n * n] = d[j];
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) row[k] = cheb[i + j * n + k * n * n] || 0;
      const d = chebDiff1D(row);
      for (let k = 0; k < n; k++) cz[i + j * n + k * n * n] = d[k];
    }
  }

  return {
    gx: idctCheb3D(cx, deg, M).dens,
    gy: idctCheb3D(cy, deg, M).dens,
    gz: idctCheb3D(cz, deg, M).dens,
    M,
    deg,
    n,
  };
}

function chebCoeffsDiff3D(
  cheb: ArrayLike<number>,
  deg: number,
  axis: ChebAxis,
): Float64Array {
  const n = deg + 1;
  const n3 = n * n * n;
  const out = new Float64Array(n3);
  const row = new Float64Array(n);

  if (axis === 0) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        for (let i = 0; i < n; i++) row[i] = cheb[i + j * n + k * n * n] || 0;
        const d = chebDiff1D(row);
        for (let i = 0; i < n; i++) out[i + j * n + k * n * n] = d[i]!;
      }
    }
    return out;
  }
  if (axis === 1) {
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        for (let j = 0; j < n; j++) row[j] = cheb[i + j * n + k * n * n] || 0;
        const d = chebDiff1D(row);
        for (let j = 0; j < n; j++) out[i + j * n + k * n * n] = d[j]!;
      }
    }
    return out;
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) row[k] = cheb[i + j * n + k * n * n] || 0;
      const d = chebDiff1D(row);
      for (let k = 0; k < n; k++) out[i + j * n + k * n * n] = d[k]!;
    }
  }
  return out;
}

function chebCoeffsDiff2_3D(
  cheb: ArrayLike<number>,
  deg: number,
  axis: ChebAxis,
): Float64Array {
  return chebCoeffsDiff3D(chebCoeffsDiff3D(cheb, deg, axis), deg, axis);
}

function addChebCoeffs(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = (a[i] || 0) + (b[i] || 0);
  return out;
}

function subChebCoeffs(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < out.length; i++) out[i] = (a[i] || 0) - (b[i] || 0);
  return out;
}

/** ∇²f on the Chebyshev-root grid (ξ derivatives; multiply by (1/half)² for world). */
export function idctChebLaplacian3D(
  cheb: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): Idct3DResult {
  const n = deg + 1;
  const d2x = chebCoeffsDiff2_3D(cheb, deg, 0);
  const d2y = chebCoeffsDiff2_3D(cheb, deg, 1);
  const d2z = chebCoeffsDiff2_3D(cheb, deg, 2);
  const lap = addChebCoeffs(addChebCoeffs(d2x, d2y), d2z);
  const { dens, M } = idctCheb3D(lap, deg, gridM);
  return { dens, M, deg, n };
}

/** ∇·V from component Chebyshev fits (ξ derivatives; scale each axis by 1/half). */
export function idctChebDivergence3D(
  chebX: Float32Array | Float64Array,
  chebY: Float32Array | Float64Array,
  chebZ: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): Idct3DResult {
  const n = deg + 1;
  const dVx = chebCoeffsDiff3D(chebX, deg, 0);
  const dVy = chebCoeffsDiff3D(chebY, deg, 1);
  const dVz = chebCoeffsDiff3D(chebZ, deg, 2);
  const div = addChebCoeffs(addChebCoeffs(dVx, dVy), dVz);
  const { dens, M } = idctCheb3D(div, deg, gridM);
  return { dens, M, deg, n };
}

/** ∇×V from component Chebyshev fits (ξ derivatives; scale by 1/half). */
export function idctChebCurl3D(
  chebX: Float32Array | Float64Array,
  chebY: Float32Array | Float64Array,
  chebZ: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): IdctCurl3DResult {
  const n = deg + 1;
  const dVx_dy = chebCoeffsDiff3D(chebX, deg, 1);
  const dVx_dz = chebCoeffsDiff3D(chebX, deg, 2);
  const dVy_dx = chebCoeffsDiff3D(chebY, deg, 0);
  const dVy_dz = chebCoeffsDiff3D(chebY, deg, 2);
  const dVz_dx = chebCoeffsDiff3D(chebZ, deg, 0);
  const dVz_dy = chebCoeffsDiff3D(chebZ, deg, 2);
  const cx = subChebCoeffs(dVz_dy, dVy_dz);
  const cy = subChebCoeffs(dVx_dz, dVz_dx);
  const cz = subChebCoeffs(dVy_dx, dVx_dy);
  const M = Math.max(n, (gridM ?? n) | 0 || n);
  return {
    fx: idctCheb3D(cx, deg, M).dens,
    fy: idctCheb3D(cy, deg, M).dens,
    fz: idctCheb3D(cz, deg, M).dens,
    M,
    deg,
    n,
  };
}

/** ∂f/∂ξ_axis on the Chebyshev-root grid (ξ derivatives; scale by 1/half for world). */
export function idctChebPartial3D(
  cheb: Float32Array | Float64Array,
  deg: number,
  axis: ChebAxis,
  gridM?: number,
): Idct3DResult {
  const n = deg + 1;
  const d = chebCoeffsDiff3D(cheb, deg, axis);
  const { dens, M } = idctCheb3D(d, deg, gridM);
  return { dens, M, deg, n };
}

/** Evaluate Σ c_i T_i(ξ) at ξ ∈ [-1, 1]. */
export function evalCheb1D(coeff: ArrayLike<number>, xi: number): number {
  const n = coeff.length;
  const phase = Math.acos(Math.min(1, Math.max(-1, xi)));
  let s = 0;
  for (let i = 0; i < n; i++) {
    const c = coeff[i];
    if (c !== 0) s += c * Math.cos(i * phase);
  }
  return s;
}

/** ∫_{ξ0}^{ξ1} f(ξ) dξ via Simpson on a dense Chebyshev grid. */
export function chebDefiniteInt1D(
  coeff: ArrayLike<number>,
  xi0: number,
  xi1: number,
  gridM = 256,
): number {
  const a = Math.min(xi0, xi1);
  const b = Math.max(xi0, xi1);
  if (b - a < 1e-15) return 0;
  const h = (b - a) / gridM;
  let sum = 0;
  for (let m = 0; m <= gridM; m++) {
    const xi = a + m * h;
    const f = evalCheb1D(coeff, xi);
    if (m === 0 || m === gridM) sum += f;
    else if (m % 2 === 0) sum += 2 * f;
    else sum += 4 * f;
  }
  const sign = xi1 >= xi0 ? 1 : -1;
  return sign * (sum * h) / 3;
}

/**
 * Definite integral along one axis in coefficient space.
 * Collapses the integrated axis to T_0 (constant along that axis).
 */
export function chebDefiniteInt3D(
  cheb: ArrayLike<number>,
  deg: number,
  axis: ChebAxis,
  worldA: number,
  worldB: number,
  half: number,
): Float64Array {
  const n = deg + 1;
  const n3 = n * n * n;
  const out = new Float64Array(n3);
  const xiA = worldA / half;
  const xiB = worldB / half;
  const row = new Float64Array(n);

  if (axis === 0) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        for (let i = 0; i < n; i++) row[i] = cheb[i + j * n + k * n * n] || 0;
        out[0 + j * n + k * n * n] = chebDefiniteInt1D(row, xiA, xiB) * half;
      }
    }
    return out;
  }
  if (axis === 1) {
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        for (let j = 0; j < n; j++) row[j] = cheb[i + j * n + k * n * n] || 0;
        out[i + 0 * n + k * n * n] = chebDefiniteInt1D(row, xiA, xiB) * half;
      }
    }
    return out;
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) row[k] = cheb[i + j * n + k * n * n] || 0;
      out[i + j * n + 0 * n * n] = chebDefiniteInt1D(row, xiA, xiB) * half;
    }
  }
  return out;
}

/** Broadcast reduced-axis Cheb tensor to full n³ layout for IDCT. */
export function embedReducedCheb3D(
  cheb: ArrayLike<number>,
  deg: number,
  integratedAxes: ChebAxis[],
): Float64Array {
  const n = deg + 1;
  const n3 = n * n * n;
  const out = new Float64Array(n3);
  const axisSet = new Set(integratedAxes);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        const si = axisSet.has(0) ? 0 : i;
        const sj = axisSet.has(1) ? 0 : j;
        const sk = axisSet.has(2) ? 0 : k;
        out[i + j * n + k * n * n] = cheb[si + sj * n + sk * n * n] || 0;
      }
    }
  }
  return out;
}
