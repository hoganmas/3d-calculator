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

type ChebAxis = 0 | 1 | 2;

/** Univariate IDCT at M Chebyshev roots: v_m = Σ_{i=0}^{n-1} c_i T_i(ξ_m). */
function idctCheb1D(coeff: ArrayLike<number>, M: number): Float64Array {
  const n = coeff.length;
  const out = new Float64Array(M);
  const invM = Math.PI / M;
  for (let m = 0; m < M; m++) {
    const phase = (m + 0.5) * invM;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const c = coeff[i];
      if (c !== 0) s += c * Math.cos(i * phase);
    }
    out[m] = s;
  }
  return out;
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
  const tmp1 = new Float64Array(M * n * n);
  const row = new Float64Array(n);

  for (let j = 0; j < n; j++) {
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < n; i++) row[i] = cheb[i + j * n + k * n * n] || 0;
      const v = idctCheb1D(row, M);
      for (let m = 0; m < M; m++) tmp1[m + j * M + k * M * n] = v[m];
    }
  }

  const tmp2 = new Float64Array(M * M * n);
  const rowY = new Float64Array(n);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) rowY[j] = tmp1[m + j * M + k * M * n];
      const v = idctCheb1D(rowY, M);
      for (let p = 0; p < M; p++) tmp2[m + p * M + k * M * M] = v[p];
    }
  }

  const dens = new Float32Array(M * M * M);
  const rowZ = new Float64Array(n);
  for (let m = 0; m < M; m++) {
    for (let p = 0; p < M; p++) {
      for (let k = 0; k < n; k++) rowZ[k] = tmp2[m + p * M + k * M * M];
      const v = idctCheb1D(rowZ, M);
      for (let q = 0; q < M; q++) dens[m + p * M + q * M * M] = v[q];
    }
  }

  return { dens, M, deg, n };
}

/**
 * ∂f/∂ξ, ∂f/∂η, ∂f/∂ζ on the same Chebyshev-root grid as idctCheb3D.
 */
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
