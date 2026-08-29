/**
 * Chebyshev–Lobatto spectral fit with nested node sets.
 *
 * Nodes: u_j = cos(π j / N), j = 0..N  (degree N → N+1 nodes per axis)
 *
 * Nesting: when refining N → 2N, old node j maps to new node 2j, so prior
 * samples are reused and only odd-index nodes need fresh evaluation.
 *
 * Convention (Clenshaw–Curtis / DCT-I):
 *   c_k = (2/N) Σ_j w_j f_j cos(π k j / N),  w_0 = w_N = ½, else 1
 * Inverse at Lobatto nodes:
 *   f_j = ½ c_0 + Σ_{k=1}^{N-1} c_k cos(π k j / N) + ½ c_N cos(π j)
 */

import { MAX_DEG } from "./limits.js";
import type { ChebFitResult, ChebFitTiming } from "../types/models.js";

/** Lobatto nodes u_j = cos(π j / deg), j = 0..deg. Length deg+1. */
export function lobattoNodes(deg: number): number[] {
  const N = Math.max(0, deg | 0);
  const nodes = new Array<number>(N + 1);
  if (N === 0) {
    nodes[0] = 1;
    return nodes;
  }
  for (let j = 0; j <= N; j++) nodes[j] = Math.cos((Math.PI * j) / N);
  return nodes;
}

/** Map old Lobatto index i (deg N) → nested index in deg 2N grid. */
export function nestedLobattoIndex(i: number): number {
  return i * 2;
}

/** True when index is an even nested slot (sampled at a coarser level). */
export function isNestedEvenIndex(i: number): boolean {
  return i % 2 === 0;
}

function fromUnit(u: number, half: number): number {
  return u * half;
}

/** 1D Lobatto DCT-I. vals length n = deg+1. */
export function lobattoDCT1D(vals: ArrayLike<number>): Float64Array {
  const n = vals.length;
  const N = n - 1;
  const out = new Float64Array(n);
  if (N === 0) {
    out[0] = vals[0] ?? 0;
    return out;
  }
  const invN = Math.PI / N;
  for (let k = 0; k < n; k++) {
    let s = 0.5 * (vals[0] ?? 0);
    for (let j = 1; j < n - 1; j++) s += (vals[j] ?? 0) * Math.cos(k * j * invN);
    s += 0.5 * (vals[n - 1] ?? 0) * Math.cos(k * N * invN);
    out[k] = (2 / N) * s;
  }
  return out;
}

/** 1D Lobatto inverse (IDCT-I) at M Lobatto nodes (default M = n). */
export function lobattoIDCT1D(coeff: ArrayLike<number>, gridM?: number): Float64Array {
  const n = coeff.length;
  const N = n - 1;
  const M = gridM ?? n;
  const out = new Float64Array(M);
  if (N === 0) {
    out.fill(coeff[0] ?? 0);
    return out;
  }
  const denom = M - 1;
  for (let j = 0; j < M; j++) {
    let s = 0.5 * (coeff[0] ?? 0);
    for (let k = 1; k < n - 1; k++) s += (coeff[k] ?? 0) * Math.cos((Math.PI * k * j) / denom);
    s += 0.5 * (coeff[n - 1] ?? 0) * Math.cos((Math.PI * (n - 1) * j) / denom);
    out[j] = s;
  }
  return out;
}

/** Separable 3D Lobatto DCT. vals packed x + y*n + z*n². */
export function lobattoDCT3DSeparable(vals: Float64Array, n: number): Float32Array {
  const N = n - 1;
  const n2 = n * n;
  const tmp = new Float64Array(n * n * n);
  const tmp2 = new Float64Array(n * n * n);
  const out = new Float32Array(n * n * n);
  const invN = N > 0 ? Math.PI / N : 0;

  // X
  for (let y = 0; y < n; y++) {
    for (let z = 0; z < n; z++) {
      const base = y * n + z * n2;
      for (let k = 0; k < n; k++) {
        let sum = 0.5 * vals[base];
        for (let j = 1; j < n - 1; j++) sum += vals[base + j] * Math.cos(k * j * invN);
        sum += 0.5 * vals[base + n - 1] * Math.cos(k * N * invN);
        tmp[base + k] = N > 0 ? (2 / N) * sum : sum;
      }
    }
  }

  // Y: tmp[x,y,z] → tmp2[x,k,z]
  for (let x = 0; x < n; x++) {
    for (let z = 0; z < n; z++) {
      for (let k = 0; k < n; k++) {
        let sum = 0.5 * tmp[x + z * n2];
        for (let j = 1; j < n - 1; j++) sum += tmp[x + j * n + z * n2] * Math.cos(k * j * invN);
        sum += 0.5 * tmp[x + (n - 1) * n + z * n2] * Math.cos(k * N * invN);
        tmp2[x + k * n + z * n2] = N > 0 ? (2 / N) * sum : sum;
      }
    }
  }

  // Z
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      const xy = x + y * n;
      for (let k = 0; k < n; k++) {
        let sum = 0.5 * tmp2[xy];
        for (let j = 1; j < n - 1; j++) sum += tmp2[xy + j * n2] * Math.cos(k * j * invN);
        sum += 0.5 * tmp2[xy + (n - 1) * n2] * Math.cos(k * N * invN);
        out[xy + k * n2] = N > 0 ? (2 / N) * sum : sum;
      }
    }
  }

  return out;
}

/** Separable 3D Lobatto IDCT → density on Lobatto node grid. */
export function idctLobatto3D(
  cheb: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): { dens: Float32Array; M: number; deg: number; n: number } {
  const n = deg + 1;
  const M = Math.max(n, (gridM ?? n) | 0 || n);
  const tmp1 = new Float64Array(M * n * n);
  const row = new Float64Array(n);

  for (let j = 0; j < n; j++) {
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < n; i++) row[i] = cheb[i + j * n + k * n * n] || 0;
      const v = lobattoIDCT1D(row, M);
      for (let m = 0; m < M; m++) tmp1[m + j * M + k * M * n] = v[m]!;
    }
  }

  const tmp2 = new Float64Array(M * M * n);
  const rowY = new Float64Array(n);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < n; j++) rowY[j] = tmp1[m + j * M + k * M * n]!;
      const v = lobattoIDCT1D(rowY, M);
      for (let p = 0; p < M; p++) tmp2[m + p * M + k * M * M] = v[p]!;
    }
  }

  const dens = new Float32Array(M * M * M);
  const rowZ = new Float64Array(n);
  for (let m = 0; m < M; m++) {
    for (let p = 0; p < M; p++) {
      for (let k = 0; k < n; k++) rowZ[k] = tmp2[m + p * M + k * M * M]!;
      const v = lobattoIDCT1D(rowZ, M);
      for (let q = 0; q < M; q++) dens[m + p * M + q * M * M] = v[q]!;
    }
  }

  return { dens, M, deg, n };
}

export interface LobattoFitState {
  deg: number;
  half: number;
  n: number;
  vals: Float64Array;
  cheb: Float32Array;
  uNodes: number[];
  newSamples: number;
  reusedSamples: number;
}

function lobattoWorldPoints(half: number, deg: number): { uNodes: number[]; pts: number[] } {
  const uNodes = lobattoNodes(deg);
  const pts = uNodes.map((u) => fromUnit(u, half));
  return { uNodes, pts };
}

/** Full Lobatto fit (no prior state). */
export function fitChebyshevLobatto3D(
  fn: (x: number, y: number, z: number) => number,
  half: number,
  deg: number,
  opts: { skipL2?: boolean } = {},
): ChebFitResult & { lobatto: LobattoFitState } {
  const tAll = performance.now();
  const N = Math.max(0, Math.min(MAX_DEG, deg | 0));
  const n = N + 1;
  const { uNodes, pts } = lobattoWorldPoints(half, N);

  let t0 = performance.now();
  const vals = new Float64Array(n * n * n);
  let fMin = Infinity;
  let fMax = -Infinity;
  for (let ix = 0; ix < n; ix++) {
    for (let iy = 0; iy < n; iy++) {
      for (let iz = 0; iz < n; iz++) {
        const v = fn(pts[ix]!, pts[iy]!, pts[iz]!);
        if (!Number.isFinite(v)) throw new Error(`f NaN at Lobatto sample (${pts[ix]}, ${pts[iy]}, ${pts[iz]})`);
        vals[ix + iy * n + iz * n * n] = v;
        fMin = Math.min(fMin, v);
        fMax = Math.max(fMax, v);
      }
    }
  }
  const sampleMs = performance.now() - t0;

  t0 = performance.now();
  const cheb = lobattoDCT3DSeparable(vals, n);
  const chebMs = performance.now() - t0;

  let fitRelL2 = NaN;
  let l2Ms = 0;
  if (!opts.skipL2) {
    t0 = performance.now();
    const { dens, M } = idctLobatto3D(cheb, N, n);
    let num = 0;
    let den = 0;
    const probes = 10;
    for (let ix = 0; ix < probes; ix++) {
      for (let iy = 0; iy < probes; iy++) {
        for (let iz = 0; iz < probes; iz++) {
          const x = -half + (2 * half * (ix + 0.5)) / probes;
          const y = -half + (2 * half * (iy + 0.5)) / probes;
          const z = -half + (2 * half * (iz + 0.5)) / probes;
          const truth = fn(x, y, z);
          // Trilinear-ish: nearest Lobatto grid cell center for quick L2 probe
          const approx = dens[densIndexNearest(ix, iy, iz, probes, M)] ?? 0;
          const d = approx - truth;
          num += d * d;
          den += truth * truth;
        }
      }
    }
    fitRelL2 = Math.sqrt(num) / (Math.sqrt(den) + 1e-15);
    l2Ms = performance.now() - t0;
  }

  const timing: ChebFitTiming = { sampleMs, chebMs, monoMs: 0, l2Ms, totalMs: performance.now() - tAll };
  const lobatto: LobattoFitState = {
    deg: N,
    half,
    n,
    vals,
    cheb,
    uNodes,
    newSamples: n * n * n,
    reusedSamples: 0,
  };
  return { cheb, mono: null, deg: N, half, fitRelL2, fMin, fMax, timing, lobatto };
}

function densIndexNearest(ix: number, iy: number, iz: number, probes: number, M: number): number {
  const mx = Math.min(M - 1, Math.round((ix / (probes - 1)) * (M - 1)));
  const my = Math.min(M - 1, Math.round((iy / (probes - 1)) * (M - 1)));
  const mz = Math.min(M - 1, Math.round((iz / (probes - 1)) * (M - 1)));
  return mx + my * M + mz * M * M;
}

/**
 * Refine an existing Lobatto fit from deg N to deg newDeg (typically 2N).
 * Reuses samples at nested even indices; evaluates only odd-index nodes.
 */
export function refineLobatto3D(
  prev: LobattoFitState,
  fn: (x: number, y: number, z: number) => number,
  newDeg: number,
): LobattoFitState {
  const N = Math.max(0, Math.min(MAX_DEG, newDeg | 0));
  if (N <= prev.deg) return prev;
  if (N !== prev.deg * 2) {
    // General path: full resample (non-doubling step)
    const full = fitChebyshevLobatto3D(fn, prev.half, N, { skipL2: true });
    return full.lobatto;
  }

  const nOld = prev.n;
  const n = N + 1;
  const { pts } = lobattoWorldPoints(prev.half, N);
  const vals = new Float64Array(n * n * n);
  let newSamples = 0;
  let reusedSamples = 0;

  for (let ix = 0; ix < n; ix++) {
    for (let iy = 0; iy < n; iy++) {
      for (let iz = 0; iz < n; iz++) {
        const idx = ix + iy * n + iz * n * n;
        if (isNestedEvenIndex(ix) && isNestedEvenIndex(iy) && isNestedEvenIndex(iz)) {
          const ox = ix / 2;
          const oy = iy / 2;
          const oz = iz / 2;
          vals[idx] = prev.vals[ox + oy * nOld + oz * nOld * nOld]!;
          reusedSamples++;
        } else {
          vals[idx] = fn(pts[ix]!, pts[iy]!, pts[iz]!);
          newSamples++;
        }
      }
    }
  }

  const cheb = lobattoDCT3DSeparable(vals, n);
  return {
    deg: N,
    half: prev.half,
    n,
    vals,
    cheb,
    uNodes: lobattoNodes(N),
    newSamples,
    reusedSamples,
  };
}

/** Degree ladder 4 → 8 → … → targetDeg using nested refinement when doubling. */
export function fitChebyshevLobattoProgressive(
  fn: (x: number, y: number, z: number) => number,
  half: number,
  targetDeg: number,
  onStep?: (state: LobattoFitState) => void,
): LobattoFitState {
  const target = Math.max(0, Math.min(MAX_DEG, targetDeg | 0));
  let deg = Math.min(4, target);
  let state = fitChebyshevLobatto3D(fn, half, deg, { skipL2: true }).lobatto;
  onStep?.(state);

  while (state.deg < target) {
    const next = Math.min(target, state.deg * 2);
    state = refineLobatto3D(state, fn, next);
    onStep?.(state);
  }
  return state;
}

/** Lobatto world coordinate for grid index j on degree-N grid in [-half, half]. */
export function lobattoWorld(j: number, deg: number, half: number): number {
  return fromUnit(lobattoNodes(deg)[j]!, half);
}
