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
import { idctChebGrad3D, beginGradAxisJob, stepGradAxisJob, type GradAxisJob } from "./idct.js";
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

/** Resumable separable Lobatto IDCT → density grid. */
export interface LobattoIdct3DJob {
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

export function beginLobattoIdct3D(
  cheb: Float32Array | Float64Array,
  deg: number,
  gridM?: number,
): LobattoIdct3DJob {
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

export function stepLobattoIdct3D(
  job: LobattoIdct3DJob,
  budgetMs: number,
): { job: LobattoIdct3DJob | null; done: boolean; dens?: Float32Array } {
  const t0 = performance.now();
  const { cheb, n, M } = job;

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
      const v = lobattoIDCT1D(job.row, M);
      for (let m = 0; m < M; m++) job.tmp1[m + j * M + k * M * n] = v[m]!;
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
      const v = lobattoIDCT1D(job.row, M);
      for (let p = 0; p < M; p++) job.tmp2[m + p * M + k * M * M] = v[p]!;
      job.cursor++;
      continue;
    }

    const total = M * M;
    if (job.cursor >= total) return { job: null, done: true, dens: job.dens };
    const m = (job.cursor / M) | 0;
    const p = job.cursor % M;
    for (let k = 0; k < n; k++) job.row[k] = job.tmp2[m + p * M + k * M * M]!;
    const v = lobattoIDCT1D(job.row, M);
    for (let q = 0; q < M; q++) job.dens[m + p * M + q * M * M] = v[q]!;
    job.cursor++;
  }

  return { job, done: false };
}

export type LobattoFinalizePhase =
  | "dens_idct"
  | "grad_x"
  | "grad_y"
  | "grad_z";

/** Chunked IDCT (+ optional iso grad) after Lobatto sampling completes. */
export interface LobattoFinalizeJob {
  lob: LobattoFitState;
  role: "cloud" | "isosurface";
  phase: LobattoFinalizePhase;
  densIdct: LobattoIdct3DJob | null;
  dens?: Float32Array;
  series?: Float32Array;
  gradAxis: GradAxisJob | null;
  gx?: Float32Array;
  gy?: Float32Array;
  gz?: Float32Array;
}

export function beginLobattoFinalize(
  lob: LobattoFitState,
  role: "cloud" | "isosurface",
): LobattoFinalizeJob {
  return {
    lob,
    role,
    phase: "dens_idct",
    densIdct: beginLobattoIdct3D(lob.cheb, lob.deg, lob.deg + 1),
    gradAxis: null,
  };
}

export function stepLobattoFinalize(
  job: LobattoFinalizeJob,
  budgetMs: number,
): { job: LobattoFinalizeJob | null; done: boolean; result?: ScalarKeyframeBakeResult } {
  const t0 = performance.now();

  while (performance.now() - t0 < budgetMs) {
    if (job.phase === "dens_idct" && job.densIdct) {
      const step = stepLobattoIdct3D(job.densIdct, budgetMs - (performance.now() - t0));
      if (!step.done) {
        job.densIdct = step.job;
        return { job, done: false };
      }
      job.dens = step.dens ?? job.densIdct.dens;
      job.densIdct = null;
      if (job.role === "cloud") {
        return {
          job: null,
          done: true,
          result: {
            frame: { dens: job.dens, cheb: job.lob.cheb, fitRel: NaN },
            lobatto: job.lob,
            deg: job.lob.deg,
          },
        };
      }
      job.series = lobattoChebToSeries(job.lob.cheb, job.lob.deg);
      job.phase = "grad_x";
      continue;
    }

    const axis = job.phase === "grad_x" ? 0 : job.phase === "grad_y" ? 1 : job.phase === "grad_z" ? 2 : -1;
    if (axis < 0) return { job, done: false };

    if (!job.gradAxis) job.gradAxis = beginGradAxisJob(job.series!, job.lob.deg, axis as 0 | 1 | 2);
    const step = stepGradAxisJob(job.gradAxis, budgetMs - (performance.now() - t0));
    if (!step.done) {
      job.gradAxis = step.job;
      return { job, done: false };
    }
    if (axis === 0) job.gx = step.grad;
    else if (axis === 1) job.gy = step.grad;
    else job.gz = step.grad;
    job.gradAxis = null;

    if (job.phase === "grad_z") {
      return {
        job: null,
        done: true,
        result: {
          frame: {
            dens: job.dens!,
            cheb: job.lob.cheb,
            fitRel: NaN,
            gx: job.gx,
            gy: job.gy,
            gz: job.gz,
          },
          lobatto: job.lob,
          deg: job.lob.deg,
        },
      };
    }
    job.phase = job.phase === "grad_x" ? "grad_y" : "grad_z";
  }

  return { job, done: false };
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

/** In-progress Lobatto sample fill (refine or full resample). */
export interface LobattoBuildJob {
  targetDeg: number;
  half: number;
  mode: "refine" | "full";
  n: number;
  vals: Float64Array;
  pts: number[];
  nOld: number;
  cursor: number;
  newSamples: number;
  reusedSamples: number;
}

export interface LobattoBuildStepOpts {
  budgetMs?: number;
  maxSamples?: number;
}

export interface LobattoBuildStepResult {
  job: LobattoBuildJob | null;
  state: LobattoFitState | null;
  done: boolean;
  samples: number;
}

function lobattoStateFromVals(
  vals: Float64Array,
  n: number,
  half: number,
  newSamples: number,
  reusedSamples: number,
): LobattoFitState {
  const N = n - 1;
  const cheb = lobattoDCT3DSeparable(vals, n);
  return {
    deg: N,
    half,
    n,
    vals,
    cheb,
    uNodes: lobattoNodes(N),
    newSamples,
    reusedSamples,
  };
}

function lobattoBuildNeedsSample(job: LobattoBuildJob, ix: number, iy: number, iz: number): boolean {
  if (job.mode === "full") return true;
  return !(isNestedEvenIndex(ix) && isNestedEvenIndex(iy) && isNestedEvenIndex(iz));
}

function initRefineBuildJob(prev: LobattoFitState, newDeg: number): LobattoBuildJob {
  const N = Math.max(0, Math.min(MAX_DEG, newDeg | 0));
  const n = N + 1;
  const nOld = prev.n;
  const { pts } = lobattoWorldPoints(prev.half, N);
  const vals = new Float64Array(n * n * n);
  let reusedSamples = 0;
  for (let ix = 0; ix < n; ix++) {
    for (let iy = 0; iy < n; iy++) {
      for (let iz = 0; iz < n; iz++) {
        if (isNestedEvenIndex(ix) && isNestedEvenIndex(iy) && isNestedEvenIndex(iz)) {
          const ox = ix / 2;
          const oy = iy / 2;
          const oz = iz / 2;
          vals[ix + iy * n + iz * n * n] = prev.vals[ox + oy * nOld + oz * nOld * nOld]!;
          reusedSamples++;
        }
      }
    }
  }
  return {
    targetDeg: N,
    half: prev.half,
    mode: "refine",
    n,
    vals,
    pts,
    nOld,
    cursor: 0,
    newSamples: 0,
    reusedSamples,
  };
}

function initFullBuildJob(half: number, deg: number): LobattoBuildJob {
  const N = Math.max(0, Math.min(MAX_DEG, deg | 0));
  const n = N + 1;
  const { pts } = lobattoWorldPoints(half, N);
  return {
    targetDeg: N,
    half,
    mode: "full",
    n,
    vals: new Float64Array(n * n * n),
    pts,
    nOld: 0,
    cursor: 0,
    newSamples: 0,
    reusedSamples: 0,
  };
}

/**
 * Start (or skip) chunked advancement toward targetDeg.
 * One job covers a single ladder step: nested refine when doubling, else full resample.
 */
export function beginLobattoBuild(
  cache: LobattoFitState | null,
  half: number,
  targetDeg: number,
): { job: LobattoBuildJob | null; state: LobattoFitState | null } {
  const target = Math.max(0, Math.min(MAX_DEG, targetDeg | 0));
  if (target <= 0) return { job: null, state: cache };

  let state = cache;
  if (state && Math.abs(state.half - half) > 1e-12) state = null;
  if (state && state.deg >= target) return { job: null, state };

  if (!state) {
    return { job: initFullBuildJob(half, Math.min(4, target)), state: null };
  }

  const doubleStep = Math.min(target, state.deg * 2);
  if (doubleStep === state.deg * 2 && doubleStep > state.deg) {
    return { job: initRefineBuildJob(state, doubleStep), state };
  }
  return { job: initFullBuildJob(half, target), state };
}

/** Sample up to budgetMs / maxSamples new grid points; finalize with DCT when complete. */
export function stepLobattoBuild(
  job: LobattoBuildJob,
  fn: (x: number, y: number, z: number) => number,
  opts: LobattoBuildStepOpts = {},
): LobattoBuildStepResult {
  const budgetMs = opts.budgetMs ?? 3;
  const maxSamples = opts.maxSamples ?? Infinity;
  const n = job.n;
  const pts = job.pts;
  const nCells = n * n * n;
  const t0 = performance.now();
  let samples = 0;

  while (job.cursor < nCells && samples < maxSamples && performance.now() - t0 < budgetMs) {
    const idx = job.cursor++;
    const iz = (idx / (n * n)) | 0;
    const rem = idx % (n * n);
    const iy = (rem / n) | 0;
    const ix = rem % n;
    if (!lobattoBuildNeedsSample(job, ix, iy, iz)) continue;
    const v = fn(pts[ix]!, pts[iy]!, pts[iz]!);
    if (!Number.isFinite(v)) {
      throw new Error(`f NaN at Lobatto sample (${pts[ix]}, ${pts[iy]}, ${pts[iz]})`);
    }
    job.vals[idx] = v;
    job.newSamples++;
    samples++;
  }

  if (job.cursor < nCells) {
    return { job, state: null, done: false, samples };
  }

  const state = lobattoStateFromVals(job.vals, n, job.half, job.newSamples, job.reusedSamples);
  return { job: null, state, done: true, samples };
}

/** Drain a build job synchronously (tests / sync bake). */
export function finishLobattoBuild(
  job: LobattoBuildJob,
  fn: (x: number, y: number, z: number) => number,
): LobattoFitState {
  let live = job;
  while (true) {
    const step = stepLobattoBuild(live, fn, { budgetMs: Infinity, maxSamples: Infinity });
    if (step.done && step.state) return step.state;
    if (!step.job) throw new Error("Lobatto build stalled");
    live = step.job;
  }
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

/** Chebyshev T_0..T_deg at normalized coord u ∈ [-1,1]. */
export function chebValues(u: number, deg: number): Float64Array {
  const n = deg + 1;
  const T = new Float64Array(n);
  T[0] = 1;
  if (n > 1) T[1] = u;
  for (let i = 2; i < n; i++) T[i] = 2 * u * T[i - 1]! - T[i - 2]!;
  return T;
}

/** Per-mode weight: DCT-I coeffs → standard Chebyshev series coeffs. */
export function lobattoSeriesWeight(i: number, deg: number): number {
  return i === 0 || i === deg ? 0.5 : 1;
}

/** Evaluate Lobatto DCT-I coefficient tensor (endpoint-halving convention). */
export function evalLobattoChebTensor3D(
  cheb: ArrayLike<number>,
  deg: number,
  half: number,
  x: number,
  y: number,
  z: number,
): number {
  const n = deg + 1;
  const Tx = chebValues(x / half, deg);
  const Ty = chebValues(y / half, deg);
  const Tz = chebValues(z / half, deg);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const wi = lobattoSeriesWeight(i, deg);
    for (let j = 0; j < n; j++) {
      const wj = lobattoSeriesWeight(j, deg);
      for (let k = 0; k < n; k++) {
        const wk = lobattoSeriesWeight(k, deg);
        s +=
          (cheb[i + j * n + k * n * n] ?? 0) *
          wi *
          wj *
          wk *
          Tx[i]! *
          Ty[j]! *
          Tz[k]!;
      }
    }
  }
  return s;
}

/** Relative L2 error of Lobatto fit vs truth on uniform interior probe grid. */
export function probeRelL2Lobatto(
  cheb: ArrayLike<number>,
  deg: number,
  half: number,
  fn: (x: number, y: number, z: number) => number,
  probes = 12,
): number {
  let num = 0;
  let den = 0;
  for (let ix = 0; ix < probes; ix++) {
    for (let iy = 0; iy < probes; iy++) {
      for (let iz = 0; iz < probes; iz++) {
        const x = -half + (2 * half * (ix + 0.5)) / probes;
        const y = -half + (2 * half * (iy + 0.5)) / probes;
        const z = -half + (2 * half * (iz + 0.5)) / probes;
        const truth = fn(x, y, z);
        const approx = evalLobattoChebTensor3D(cheb, deg, half, x, y, z);
        const d = approx - truth;
        num += d * d;
        den += truth * truth;
      }
    }
  }
  return Math.sqrt(num) / (Math.sqrt(den) + 1e-15);
}

/** Evaluate tensor Cheb series at world (x,y,z). For Gauss-root DCT coefficients. */
export function evalChebTensor3D(
  cheb: ArrayLike<number>,
  deg: number,
  half: number,
  x: number,
  y: number,
  z: number,
): number {
  const n = deg + 1;
  const Tx = chebValues(x / half, deg);
  const Ty = chebValues(y / half, deg);
  const Tz = chebValues(z / half, deg);
  let s = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        s += (cheb[i + j * n + k * n * n] ?? 0) * Tx[i]! * Ty[j]! * Tz[k]!;
      }
    }
  }
  return s;
}

/** Relative L2 error of Cheb fit vs truth on uniform interior probe grid. */
export function probeRelL2Cheb(
  cheb: ArrayLike<number>,
  deg: number,
  half: number,
  fn: (x: number, y: number, z: number) => number,
  probes = 12,
): number {
  let num = 0;
  let den = 0;
  for (let ix = 0; ix < probes; ix++) {
    for (let iy = 0; iy < probes; iy++) {
      for (let iz = 0; iz < probes; iz++) {
        const x = -half + (2 * half * (ix + 0.5)) / probes;
        const y = -half + (2 * half * (iy + 0.5)) / probes;
        const z = -half + (2 * half * (iz + 0.5)) / probes;
        const truth = fn(x, y, z);
        const approx = evalChebTensor3D(cheb, deg, half, x, y, z);
        const d = approx - truth;
        num += d * d;
        den += truth * truth;
      }
    }
  }
  return Math.sqrt(num) / (Math.sqrt(den) + 1e-15);
}

/** Convert Lobatto DCT-I tensor → standard Chebyshev series coefficients. */
export function lobattoChebToSeries(cheb: ArrayLike<number>, deg: number): Float32Array {
  const n = deg + 1;
  const out = new Float32Array(n * n * n);
  for (let i = 0; i < n; i++) {
    const wi = lobattoSeriesWeight(i, deg);
    for (let j = 0; j < n; j++) {
      const wj = lobattoSeriesWeight(j, deg);
      for (let k = 0; k < n; k++) {
        const idx = i + j * n + k * n * n;
        out[idx] = (cheb[idx] ?? 0) * wi * wj * lobattoSeriesWeight(k, deg);
      }
    }
  }
  return out;
}

export interface LobattoScalarFitResult {
  dens: Float32Array;
  cheb: Float32Array;
  fitRelL2: number;
  M: number;
  deg: number;
  timing?: ChebFitTiming;
}

/** Plain scalar field via Lobatto fit → Lobatto-grid IDCT. */
export function fitScalarFieldLobatto(
  fn: (x: number, y: number, z: number) => number,
  half: number,
  deg: number,
  opts: { skipL2?: boolean } = {},
): LobattoScalarFitResult {
  const fit = fitChebyshevLobatto3D(fn, half, deg, { skipL2: opts.skipL2 ?? true });
  const idct = idctLobatto3D(fit.cheb, fit.deg, fit.deg + 1);
  return {
    dens: idct.dens,
    cheb: fit.cheb,
    fitRelL2: fit.fitRelL2,
    M: idct.M,
    deg: fit.deg,
    timing: fit.timing,
  };
}

/** Advance cached Lobatto state toward targetDeg (nested when doubling). */
export function ensureLobattoDegree(
  cache: LobattoFitState | null,
  fn: (x: number, y: number, z: number) => number,
  half: number,
  targetDeg: number,
): LobattoFitState {
  const target = Math.max(0, Math.min(MAX_DEG, targetDeg | 0));
  let state = cache;
  if (!state || Math.abs(state.half - half) > 1e-12 || state.deg > target) {
    state = fitChebyshevLobatto3D(fn, half, Math.min(4, target), { skipL2: true }).lobatto;
  }
  while (state.deg < target) {
    const next = Math.min(target, state.deg * 2);
    if (next <= state.deg) {
      state = fitChebyshevLobatto3D(fn, half, target, { skipL2: true }).lobatto;
      break;
    }
    state = refineLobatto3D(state, fn, next);
  }
  if (state.deg !== target) {
    state = fitChebyshevLobatto3D(fn, half, target, { skipL2: true }).lobatto;
  }
  return state;
}

/** Next rung on the degree ladder strictly after currentDeg, or null if at target. */
export function nextLadderDeg(currentDeg: number, targetDeg: number): number | null {
  for (const d of lobattoLadderDegrees(targetDeg)) {
    if (d > currentDeg) return d;
  }
  return null;
}

/** First ladder step (coarse start degree). */
export function startLadderDeg(targetDeg: number): number {
  return lobattoLadderDegrees(targetDeg)[0] ?? Math.max(1, targetDeg | 0);
}

export interface ScalarKeyframeBakeResult {
  frame: {
    dens: Float32Array;
    cheb: Float32Array;
    fitRel: number;
    gx?: Float32Array;
    gy?: Float32Array;
    gz?: Float32Array;
  };
  lobatto: LobattoFitState;
  deg: number;
  timing?: ChebFitTiming;
}

/** Bake one scalar keyframe slot at target degree using Lobatto (+ optional iso grad). */
export function bakeScalarKeyframeFrame(
  fn: (x: number, y: number, z: number) => number,
  half: number,
  deg: number,
  role: "cloud" | "isosurface",
  lobattoCache: LobattoFitState | null,
  stages?: { sampleMs: number; chebMs: number; idctMs: number; gradMs: number } | null,
): ScalarKeyframeBakeResult {
  const tFit = performance.now();
  const lob = ensureLobattoDegree(lobattoCache, fn, half, deg);
  if (stages) stages.sampleMs += performance.now() - tFit;

  let tStage = performance.now();
  const idct = idctLobatto3D(lob.cheb, lob.deg, lob.deg + 1);
  if (stages) stages.idctMs += performance.now() - tStage;
  const frame: ScalarKeyframeBakeResult["frame"] = {
    dens: idct.dens,
    cheb: lob.cheb,
    fitRel: NaN,
  };
  if (role === "isosurface") {
    tStage = performance.now();
    const series = lobattoChebToSeries(lob.cheb, lob.deg);
    const grad = idctChebGrad3D(series, lob.deg, lob.deg + 1);
    if (stages) stages.gradMs += performance.now() - tStage;
    frame.gx = grad.gx;
    frame.gy = grad.gy;
    frame.gz = grad.gz;
  }
  return { frame, lobatto: lob, deg: lob.deg };
}

/**
 * Chunked scalar keyframe bake: advance Lobatto sampling by budgetMs per call.
 * Pass budgetMs=null for a synchronous full bake.
 */
export function bakeScalarKeyframeFrameChunked(
  fn: (x: number, y: number, z: number) => number,
  half: number,
  deg: number,
  role: "cloud" | "isosurface",
  lobattoCache: LobattoFitState | null,
  job: LobattoBuildJob | null,
  stages?: { sampleMs: number; chebMs: number; idctMs: number; gradMs: number } | null,
  budgetMs: number | null = 3,
  finalizeJob: LobattoFinalizeJob | null = null,
): {
  result: ScalarKeyframeBakeResult | null;
  job: LobattoBuildJob | null;
  finalizeJob: LobattoFinalizeJob | null;
  complete: boolean;
  finalizePhase?: LobattoFinalizePhase;
} {
  if (budgetMs == null) {
    const result = bakeScalarKeyframeFrame(fn, half, deg, role, lobattoCache, stages);
    return { result, job: null, finalizeJob: null, complete: true };
  }

  if (finalizeJob) {
    const tFin = performance.now();
    const fin = stepLobattoFinalize(finalizeJob, budgetMs);
    const finMs = performance.now() - tFin;
    if (stages) {
      if (finalizeJob.phase === "dens_idct") stages.idctMs += finMs;
      else stages.gradMs += finMs;
    }
    if (!fin.done) {
      return {
        result: null,
        job: null,
        finalizeJob: fin.job,
        complete: false,
        finalizePhase: fin.job?.phase,
      };
    }
    return {
      result: fin.result ?? null,
      job: null,
      finalizeJob: null,
      complete: true,
    };
  }

  const tFit = performance.now();
  let liveJob = job;
  let lob = lobattoCache;

  if (!liveJob && (lob?.deg ?? 0) < deg) {
    const begun = beginLobattoBuild(lob, half, deg);
    liveJob = begun.job;
    if (!liveJob && begun.state) lob = begun.state;
  }

  if (liveJob) {
    const stepped = stepLobattoBuild(liveJob, fn, { budgetMs });
    liveJob = stepped.job;
    if (!stepped.done) {
      if (stages) stages.sampleMs += performance.now() - tFit;
      return { result: null, job: liveJob, finalizeJob: null, complete: false };
    }
    lob = stepped.state!;
  } else if ((lob?.deg ?? 0) < deg) {
    lob = ensureLobattoDegree(lob, fn, half, deg);
  }

  if (stages) stages.sampleMs += performance.now() - tFit;
  if (!lob || lob.deg < deg) {
    return { result: null, job: liveJob, finalizeJob: null, complete: false };
  }

  return {
    result: null,
    job: null,
    finalizeJob: beginLobattoFinalize(lob, role),
    complete: false,
    finalizePhase: "dens_idct",
  };
}

/** Ladder degrees for progressive preview: 4 → 8 → … → target. */
export function lobattoLadderDegrees(targetDeg: number): number[] {
  const target = Math.max(1, Math.min(MAX_DEG, targetDeg | 0));
  const steps: number[] = [];
  let d = Math.min(4, target);
  steps.push(d);
  while (d < target) {
    const next = Math.min(target, d * 2);
    if (next !== d) steps.push(next);
    d = next;
  }
  return steps;
}

/** Gauss-root world coordinate (matches shipping fit / idct grid). */
export function gaussWorld(i: number, deg: number, half: number): number {
  const M = deg + 1;
  const u = Math.cos((Math.PI * (2 * i + 1)) / (2 * M));
  return u * half;
}
