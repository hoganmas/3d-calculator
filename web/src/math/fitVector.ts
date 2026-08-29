/**
 * Vector field expressions: tuple / gradient LaTeX → Chebyshev velocity volumes.
 */

import { compile, ComputeEngine } from "@cortex-js/compute-engine";
import { fitChebyshev3D } from "./fit.js";
import { idctCheb3D, idctChebCurl3D, idctChebGrad3D } from "./idct.js";
import {
  extractTriple,
  normalizeCalcLatex,
  scalarFromUnaryOpJson,
  tripleFromUnaryOpJson,
} from "./calcOps.js";
import type {
  ClassifiedVectorExpr,
  CompiledVectorExpr,
  VectorFieldKind,
  VectorFitResult,
} from "../types/models.js";

const ce = new ComputeEngine();

type CeRun = (scope: Record<string, unknown>) => unknown;

interface CeCompileResult {
  success: boolean;
  unsupported?: string[];
  run: CeRun | null;
  freeSymbols?: Iterable<unknown>;
}

const SPATIAL = new Set(["x", "y", "z", "r", "theta", "phi", "rho"]);
const RESERVED = new Set([...SPATIAL]);
const KNOWN_FNS = new Set([
  "sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs",
  "grad", "nabla", "laplacian", "div", "curl",
]);

function jsonArr(j: unknown): unknown[] {
  return Array.isArray(j) ? j : [];
}

function polarFromCartesian(x: number, y: number, z: number) {
  const rho = Math.hypot(x, y);
  const r = Math.hypot(rho, z);
  const phi = Math.atan2(y, x);
  const theta = r > 1e-15 ? Math.acos(Math.min(1, Math.max(-1, z / r))) : 0;
  return { r, theta, phi, rho };
}

function coerceNumber(v: unknown): number {
  if (typeof v === "number") return v;
  if (v && typeof (v as { valueOf?: () => unknown }).valueOf === "function") {
    return Number((v as { valueOf: () => unknown }).valueOf());
  }
  return Number(v);
}

function collectFreeParams(freeSymbols: Iterable<unknown> | null | undefined, latex: string) {
  const ids: string[] = [];
  let usesSpace = false;
  for (const s of freeSymbols ?? []) {
    const id = String(s);
    if (SPATIAL.has(id)) usesSpace = true;
    if (RESERVED.has(id)) continue;
    if (KNOWN_FNS.has(id.toLowerCase())) continue;
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id)) {
      throw new Error(`Invalid parameter “${id}” (use letters / digits)`);
    }
    ids.push(id);
  }
  const compact = String(latex ?? "").replace(/\s+/g, "");
  if (/^[A-Za-z]+$/.test(compact) && KNOWN_FNS.has(compact.toLowerCase())) {
    return { freeParams: [], usesSpace: false };
  }
  ids.sort();
  return { freeParams: ids, usesSpace };
}

function compileLatex(latex: string): CeCompileResult {
  let box;
  try {
    box = ce.parse(latex);
  } catch {
    return { success: false, unsupported: ["parse"], run: null, freeSymbols: [] };
  }
  return compile(box) as CeCompileResult;
}

function bindScalar(latex: string, freeParams: string[]) {
  const result = compileLatex(latex);
  if (!result?.success || typeof result.run !== "function") {
    const why = result?.unsupported?.length
      ? `unsupported: ${result.unsupported.join(", ")}`
      : "could not compile";
    throw new Error(`Expression ${why}`);
  }
  const { run } = result;
  return (params: Record<string, number> = {}) =>
    (x: number, y: number, z: number) => {
      const { r, theta, phi, rho } = polarFromCartesian(x, y, z);
      const scope: Record<string, number> = { x, y, z, r, theta, phi, rho };
      for (const name of freeParams) {
        const v = params[name];
        scope[name] = Number.isFinite(v) ? v : 1;
      }
      return coerceNumber(run(scope));
    };
}

function bindVectorFromScalar(
  scalarFn: (x: number, y: number, z: number) => number,
  eps = 1e-5,
): (x: number, y: number, z: number) => [number, number, number] {
  return (x: number, y: number, z: number) => {
    const dfx = (scalarFn(x + eps, y, z) - scalarFn(x - eps, y, z)) / (2 * eps);
    const dfy = (scalarFn(x, y + eps, z) - scalarFn(x, y - eps, z)) / (2 * eps);
    const dfz = (scalarFn(x, y, z + eps) - scalarFn(x, y, z - eps)) / (2 * eps);
    return [dfx, dfy, dfz];
  };
}

function bindCurlFromVector(
  vectorFn: (x: number, y: number, z: number) => [number, number, number],
  eps = 1e-5,
): (x: number, y: number, z: number) => [number, number, number] {
  return (x: number, y: number, z: number) => {
    const [vx0, vy0, vz0] = vectorFn(x, y, z);
    const dVy_dz =
      (vectorFn(x, y, z + eps)[1]! - vectorFn(x, y, z - eps)[1]!) / (2 * eps);
    const dVz_dy =
      (vectorFn(x, y + eps, z)[2]! - vectorFn(x, y - eps, z)[2]!) / (2 * eps);
    const dVz_dx =
      (vectorFn(x + eps, y, z)[2]! - vectorFn(x - eps, y, z)[2]!) / (2 * eps);
    const dVx_dz =
      (vectorFn(x, y, z + eps)[0]! - vectorFn(x, y, z - eps)[0]!) / (2 * eps);
    const dVy_dx =
      (vectorFn(x + eps, y, z)[1]! - vectorFn(x - eps, y, z)[1]!) / (2 * eps);
    const dVx_dy =
      (vectorFn(x, y + eps, z)[0]! - vectorFn(x, y - eps, z)[0]!) / (2 * eps);
    void vx0;
    void vy0;
    void vz0;
    return [dVz_dy - dVy_dz, dVx_dz - dVz_dx, dVy_dx - dVx_dy];
  };
}

/** Normalize vector-calculus LaTeX for CE parsing. */
export function normalizeVectorLatex(latex: string): string {
  return normalizeCalcLatex(latex);
}

export { extractTriple } from "./calcOps.js";

function isUnaryOpErrorNode(node: unknown, names: RegExp): boolean {
  if (!Array.isArray(node) || node[0] !== "Error") return false;
  const msg = String(node[1] ?? "").toLowerCase();
  const lit = node[2];
  const latexStr =
    Array.isArray(lit) && lit[0] === "LatexString"
      ? String(lit[1] ?? "").replace(/^'+|'+$/g, "").toLowerCase()
      : "";
  return msg.includes("unexpected-command") && names.test(latexStr);
}

function isGradErrorNode(node: unknown): boolean {
  return isUnaryOpErrorNode(node, /\\?(grad|nabla)/);
}

function isCurlErrorNode(node: unknown): boolean {
  return isUnaryOpErrorNode(node, /\\?curl/);
}

function scalarFromGradJson(json: unknown): string | null {
  if (!Array.isArray(json)) return null;
  const head = String(json[0]);
  const inner = scalarFromUnaryOpJson(json, "grad");
  if (inner) return inner;
  if ((head === "grad" || head === "Gradient" || head === "nabla") && json[1] != null) {
    return ce.box(json[1] as never).latex;
  }
  if (head === "Tuple" && json.length >= 3 && isGradErrorNode(json[1]) && json[2] != null) {
    return ce.box(json[2] as never).latex;
  }
  return null;
}

function looksLikeGrad(src: string, json: unknown): string | null {
  const fromJson = scalarFromGradJson(json);
  if (fromJson) return fromJson.trim();

  const lower = src.toLowerCase();
  if (/\\nabla\s*\^|\^2|\\laplacian|\\Delta|\\nabla\s*\\cdot|\\nabla\s*\\times/i.test(lower)) {
    return null;
  }
  if (/\\grad|\\nabla|\\operatorname\s*\{\s*grad/.test(lower)) {
    const m = src.match(
      /\\(?:operatorname\s*\{\s*grad\s*\}|grad|nabla)\s*(?:\\left)?[\{\(]?\s*([\s\S]+?)\s*(?:\\right)?[\}\)]?\s*$/,
    );
    if (m?.[1]) return m[1].trim();
  }
  return null;
}

function tripleFromUnaryOpWithError(
  json: unknown,
  op: string,
  errorTest: (node: unknown) => boolean,
): string[] | null {
  const fromJson = tripleFromUnaryOpJson(json, op);
  if (fromJson?.length === 3) return fromJson;
  if (!Array.isArray(json)) return null;
  const head = String(json[0]);
  if (head === "Tuple" && json.length >= 3 && errorTest(json[1]) && json[2] != null) {
    return extractTriple(json[2]);
  }
  return null;
}

function looksLikeCurl(src: string, json: unknown): string[] | null {
  const fromJson = tripleFromUnaryOpWithError(json, "curl", isCurlErrorNode);
  if (fromJson?.length === 3) return fromJson;

  if (/\\curl|\\operatorname\s*\{\s*curl/i.test(src)) {
    const m = src.match(
      /\\(?:operatorname\s*\{\s*curl\s*\}|curl)\s*(?:\\left)?[\{\(]?\s*([\s\S]+?)\s*(?:\\right)?[\}\)]?\s*$/,
    );
    if (m?.[1]) {
      try {
        const inner = ce.parse(m[1].trim());
        const j = inner?.json ?? (typeof inner?.toJSON === "function" ? inner.toJSON() : null);
        const triple = extractTriple(j);
        if (triple?.length === 3) return triple;
      } catch {
        /* fall through */
      }
    }
  }
  return null;
}

/** Quick check for auto role inference. */
export function isVectorFieldLatex(raw: string): boolean {
  try {
    classifyVectorExpr(raw);
    return true;
  } catch {
    return false;
  }
}

export function classifyVectorExpr(raw: string): ClassifiedVectorExpr {
  const src = normalizeVectorLatex(String(raw ?? "").trim());
  if (!src) throw new Error("Empty vector expression");

  let box;
  try {
    box = ce.parse(src);
  } catch {
    throw new Error("Could not parse vector expression");
  }
  const j = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);

  const gradInner = looksLikeGrad(src, j);
  if (gradInner) {
    return {
      kind: "gradient",
      label: "gradient field",
      compileParts: [gradInner],
    };
  }

  const curlParts = looksLikeCurl(src, j);
  if (curlParts) {
    return {
      kind: "curl",
      label: "curl field",
      compileParts: curlParts,
    };
  }

  const triple = extractTriple(j);
  if (triple?.length === 3) {
    return {
      kind: "tuple",
      label: "vector tuple",
      compileParts: triple,
    };
  }

  throw new Error(
    "Flow fields need a 3-component tuple like (Fx, Fy, Fz), \\grad f, or \\curl(Fx,Fy,Fz)",
  );
}

export function compileVectorExpr(raw: string): CompiledVectorExpr {
  const classified = classifyVectorExpr(raw);

  if (classified.kind === "gradient") {
    const scalarLatex = classified.compileParts[0]!;
    const scalarResult = compileLatex(scalarLatex);
    if (!scalarResult?.success || typeof scalarResult.run !== "function") {
      throw new Error("Could not compile scalar inside gradient");
    }
    const { freeParams, usesSpace } = collectFreeParams(scalarResult.freeSymbols, scalarLatex);
    if (!usesSpace) throw new Error("Gradient field must depend on x, y, or z");

    return {
      freeParams,
      usesSpace,
      kind: "gradient",
      classifyLabel: classified.label,
      scalarCompileLatex: scalarLatex,
      bind(params: Record<string, number> = {}) {
        const scalar = bindScalar(scalarLatex, freeParams)(params);
        return bindVectorFromScalar(scalar);
      },
      bindScalar(params: Record<string, number> = {}) {
        return bindScalar(scalarLatex, freeParams)(params);
      },
    };
  }

  if (classified.kind === "curl") {
    const compLatex = classified.compileParts;
    const compiled = compLatex.map((latex) => {
      const r = compileLatex(latex);
      if (!r?.success || typeof r.run !== "function") {
        throw new Error(`Could not compile curl component: ${latex}`);
      }
      return r;
    });

    const freeSet = new Set<string>();
    let usesSpace = false;
    for (let i = 0; i < compiled.length; i++) {
      const { freeParams, usesSpace: us } = collectFreeParams(
        compiled[i]!.freeSymbols,
        compLatex[i]!,
      );
      for (const p of freeParams) freeSet.add(p);
      if (us) usesSpace = true;
    }
    const freeParams = [...freeSet].sort();
    if (!usesSpace) throw new Error("Curl field must depend on x, y, or z");

    const bindTuple = (params: Record<string, number> = {}) => {
      const fns = compLatex.map((latex, i) => {
        const fp = collectFreeParams(compiled[i]!.freeSymbols, latex).freeParams;
        const run = compiled[i]!.run!;
        return (x: number, y: number, z: number) => {
          const { r, theta, phi, rho } = polarFromCartesian(x, y, z);
          const scope: Record<string, number> = { x, y, z, r, theta, phi, rho };
          for (const name of fp) {
            const v = params[name];
            scope[name] = Number.isFinite(v) ? v : 1;
          }
          return coerceNumber(run(scope));
        };
      });
      return (x: number, y: number, z: number): [number, number, number] => [
        fns[0]!(x, y, z),
        fns[1]!(x, y, z),
        fns[2]!(x, y, z),
      ];
    };

    return {
      freeParams,
      usesSpace,
      kind: "curl",
      classifyLabel: classified.label,
      bind(params: Record<string, number> = {}) {
        return bindCurlFromVector(bindTuple(params));
      },
      bindTuple,
    };
  }

  const compLatex = classified.compileParts;
  const compiled = compLatex.map((latex) => {
    const r = compileLatex(latex);
    if (!r?.success || typeof r.run !== "function") {
      throw new Error(`Could not compile vector component: ${latex}`);
    }
    return r;
  });

  const freeSet = new Set<string>();
  let usesSpace = false;
  for (let i = 0; i < compiled.length; i++) {
    const { freeParams, usesSpace: us } = collectFreeParams(
      compiled[i]!.freeSymbols,
      compLatex[i]!,
    );
    for (const p of freeParams) freeSet.add(p);
    if (us) usesSpace = true;
  }
  const freeParams = [...freeSet].sort();
  if (!usesSpace) throw new Error("Vector field must depend on x, y, or z");

  return {
    freeParams,
    usesSpace,
    kind: "tuple",
    classifyLabel: classified.label,
    bind(params: Record<string, number> = {}) {
      const fns = compLatex.map((latex, i) => {
        const fp = collectFreeParams(compiled[i]!.freeSymbols, latex).freeParams;
        const run = compiled[i]!.run!;
        return (x: number, y: number, z: number) => {
          const { r, theta, phi, rho } = polarFromCartesian(x, y, z);
          const scope: Record<string, number> = { x, y, z, r, theta, phi, rho };
          for (const name of fp) {
            const v = params[name];
            scope[name] = Number.isFinite(v) ? v : 1;
          }
          return coerceNumber(run(scope));
        };
      });
      return (x: number, y: number, z: number): [number, number, number] => [
        fns[0]!(x, y, z),
        fns[1]!(x, y, z),
        fns[2]!(x, y, z),
      ];
    },
  };
}

/** 1 where |V| is nonzero, else 0 — masks stagnation without magnitude weighting. */
export function flowPresenceSlice(
  fx: Float32Array,
  fy: Float32Array,
  fz: Float32Array,
  M: number,
): Float32Array {
  const volN = M * M * M;
  const out = new Float32Array(volN);
  for (let i = 0; i < volN; i++) {
    out[i] = Math.hypot(fx[i]!, fy[i]!, fz[i]!) > 1e-5 ? 1 : 0;
  }
  return out;
}

/** |V| on the Chebyshev grid. */
export function flowSpeedSlice(
  fx: Float32Array,
  fy: Float32Array,
  fz: Float32Array,
  M: number,
): Float32Array {
  const volN = M * M * M;
  const out = new Float32Array(volN);
  for (let i = 0; i < volN; i++) {
    out[i] = Math.hypot(fx[i]!, fy[i]!, fz[i]!);
  }
  return out;
}

/** Min/max |V| over a velocity grid (ignores near-stagnation cells). */
export function flowSpeedMinMax(
  fx: Float32Array,
  fy: Float32Array,
  fz: Float32Array,
): [min: number, max: number] {
  let vmin = Infinity;
  let vmax = 0;
  const n = fx.length;
  for (let i = 0; i < n; i++) {
    const s = Math.hypot(fx[i]!, fy[i]!, fz[i]!);
    if (s <= 1e-8) continue;
    if (s < vmin) vmin = s;
    if (s > vmax) vmax = s;
  }
  if (!Number.isFinite(vmin) || vmax <= vmin) return [0, 1];
  return [vmin, vmax];
}

/** Robust speed range on a velocity grid (percentile spread avoids outlier compression). */
export function flowSpeedPercentileMinMax(
  fx: Float32Array,
  fy: Float32Array,
  fz: Float32Array,
  loPct = 0.05,
  hiPct = 0.95,
): [min: number, max: number] {
  const speeds: number[] = [];
  const n = fx.length;
  for (let i = 0; i < n; i++) {
    const s = Math.hypot(fx[i]!, fy[i]!, fz[i]!);
    if (s > 1e-8) speeds.push(s);
  }
  if (speeds.length < 2) return flowSpeedMinMax(fx, fy, fz);
  speeds.sort((a, b) => a - b);
  const lo = speeds[Math.floor(speeds.length * loPct)]!;
  const hi = speeds[Math.min(speeds.length - 1, Math.floor(speeds.length * hiPct))]!;
  if (!(hi > lo)) return flowSpeedMinMax(fx, fy, fz);
  return [lo, hi];
}

/** Min/max speed among live particles and their trail history. */
export function flowParticleSpeedMinMax(
  posAge: Float32Array,
  trailHist: Float32Array | null,
  count: number,
  trailSteps: number,
): [min: number, max: number] | null {
  let vmin = Infinity;
  let vmax = 0;
  let samples = 0;
  const note = (s: number) => {
    if (s <= 1e-8) return;
    samples++;
    if (s < vmin) vmin = s;
    if (s > vmax) vmax = s;
  };
  for (let i = 0; i < count; i++) {
    const o = i * FLOW_PARTICLE_STRIDE;
    note(posAge[o + 4]!);
    if (trailHist && trailSteps >= 2) {
      const ho = flowTrailBaseIndex(i);
      for (let j = 0; j < trailSteps; j++) {
        note(trailHist[ho + j * FLOW_TRAIL_SLOT_STRIDE + 4]!);
      }
    }
  }
  if (samples < 2 || !Number.isFinite(vmin)) return null;
  if (vmax <= vmin) {
    const s = Math.max(vmax, 1e-8);
    return [Math.max(0, s * 0.55), s * 1.05];
  }
  return [vmin, vmax];
}

/** Max |V| stored in trail history (same source the particle shader colors from). */
export function flowTrailSpeedMax(
  trailHist: Float32Array,
  count: number,
  trailSteps: number,
): number | null {
  let vmax = 0;
  let samples = 0;
  for (let i = 0; i < count; i++) {
    const ho = flowTrailBaseIndex(i);
    for (let j = 0; j < trailSteps; j++) {
      const s = trailHist[ho + j * FLOW_TRAIL_SLOT_STRIDE + 4]!;
      if (s <= 1e-8) continue;
      samples++;
      if (s > vmax) vmax = s;
    }
  }
  if (samples === 0) return null;
  return vmax;
}

/** Trail speed distribution (same buffer the shader reads for color). */
export function flowTrailSpeedStats(
  trailHist: Float32Array,
  count: number,
  trailSteps: number,
): {
  min: number;
  max: number;
  mean: number;
  nonzero: number;
  total: number;
} | null {
  let vmin = Infinity;
  let vmax = 0;
  let sum = 0;
  let nonzero = 0;
  let total = 0;
  for (let i = 0; i < count; i++) {
    const ho = flowTrailBaseIndex(i);
    for (let j = 0; j < trailSteps; j++) {
      total++;
      const s = trailHist[ho + j * FLOW_TRAIL_SLOT_STRIDE + 4]!;
      if (s <= 1e-8) continue;
      nonzero++;
      sum += s;
      if (s < vmin) vmin = s;
      if (s > vmax) vmax = s;
    }
  }
  if (nonzero === 0) return null;
  return { min: vmin, max: vmax, mean: sum / nonzero, nonzero, total };
}

function flowColorSpeedSpan(hi: number): number {
  return Math.max(hi * 0.12, 1e-6);
}

/** Particle color range: trail min–max, widened with field percentiles when spread is tiny. */
export function resolveFlowParticleColorRange(
  trailHist: Float32Array | null,
  count: number,
  trailSteps: number,
  vRef: number,
  fieldRange: [number, number] | null = null,
): [min: number, max: number] {
  const stats = trailHist && trailSteps >= 2
    ? flowTrailSpeedStats(trailHist, count, trailSteps)
    : null;

  if (stats && stats.max > 1e-8) {
    let lo = stats.min;
    let hi = stats.max;
    const span = hi - lo;
    const minSpan = flowColorSpeedSpan(hi);
    if (span < minSpan * 0.35 && fieldRange && fieldRange[1] > fieldRange[0] + 1e-8) {
      lo = fieldRange[0];
      hi = fieldRange[1];
    } else if (span < minSpan) {
      lo = Math.max(0, hi - minSpan);
    }
    return [lo, hi];
  }

  if (fieldRange && fieldRange[1] > 1e-8) {
    const lo = fieldRange[0];
    const hi = fieldRange[1];
    const span = hi - lo;
    const minSpan = flowColorSpeedSpan(hi);
    if (span < minSpan) return [Math.max(0, hi - minSpan), hi];
    return [lo, hi];
  }

  return [0, Math.max(vRef, 1e-6)];
}

function sampleFlowParticleSpeedAt(
  layers: FlowParticleLayerVel[],
  layer: number,
  M: number,
  half: number,
  px: number,
  py: number,
  pz: number,
): number {
  const vel = layers[layer];
  if (!vel) return 0;
  const [vx, vy, vz] = sampleVelGridAt(vel.fx, vel.fy, vel.fz, M, half, px, py, pz);
  return Math.hypot(vx, vy, vz);
}

/** CPU mirror: distance to nearest axis-aligned grid plane. */
function ibfvGridLineDist(coord: number, spacing: number): number {
  const s = Math.max(spacing, 1e-4);
  const f = ((coord / s) % 1 + 1) % 1;
  return Math.min(f, 1 - f) * s;
}

/** Gridline injection G(p) in [0,1] on world-axis planes. */
export function ibfvBackgroundGridlines(
  px: number,
  py: number,
  pz: number,
  gridSpacing: number,
): number {
  const s = Math.max(gridSpacing, 1e-4);
  const w = s * 0.22;
  const smooth = (d: number) => Math.max(0, Math.min(1, 1 - d / w));
  const lx = smooth(ibfvGridLineDist(px, s));
  const ly = smooth(ibfvGridLineDist(py, s));
  const lz = smooth(ibfvGridLineDist(pz, s));
  return Math.max(lx, ly, lz);
}

/** Grid-point injection G(p) in [0,1] at lattice intersections only. */
export function ibfvBackgroundGridPoints(
  px: number,
  py: number,
  pz: number,
  gridSpacing: number,
): number {
  const s = Math.max(gridSpacing, 1e-4);
  const w = s * 0.22;
  const smooth = (d: number) => Math.max(0, Math.min(1, 1 - d / w));
  const lx = smooth(ibfvGridLineDist(px, s));
  const ly = smooth(ibfvGridLineDist(py, s));
  const lz = smooth(ibfvGridLineDist(pz, s));
  return lx * ly * lz;
}

export function ibfvBackgroundGrid(
  px: number,
  py: number,
  pz: number,
  gridSpacing: number,
  points: boolean,
): number {
  return points
    ? ibfvBackgroundGridPoints(px, py, pz, gridSpacing)
    : ibfvBackgroundGridlines(px, py, pz, gridSpacing);
}

/** Dye channels per voxel: density + advected age. */
export const FLOW_DYE_CHANNELS = 2;
export const FLOW_DYE_TOTAL = 0;
export const FLOW_DYE_AGE = 1;

function dyeVoxelIndex(ix: number, iy: number, iz: number, M: number): number {
  return (ix + iy * M + iz * M * M) * FLOW_DYE_CHANNELS;
}

/** Seed dye buffer with grid injection mask (one layer block per layer). */
export function seedFlowDyeGridlines(
  M: number,
  half: number,
  gridSpacing: number,
  layers: number,
  points = false,
): Float32Array {
  const volN = M * M * M;
  const buf = new Float32Array(volN * layers * FLOW_DYE_CHANNELS);
  const cell = (2 * half) / M;
  for (let layer = 0; layer < layers; layer++) {
    const off = layer * volN * FLOW_DYE_CHANNELS;
    for (let iz = 0; iz < M; iz++) {
      for (let iy = 0; iy < M; iy++) {
        for (let ix = 0; ix < M; ix++) {
          const px = (ix + 0.5) * cell - half;
          const py = (iy + 0.5) * cell - half;
          const pz = (iz + 0.5) * cell - half;
          const g = ibfvBackgroundGrid(px, py, pz, gridSpacing, points);
          const o = off + dyeVoxelIndex(ix, iy, iz, M);
          buf[o + FLOW_DYE_TOTAL] = g;
          buf[o + FLOW_DYE_AGE] = 0;
        }
      }
    }
  }
  return buf;
}

export function ibfvClampVelocity(
  vx: number,
  vy: number,
  vz: number,
  vMax: number,
): [number, number, number] {
  const speed = Math.hypot(vx, vy, vz);
  if (speed <= vMax || vMax <= 1e-8) return [vx, vy, vz];
  const s = vMax / speed;
  return [vx * s, vy * s, vz * s];
}

function gridToWorld(ix: number, iy: number, iz: number, M: number, half: number): [number, number, number] {
  const s = (2 * half) / M;
  return [(ix + 0.5) * s - half, (iy + 0.5) * s - half, (iz + 0.5) * s - half];
}

function sampleDyeChannel(
  dye: Float32Array,
  px: number,
  py: number,
  pz: number,
  M: number,
  half: number,
  ch: number,
): number {
  if (Math.abs(px) > half || Math.abs(py) > half || Math.abs(pz) > half) return 0;
  const f = [(px + half) / (2 * half) * M - 0.5, (py + half) / (2 * half) * M - 0.5, (pz + half) / (2 * half) * M - 0.5];
  const x0 = Math.floor(f[0]!);
  const y0 = Math.floor(f[1]!);
  const z0 = Math.floor(f[2]!);
  const tx = Math.max(0, Math.min(1, f[0]! - x0));
  const ty = Math.max(0, Math.min(1, f[1]! - y0));
  const tz = Math.max(0, Math.min(1, f[2]! - z0));
  const mi = M - 1;
  const idx = (x: number, y: number, z: number) => dyeVoxelIndex(
    Math.max(0, Math.min(mi, x)),
    Math.max(0, Math.min(mi, y)),
    Math.max(0, Math.min(mi, z)),
    M,
  ) + ch;
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const c000 = dye[idx(x0, y0, z0)]!;
  const c100 = dye[idx(x0 + 1, y0, z0)]!;
  const c010 = dye[idx(x0, y0 + 1, z0)]!;
  const c110 = dye[idx(x0 + 1, y0 + 1, z0)]!;
  const c001 = dye[idx(x0, y0, z0 + 1)]!;
  const c101 = dye[idx(x0 + 1, y0, z0 + 1)]!;
  const c011 = dye[idx(x0, y0 + 1, z0 + 1)]!;
  const c111 = dye[idx(x0 + 1, y0 + 1, z0 + 1)]!;
  return lerp(
    lerp(lerp(c000, c100, tx), lerp(c010, c110, tx), ty),
    lerp(lerp(c001, c101, tx), lerp(c011, c111, tx), ty),
    tz,
  );
}

export interface IbfvAdvectParams {
  alpha: number;
  gridSpacing: number;
  dt: number;
  vMax: number;
  frameIdx: number;
  half: number;
  gridPoints?: boolean;
}

/** One IBFV advection step: total dye + advected age (CPU reference). */
export function ibfvAdvectStep(
  dyeIn: Float32Array,
  dyeOut: Float32Array,
  sampleVel: (x: number, y: number, z: number) => [number, number, number],
  M: number,
  params: IbfvAdvectParams,
): void {
  const { alpha, gridSpacing, dt, vMax, half, gridPoints = false } = params;
  for (let iz = 0; iz < M; iz++) {
    for (let iy = 0; iy < M; iy++) {
      for (let ix = 0; ix < M; ix++) {
        const [px, py, pz] = gridToWorld(ix, iy, iz, M, half);
        const [vx, vy, vz] = sampleVel(px, py, pz);
        const speed = Math.hypot(vx, vy, vz);
        const [cx, cy, cz] = ibfvClampVelocity(vx, vy, vz, vMax);
        const pPrevX = px - cx * dt;
        const pPrevY = py - cy * dt;
        const pPrevZ = pz - cz * dt;
        const o = dyeVoxelIndex(ix, iy, iz, M);
        if (speed <= 1e-5) {
          dyeOut[o + FLOW_DYE_TOTAL] = 0;
          dyeOut[o + FLOW_DYE_AGE] = 0;
          continue;
        }
        const totalPrev = sampleDyeChannel(dyeIn, pPrevX, pPrevY, pPrevZ, M, half, FLOW_DYE_TOTAL);
        const agePrev = sampleDyeChannel(dyeIn, pPrevX, pPrevY, pPrevZ, M, half, FLOW_DYE_AGE) + dt;
        let G = 0;
        if (alpha > 1e-6) {
          G = ibfvBackgroundGrid(pPrevX, pPrevY, pPrevZ, gridSpacing, gridPoints);
        }
        const totalNew = (1 - alpha) * totalPrev + alpha * G;
        dyeOut[o + FLOW_DYE_TOTAL] = totalNew;
        dyeOut[o + FLOW_DYE_AGE] =
          totalNew > 1e-6 ? ((1 - alpha) * totalPrev * agePrev) / totalNew : 0;
      }
    }
  }
}

/** Floats per particle: x, y, z, age, speed. */
export const FLOW_PARTICLE_STRIDE = 5;

/** Floats per trail history slot: x, y, z, age, speed. */
export const FLOW_TRAIL_SLOT_STRIDE = 5;

export const MAX_FLOW_TRAIL_STEPS = 32;
export const DEFAULT_FLOW_TRAIL_STEPS = 32;

export function flowTrailBaseIndex(particleIndex: number): number {
  return particleIndex * MAX_FLOW_TRAIL_STEPS * FLOW_TRAIL_SLOT_STRIDE;
}

function chebIndexWorld(xi: number, M: number): number {
  const x = Math.max(-1, Math.min(1, xi));
  return (M / Math.PI) * Math.acos(x) - 0.5;
}

/** Trilinear sample of fitted velocity on the Chebyshev grid. */
export function sampleVelGridAt(
  fx: Float32Array,
  fy: Float32Array,
  fz: Float32Array,
  M: number,
  half: number,
  px: number,
  py: number,
  pz: number,
): [number, number, number] {
  const xi = Math.max(-1, Math.min(1, px / half));
  const yi = Math.max(-1, Math.min(1, py / half));
  const zi = Math.max(-1, Math.min(1, pz / half));
  const fx_ = chebIndexWorld(xi, M);
  const fy_ = chebIndexWorld(yi, M);
  const fz_ = chebIndexWorld(zi, M);
  const x0 = Math.floor(fx_);
  const y0 = Math.floor(fy_);
  const z0 = Math.floor(fz_);
  const tx = Math.max(0, Math.min(1, fx_ - x0));
  const ty = Math.max(0, Math.min(1, fy_ - y0));
  const tz = Math.max(0, Math.min(1, fz_ - z0));
  const mi = M - 1;
  const idx = (x: number, y: number, z: number) =>
    Math.max(0, Math.min(mi, x)) + Math.max(0, Math.min(mi, y)) * M + Math.max(0, Math.min(mi, z)) * M * M;
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const i000 = idx(x0, y0, z0);
  const i100 = idx(x0 + 1, y0, z0);
  const i010 = idx(x0, y0 + 1, z0);
  const i110 = idx(x0 + 1, y0 + 1, z0);
  const i001 = idx(x0, y0, z0 + 1);
  const i101 = idx(x0 + 1, y0, z0 + 1);
  const i011 = idx(x0, y0 + 1, z0 + 1);
  const i111 = idx(x0 + 1, y0 + 1, z0 + 1);
  const sample = (field: Float32Array) => lerp(
    lerp(lerp(field[i000]!, field[i100]!, tx), lerp(field[i010]!, field[i110]!, tx), ty),
    lerp(lerp(field[i001]!, field[i101]!, tx), lerp(field[i011]!, field[i111]!, tx), ty),
    tz,
  );
  return [sample(fx), sample(fy), sample(fz)];
}

function hash01(a: number, b: number, c: number): number {
  let x = Math.sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453;
  return x - Math.floor(x);
}

/** Seed `perLayer` particles for each flow layer (`perLayer × layers` total). */
export function seedFlowParticles(
  perLayer: number,
  layers: number,
  half: number,
  gridSpacing: number,
  points: boolean,
): { posAge: Float32Array; layerIds: Uint32Array } {
  const layerCount = Math.max(1, layers | 0);
  const quota = Math.max(1, perLayer | 0);
  const count = quota * layerCount;
  const posAge = new Float32Array(count * FLOW_PARTICLE_STRIDE);
  const layerIds = new Uint32Array(count);
  let n = 0;
  for (let layer = 0; layer < layerCount && n < count; layer++) {
    const target = quota;
    let placed = 0;
    let tries = 0;
    while (placed < target && n < count && tries < target * 40) {
      tries++;
      const px = (hash01(n, tries, layer) * 2 - 1) * half;
      const py = (hash01(n + 1, tries, layer) * 2 - 1) * half;
      const pz = (hash01(n + 2, tries, layer) * 2 - 1) * half;
      const g = ibfvBackgroundGrid(px, py, pz, gridSpacing, points);
      if (g < 0.15) continue;
      const o = n * FLOW_PARTICLE_STRIDE;
      posAge[o] = px;
      posAge[o + 1] = py;
      posAge[o + 2] = pz;
      posAge[o + 3] = 0;
      posAge[o + 4] = 0;
      layerIds[n] = layer;
      n++;
      placed++;
    }
    while (placed < target && n < count) {
      const o = n * FLOW_PARTICLE_STRIDE;
      posAge[o] = (hash01(n, 0, layer) * 2 - 1) * half;
      posAge[o + 1] = (hash01(n, 1, layer) * 2 - 1) * half;
      posAge[o + 2] = (hash01(n, 2, layer) * 2 - 1) * half;
      posAge[o + 3] = 0;
      posAge[o + 4] = 0;
      layerIds[n] = layer;
      n++;
      placed++;
    }
  }
  return { posAge, layerIds };
}

export interface FlowParticleLayerVel {
  fx: Float32Array;
  fy: Float32Array;
  fz: Float32Array;
}

export interface FlowParticleAdvectParams {
  dt: number;
  vMax: number;
  half: number;
  alpha: number;
  gridSpacing: number;
  gridPoints: boolean;
  ageMax: number;
  frameIdx: number;
}

export const FLOW_PARTICLE_DENSITY_GRID = 8;

export function flowParticleCellIndex(
  x: number,
  y: number,
  z: number,
  half: number,
  res: number,
): number {
  const h = Math.max(half, 1e-6);
  const fx = Math.min(res - 1, Math.max(0, Math.floor(((x / h) * 0.5 + 0.5) * res)));
  const fy = Math.min(res - 1, Math.max(0, Math.floor(((y / h) * 0.5 + 0.5) * res)));
  const fz = Math.min(res - 1, Math.max(0, Math.floor(((z / h) * 0.5 + 0.5) * res)));
  return fx + fy * res + fz * res * res;
}

/** Coarse occupancy counts for particle redistribution. */
export function buildFlowParticleDensityGrid(
  posAge: Float32Array,
  count: number,
  half: number,
  res = FLOW_PARTICLE_DENSITY_GRID,
): Uint16Array {
  const cells = new Uint16Array(res * res * res);
  for (let i = 0; i < count; i++) {
    const o = i * FLOW_PARTICLE_STRIDE;
    const ci = flowParticleCellIndex(posAge[o]!, posAge[o + 1]!, posAge[o + 2]!, half, res);
    if (cells[ci]! < 65535) cells[ci] = (cells[ci]! + 1) as number;
  }
  return cells;
}

/** Per-flow-layer occupancy (spawn/redistribute stay layer-local). */
export function buildFlowParticleDensityGrids(
  posAge: Float32Array,
  layerIds: Uint32Array,
  count: number,
  half: number,
  layerCount: number,
  res = FLOW_PARTICLE_DENSITY_GRID,
): Uint16Array[] {
  const grids: Uint16Array[] = [];
  for (let L = 0; L < layerCount; L++) {
    grids.push(new Uint16Array(res * res * res));
  }
  for (let i = 0; i < count; i++) {
    const layer = layerIds[i]!;
    if (layer < 0 || layer >= layerCount) continue;
    const o = i * FLOW_PARTICLE_STRIDE;
    const ci = flowParticleCellIndex(posAge[o]!, posAge[o + 1]!, posAge[o + 2]!, half, res);
    const grid = grids[layer]!;
    if (grid[ci]! < 65535) grid[ci] = (grid[ci]! + 1) as number;
  }
  return grids;
}

function flowParticleDensityForLayer(
  density: Uint16Array | Uint16Array[] | null | undefined,
  layer: number,
): Uint16Array | null {
  if (!density) return null;
  if (Array.isArray(density)) return density[layer] ?? null;
  return density;
}

function randomGridPointInCell(
  cell: number,
  res: number,
  half: number,
  seed: number,
  layer: number,
  frameIdx: number,
  gridSpacing: number,
  gridPoints: boolean,
): [number, number, number] | null {
  const fz = Math.floor(cell / (res * res));
  const rem = cell % (res * res);
  const fy = Math.floor(rem / res);
  const fx = rem % res;
  const cellSize = (2 * half) / res;
  for (let t = 0; t < 20; t++) {
    const px = -half + (fx + hash01(seed, t, frameIdx)) * cellSize;
    const py = -half + (fy + hash01(seed + 3, t, frameIdx + 1)) * cellSize;
    const pz = -half + (fz + hash01(seed + 7, t, frameIdx + 2)) * cellSize;
    const g = ibfvBackgroundGrid(px, py, pz, gridSpacing, gridPoints);
    if (g >= 0.12) return [px, py, pz];
  }
  return null;
}

/** Pick a spawn point in an under-populated cell (inverse local density). */
export function pickLowDensitySpawn(
  density: Uint16Array,
  res: number,
  half: number,
  seed: number,
  layer: number,
  frameIdx: number,
  gridSpacing: number,
  gridPoints: boolean,
): [number, number, number] | null {
  let minCount = Infinity;
  for (let c = 0; c < density.length; c++) {
    if (density[c]! < minCount) minCount = density[c]!;
  }
  const target = minCount;
  for (let attempt = 0; attempt < 48; attempt++) {
    const cell = (hash01(seed, attempt, frameIdx + layer) * density.length) | 0;
    if (density[cell]! > target + 1) continue;
    const pt = randomGridPointInCell(cell, res, half, seed + attempt, layer, frameIdx, gridSpacing, gridPoints);
    if (pt) return pt;
  }
  for (let c = 0; c < density.length; c++) {
    if (density[c]! > target + 1) continue;
    const pt = randomGridPointInCell(c, res, half, seed + c, layer, frameIdx, gridSpacing, gridPoints);
    if (pt) return pt;
  }
  return null;
}

function applyParticleSpawn(
  posAge: Float32Array,
  i: number,
  px: number,
  py: number,
  pz: number,
  trailHist: Float32Array | null | undefined,
  trailSteps: number,
  speed = 0,
): void {
  const o = i * FLOW_PARTICLE_STRIDE;
  posAge[o] = px;
  posAge[o + 1] = py;
  posAge[o + 2] = pz;
  posAge[o + 3] = 0;
  posAge[o + 4] = speed;
  resetFlowTrailHistSlot(trailHist, trailSteps, i, px, py, pz, 0, speed);
}

/** Move particles out of overcrowded cells into sparse regions. */
export function redistributeOvercrowdedFlowParticles(
  posAge: Float32Array,
  layerIds: Uint32Array,
  count: number,
  half: number,
  gridSpacing: number,
  gridPoints: boolean,
  frameIdx: number,
  trailHist: Float32Array | null,
  trailSteps: number,
  res = FLOW_PARTICLE_DENSITY_GRID,
  layers: FlowParticleLayerVel[] | null = null,
  gridM = 0,
  densityGrids: Uint16Array | Uint16Array[] | null = null,
): void {
  const density = densityGrids ?? buildFlowParticleDensityGrid(posAge, count, half, res);
  const cellCount = res * res * res;
  const layerCount = Array.isArray(density) ? density.length : 1;
  const mean = (count / Math.max(layerCount, 1)) / Math.max(cellCount, 1);
  const threshold = Math.max(4, Math.ceil(mean * 2.5));
  for (let i = 0; i < count; i++) {
    const o = i * FLOW_PARTICLE_STRIDE;
    const ci = flowParticleCellIndex(posAge[o]!, posAge[o + 1]!, posAge[o + 2]!, half, res);
    const layer = layerIds[i]!;
    const layerDensity = flowParticleDensityForLayer(density, layer);
    const grid = layerDensity ?? (!Array.isArray(density) ? density : null);
    if (!grid || grid[ci]! <= threshold) continue;
    const picked = pickLowDensitySpawn(
      grid, res, half, i, layer, frameIdx, gridSpacing, gridPoints,
    );
    if (!picked) continue;
    const speed = layers?.length && gridM > 0
      ? sampleFlowParticleSpeedAt(layers, layer, gridM, half, picked[0], picked[1], picked[2])
      : 0;
    applyParticleSpawn(posAge, i, picked[0], picked[1], picked[2], trailHist, trailSteps, speed);
    if (grid[ci]! > 0) grid[ci] = (grid[ci]! - 1) as number;
    const ni = flowParticleCellIndex(picked[0], picked[1], picked[2], half, res);
    if (grid[ni]! < 65535) grid[ni] = (grid[ni]! + 1) as number;
  }
}

function respawnParticle(
  posAge: Float32Array,
  i: number,
  layer: number,
  half: number,
  gridSpacing: number,
  gridPoints: boolean,
  frameIdx: number,
  trailHist?: Float32Array | null,
  trailSteps?: number,
  density?: Uint16Array | null,
  densityRes = FLOW_PARTICLE_DENSITY_GRID,
  layers: FlowParticleLayerVel[] | null = null,
  gridM = 0,
): void {
  const spawnAt = (px: number, py: number, pz: number) => {
    const speed = layers?.length && gridM > 0
      ? sampleFlowParticleSpeedAt(layers, layer, gridM, half, px, py, pz)
      : 0;
    applyParticleSpawn(posAge, i, px, py, pz, trailHist, trailSteps ?? 0, speed);
  };
  if (density) {
    const picked = pickLowDensitySpawn(
      density, densityRes, half, i, layer, frameIdx, gridSpacing, gridPoints,
    );
    if (picked) {
      spawnAt(picked[0], picked[1], picked[2]);
      return;
    }
  }
  for (let t = 0; t < 24; t++) {
    const px = (hash01(i, t, frameIdx + layer) * 2 - 1) * half;
    const py = (hash01(i + 7, t, frameIdx) * 2 - 1) * half;
    const pz = (hash01(i + 13, t, frameIdx + 1) * 2 - 1) * half;
    const g = ibfvBackgroundGrid(px, py, pz, gridSpacing, gridPoints);
    if (g < 0.12 && t < 23) continue;
    spawnAt(px, py, pz);
    return;
  }
  const o = i * FLOW_PARTICLE_STRIDE;
  spawnAt(posAge[o]!, posAge[o + 1]!, posAge[o + 2]!);
}

/** Fill all trail slots for one particle (e.g. on respawn). */
export function resetFlowTrailHistSlot(
  trailHist: Float32Array | null | undefined,
  trailSteps: number,
  i: number,
  px: number,
  py: number,
  pz: number,
  age: number,
  speed = 0,
): void {
  if (!trailHist || trailSteps < 2) return;
  const ho = flowTrailBaseIndex(i);
  for (let j = 0; j < trailSteps; j++) {
    const o = ho + j * FLOW_TRAIL_SLOT_STRIDE;
    trailHist[o] = px;
    trailHist[o + 1] = py;
    trailHist[o + 2] = pz;
    trailHist[o + 3] = age;
    trailHist[o + 4] = speed;
  }
}

/** Initialize trail ring from seeded particle positions. */
export function seedFlowTrailHist(
  posAge: Float32Array,
  trailHist: Float32Array,
  trailSteps: number,
  count: number,
): void {
  if (trailSteps < 2) return;
  for (let i = 0; i < count; i++) {
    const o = i * FLOW_PARTICLE_STRIDE;
    resetFlowTrailHistSlot(
      trailHist,
      trailSteps,
      i,
      posAge[o]!,
      posAge[o + 1]!,
      posAge[o + 2]!,
      posAge[o + 3]!,
    );
  }
}

/** Update slot 0 only (current head) without shifting history. */
export function updateFlowTrailHead(
  posAge: Float32Array,
  trailHist: Float32Array,
  count: number,
): void {
  for (let i = 0; i < count; i++) {
    const po = i * FLOW_PARTICLE_STRIDE;
    const ho = flowTrailBaseIndex(i);
    trailHist[ho] = posAge[po]!;
    trailHist[ho + 1] = posAge[po + 1]!;
    trailHist[ho + 2] = posAge[po + 2]!;
    trailHist[ho + 3] = posAge[po + 3]!;
    trailHist[ho + 4] = posAge[po + 4]!;
  }
}

/** Shift trail history and insert the current particle state at slot 0. */
export function pushFlowTrailHist(
  posAge: Float32Array,
  trailHist: Float32Array,
  trailSteps: number,
  count: number,
): void {
  if (trailSteps < 2) return;
  for (let i = 0; i < count; i++) {
    const po = i * FLOW_PARTICLE_STRIDE;
    const ho = flowTrailBaseIndex(i);
    for (let j = trailSteps - 1; j >= 1; j--) {
      const dst = ho + j * FLOW_TRAIL_SLOT_STRIDE;
      const src = ho + (j - 1) * FLOW_TRAIL_SLOT_STRIDE;
      trailHist[dst] = trailHist[src]!;
      trailHist[dst + 1] = trailHist[src + 1]!;
      trailHist[dst + 2] = trailHist[src + 2]!;
      trailHist[dst + 3] = trailHist[src + 3]!;
      trailHist[dst + 4] = trailHist[src + 4]!;
    }
    trailHist[ho] = posAge[po]!;
    trailHist[ho + 1] = posAge[po + 1]!;
    trailHist[ho + 2] = posAge[po + 2]!;
    trailHist[ho + 3] = posAge[po + 3]!;
    trailHist[ho + 4] = posAge[po + 4]!;
  }
}

/** One CPU advection step for all particles (reference + runtime). */
export function advectFlowParticles(
  posAge: Float32Array,
  layerIds: Uint32Array,
  layers: FlowParticleLayerVel[],
  M: number,
  params: FlowParticleAdvectParams,
  trailHist: Float32Array | null = null,
  trailSteps = 0,
  density: Uint16Array | Uint16Array[] | null = null,
  densityRes = FLOW_PARTICLE_DENSITY_GRID,
): void {
  const { dt, vMax, half, alpha, gridSpacing, gridPoints, ageMax, frameIdx } = params;
  const n = layerIds.length;
  for (let i = 0; i < n; i++) {
    const layer = layerIds[i]!;
    const vel = layers[layer];
    if (!vel) continue;
    const layerDensity = flowParticleDensityForLayer(density, layer);
    const o = i * FLOW_PARTICLE_STRIDE;
    let px = posAge[o]!;
    let py = posAge[o + 1]!;
    let pz = posAge[o + 2]!;
    let age = posAge[o + 3]!;
    const [vx, vy, vz] = sampleVelGridAt(vel.fx, vel.fy, vel.fz, M, half, px, py, pz);
    const speed = Math.hypot(vx, vy, vz);
    const [cx, cy, cz] = ibfvClampVelocity(vx, vy, vz, vMax);
    const expired = age > ageMax;
    const stuck = speed <= 1e-5 && age > ageMax * 0.5;
    if (expired || stuck) {
      respawnParticle(
        posAge, i, layer, half, gridSpacing, gridPoints, frameIdx,
        trailHist, trailSteps, layerDensity, densityRes, layers, M,
      );
      continue;
    }
    px += cx * dt;
    py += cy * dt;
    pz += cz * dt;
    age += dt;
    if (alpha > 1e-6 && hash01(i, frameIdx, 919) < alpha) {
      respawnParticle(
        posAge, i, layer, half, gridSpacing, gridPoints, frameIdx,
        trailHist, trailSteps, layerDensity, densityRes, layers, M,
      );
      continue;
    }
    posAge[o] = px;
    posAge[o + 1] = py;
    posAge[o + 2] = pz;
    posAge[o + 3] = age;
    posAge[o + 4] = speed;
  }
}

/** Min/max head speed among live particles (cheap color-range probe). */
export function flowHeadSpeedMinMax(
  posAge: Float32Array,
  count: number,
): [min: number, max: number] | null {
  let vmin = Infinity;
  let vmax = 0;
  let samples = 0;
  for (let i = 0; i < count; i++) {
    const s = posAge[i * FLOW_PARTICLE_STRIDE + 4]!;
    if (s <= 1e-8) continue;
    samples++;
    if (s < vmin) vmin = s;
    if (s > vmax) vmax = s;
  }
  if (samples < 2 || !Number.isFinite(vmin)) return null;
  if (vmax <= vmin) {
    const s = Math.max(vmax, 1e-8);
    return [Math.max(0, s * 0.55), s * 1.05];
  }
  return [vmin, vmax];
}

/** Particle color range from head speeds + cached field percentiles (no trail scan). */
export function resolveFlowParticleColorRangeFast(
  posAge: Float32Array,
  count: number,
  vRef: number,
  fieldRange: [number, number] | null = null,
): [min: number, max: number] {
  const head = flowHeadSpeedMinMax(posAge, count);
  if (head && head[1] > 1e-8) {
    let [lo, hi] = head;
    const span = hi - lo;
    const minSpan = flowColorSpeedSpan(hi);
    if (span < minSpan * 0.35 && fieldRange && fieldRange[1] > fieldRange[0] + 1e-8) {
      lo = fieldRange[0];
      hi = fieldRange[1];
    } else if (span < minSpan) {
      lo = Math.max(0, hi - minSpan);
    }
    return [lo, hi];
  }
  if (fieldRange && fieldRange[1] > 1e-8) return fieldRange;
  return [0, Math.max(vRef, 1e-6)];
}

/** Sort particle indices back-to-front for alpha compositing. */
export function sortFlowParticlesByDepth(
  posAge: Float32Array,
  sortOrder: Uint32Array,
  ro: [number, number, number],
  viewDir: [number, number, number],
  depthKeys?: Float32Array,
): void {
  const n = sortOrder.length;
  const v0 = viewDir[0];
  const v1 = viewDir[1];
  const v2 = viewDir[2];
  const r0 = ro[0];
  const r1 = ro[1];
  const r2 = ro[2];
  for (let i = 0; i < n; i++) {
    sortOrder[i] = i;
    if (depthKeys) {
      const o = i * FLOW_PARTICLE_STRIDE;
      depthKeys[i] =
        (posAge[o]! - r0) * v0 +
        (posAge[o + 1]! - r1) * v1 +
        (posAge[o + 2]! - r2) * v2;
    }
  }
  if (depthKeys) {
    sortOrder.sort((a, b) => depthKeys[b]! - depthKeys[a]!);
    return;
  }
  sortOrder.sort((a, b) => {
    const oa = a * FLOW_PARTICLE_STRIDE;
    const ob = b * FLOW_PARTICLE_STRIDE;
    const da =
      (posAge[oa]! - r0) * v0 +
      (posAge[oa + 1]! - r1) * v1 +
      (posAge[oa + 2]! - r2) * v2;
    const db =
      (posAge[ob]! - r0) * v0 +
      (posAge[ob + 1]! - r1) * v1 +
      (posAge[ob + 2]! - r2) * v2;
    return db - da;
  });
}

export function fitVectorField(
  compiled: CompiledVectorExpr,
  vectorFn: (x: number, y: number, z: number) => [number, number, number],
  half: number,
  deg: number,
  opts: { skipL2?: boolean } = {},
): VectorFitResult {
  if (compiled.kind === "curl" && compiled.bindTuple) {
    const tupleFn = compiled.bindTuple();
    const fitX = fitChebyshev3D((x, y, z) => tupleFn(x, y, z)[0]!, half, deg, {
      skipL2: opts.skipL2 ?? true,
      skipMono: true,
    });
    const fitY = fitChebyshev3D((x, y, z) => tupleFn(x, y, z)[1]!, half, deg, {
      skipL2: opts.skipL2 ?? true,
      skipMono: true,
    });
    const fitZ = fitChebyshev3D((x, y, z) => tupleFn(x, y, z)[2]!, half, deg, {
      skipL2: opts.skipL2 ?? true,
      skipMono: true,
    });
    const curl = idctChebCurl3D(fitX.cheb, fitY.cheb, fitZ.cheb, deg, deg + 1);
    const scale = 1 / half;
    const fx = new Float32Array(curl.fx.length);
    const fy = new Float32Array(curl.fy.length);
    const fz = new Float32Array(curl.fz.length);
    for (let i = 0; i < fx.length; i++) {
      fx[i] = curl.fx[i]! * scale;
      fy[i] = curl.fy[i]! * scale;
      fz[i] = curl.fz[i]! * scale;
    }
    return {
      fx,
      fy,
      fz,
      fitRel: fitX.fitRelL2,
      M: curl.M,
      source: "curl",
    };
  }

  if (compiled.kind === "gradient" && compiled.bindScalar) {
    const scalarFn = compiled.bindScalar();
    const fit = fitChebyshev3D(scalarFn, half, deg, {
      skipL2: opts.skipL2 ?? true,
      skipMono: true,
    });
    const grad = idctChebGrad3D(fit.cheb, deg, deg + 1);
    // World-coordinate chain rule: d/dx_world = (1/half) * d/dxi
    const invHalf = 1 / half;
    const scale = invHalf;
    const fx = new Float32Array(grad.gx.length);
    const fy = new Float32Array(grad.gy.length);
    const fz = new Float32Array(grad.gz.length);
    for (let i = 0; i < fx.length; i++) {
      fx[i] = grad.gx[i]! * scale;
      fy[i] = grad.gy[i]! * scale;
      fz[i] = grad.gz[i]! * scale;
    }
    return {
      fx,
      fy,
      fz,
      cheb: fit.cheb,
      fitRel: fit.fitRelL2,
      M: grad.M,
      source: "gradient",
    };
  }

  const fitX = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[0]!, half, deg, {
    skipL2: opts.skipL2 ?? true,
    skipMono: true,
  });
  const fitY = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[1]!, half, deg, {
    skipL2: opts.skipL2 ?? true,
    skipMono: true,
  });
  const fitZ = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[2]!, half, deg, {
    skipL2: opts.skipL2 ?? true,
    skipMono: true,
  });
  const M = deg + 1;
  const ix = idctCheb3D(fitX.cheb, deg, M);
  const iy = idctCheb3D(fitY.cheb, deg, M);
  const iz = idctCheb3D(fitZ.cheb, deg, M);
  return {
    fx: ix.dens,
    fy: iy.dens,
    fz: iz.dens,
    fitRel: fitX.fitRelL2,
    M: ix.M,
    source: "tuple" as VectorFieldKind,
  };
}
