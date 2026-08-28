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
