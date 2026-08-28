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

function cross3(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
): [number, number, number] {
  return [
    ay * bz - az * by,
    az * bx - ax * bz,
    ax * by - ay * bx,
  ];
}

function norm3(x: number, y: number, z: number): number {
  return Math.hypot(x, y, z);
}

/** Spatial phase coordinate: k·p for stream-aligned flow; azimuth in plane ⊥ (p×V) for rotation. */
export function flowSpatialPhase(
  vx: number,
  vy: number,
  vz: number,
  px: number,
  py: number,
  pz: number,
  half: number,
): number {
  const speed = Math.hypot(vx, vy, vz);
  if (speed < 1e-12) return 0;
  const kx = vx / speed;
  const ky = vy / speed;
  const kz = vz / speed;
  const along = kx * px + ky * py + kz * pz;
  const r = norm3(px, py, pz);
  const align = r > 1e-4 ? Math.abs(along / r) : 0;
  const spinMix = 1 - Math.min(1, align / 0.35);
  let azimuth = Math.atan2(py, px) * (half / Math.PI);
  const [ax, ay, az] = cross3(px, py, pz, vx, vy, vz);
  const axisLen = norm3(ax, ay, az);
  if (axisLen > 1e-4) {
    const axis = [ax / axisLen, ay / axisLen, az / axisLen] as const;
    let rx = 0; let ry = 0; let rz = 1;
    if (Math.abs(axis[0] * rx + axis[1] * ry + axis[2] * rz) > 0.9) {
      rx = 0; ry = 1; rz = 0;
    }
    if (Math.abs(axis[0] * rx + axis[1] * ry + axis[2] * rz) > 0.9) {
      rx = 1; ry = 0; rz = 0;
    }
    let [ux, uy, uz] = cross3(rx, ry, rz, axis[0], axis[1], axis[2]);
    const uNorm = norm3(ux, uy, uz);
    const uxN = ux / uNorm;
    const uyN = uy / uNorm;
    const uzN = uz / uNorm;
    const [vx2, vy2, vz2] = cross3(axis[0], axis[1], axis[2], uxN, uyN, uzN);
    const theta = Math.atan2(
      px * vx2 + py * vy2 + pz * vz2,
      px * uxN + py * uyN + pz * uzN,
    );
    azimuth = theta * (half / Math.PI);
  }
  return along * (1 - spinMix) + azimuth * spinMix;
}

/** Phase φ = spatial·stripeScale − t·timeScale with k ∥ V. */
export function flowPhaseAt(
  vx: number,
  vy: number,
  vz: number,
  px: number,
  py: number,
  pz: number,
  t: number,
  stripeScale: number,
  timeScale: number,
  half: number = 2.5,
): number {
  const spatial = flowSpatialPhase(vx, vy, vz, px, py, pz, half);
  return spatial * stripeScale - t * timeScale;
}

/** Soft sine band in 0..1. */
export function flowSoftBand(phi: number): number {
  return 0.5 + 0.5 * Math.sin(phi);
}

/** Fixed base opacity modulated by soft sine band; zero at stagnation. */
export function flowPhaseOpacity(
  speed: number,
  phi: number,
  opacity: number,
): number {
  if (speed < 1e-5) return 0;
  return opacity * (0.12 + 0.88 * flowSoftBand(phi));
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
