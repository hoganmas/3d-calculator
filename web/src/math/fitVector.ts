/**
 * Vector field expressions: tuple / gradient LaTeX → Chebyshev velocity volumes.
 */

import { compile, ComputeEngine } from "@cortex-js/compute-engine";
import { fitChebyshev3D } from "./fit.js";
import { idctCheb3D, idctChebGrad3D } from "./idct.js";
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
  "sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs", "grad", "nabla",
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

/** Normalize grad/nabla LaTeX for CE parsing. */
export function normalizeVectorLatex(latex: string): string {
  let s = String(latex ?? "").trim();
  s = s.replace(/\\operatorname\s*\{\s*grad\s*\}/gi, "\\grad");
  s = s.replace(/\\grad\s*\{/g, "\\operatorname{grad}{");
  s = s.replace(/\\grad\s*\(/g, "\\operatorname{grad}(");
  s = s.replace(/\\nabla\s+/g, "\\operatorname{grad} ");
  return s;
}

function extractTriple(json: unknown): string[] | null {
  if (!Array.isArray(json)) return null;
  const head = json[0];
  if (head === "List" || head === "Tuple" || head === "Sequence") {
    const parts = json.slice(1).filter((p) => p != null);
    if (parts.length === 3) {
      return parts.map((p) => ce.box(p as never).latex).filter(Boolean) as string[];
    }
  }
  if (head === "Delimiter") {
    const inner = json[2];
    return extractTriple(inner);
  }
  if (head === "Matrix" || head === "MatrixExpression") {
    const rows = json.slice(1);
    const flat: string[] = [];
    for (const row of rows) {
      if (Array.isArray(row)) flat.push(...(extractTriple(row) ?? []));
      else flat.push(ce.box(row as never).latex);
    }
    if (flat.length === 3) return flat;
  }
  return null;
}

function looksLikeGrad(src: string, json: unknown): string | null {
  const lower = src.toLowerCase();
  if (/\\grad|\\nabla|\\operatorname\s*\{\s*grad/.test(lower)) {
    const m = src.match(/\\(?:operatorname\s*\{\s*grad\s*\}|grad|nabla)\s*[\{\(]?\s*([\s\S]+?)\s*[\}\)]?\s*$/);
    if (m?.[1]) return m[1].trim();
  }
  if (Array.isArray(json)) {
    const head = String(json[0]).toLowerCase();
    if (head === "grad" || head === "gradient" || head === "nabla") {
      const arg = json[1];
      if (arg != null) return ce.box(arg as never).latex;
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

  const triple = extractTriple(j);
  if (triple?.length === 3) {
    return {
      kind: "tuple",
      label: "vector tuple",
      compileParts: triple,
    };
  }

  throw new Error(
    "Flow fields need a 3-component tuple like (Fx, Fy, Fz) or a gradient like \\grad f",
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

/** Gaussian dye blob centered in the box (Chebyshev-root grid). */
export function seedFlowDye(M: number, half: number): Float32Array {
  const dye = new Float32Array(M * M * M);
  const sigma = half * 0.22;
  const invSigma2 = 1 / (sigma * sigma);
  for (let ix = 0; ix < M; ix++) {
    const xi = Math.cos((Math.PI * (2 * ix + 1)) / (2 * M));
    const x = xi * half;
    for (let iy = 0; iy < M; iy++) {
      const yi = Math.cos((Math.PI * (2 * iy + 1)) / (2 * M));
      const y = yi * half;
      for (let iz = 0; iz < M; iz++) {
        const zi = Math.cos((Math.PI * (2 * iz + 1)) / (2 * M));
        const z = zi * half;
        const r2 = x * x + y * y + z * z;
        dye[ix + iy * M + iz * M * M] = Math.exp(-r2 * invSigma2);
      }
    }
  }
  return dye;
}

export function fitVectorField(
  compiled: CompiledVectorExpr,
  vectorFn: (x: number, y: number, z: number) => [number, number, number],
  half: number,
  deg: number,
  opts: { skipL2?: boolean } = {},
): VectorFitResult {
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
