/** Density expression (MathLive/LaTeX via Compute Engine) + 3D Chebyshev fit → IDCT dens. */

import { compile, ComputeEngine } from "@cortex-js/compute-engine";
import { MAX_DEG } from "./limits.js";
import { extractTriple, normalizeCalcLatex, scalarFromUnaryOpJson, tripleFromUnaryOpJson } from "./calcOps.js";
import { idctCheb3D, idctChebDivergence3D, idctChebLaplacian3D } from "./idct.js";
import type {
  ChebFitResult,
  ClassifiedExpr,
  CompiledExpr,
  CompiledParam,
  FieldKind,
  PresetDef,
  ScalarFitResult,
  ChebFitTiming,
} from "../types/models.js";
import type { SymbolRegistry } from "../model/symbols.js";

type MathJsonArray = unknown[];
type CeRun = (scope: Record<string, unknown>) => unknown;

interface CeCompileResult {
  success: boolean;
  unsupported?: string[];
  run: CeRun | null;
  freeSymbols?: Iterable<unknown>;
}

function jsonArr(j: unknown): MathJsonArray {
  return Array.isArray(j) ? j : [];
}

const ce = new ComputeEngine();

/**
 * Spatial vars (not sliders). Cartesian + spherical + cylindrical:
 *   r, θ, φ  — physics spherical (θ from +z, φ = atan2(y,x))
 *   ρ        — cylindrical radius √(x²+y²)
 * LaTeX `\theta`/`\phi`/`\rho` bind as `theta`/`phi`/`rho`.
 */
const RESERVED_SYMBOLS = new Set([
  "x",
  "y",
  "z",
  "r",
  "theta",
  "phi",
  "rho",
]);

/** World / polar coords — required for a field to be graphed (fit + march). */
const SPATIAL_SYMBOLS = new Set(["x", "y", "z", "r", "theta", "phi", "rho"]);

/**
 * Built-in function names — never auto-promoted to slider parameters.
 * (CE's `compile(string)` treats these as identifiers; LaTeX must be `ce.parse`d first.)
 */
const KNOWN_FUNCTION_NAMES = new Set([
  "sin", "cos", "tan", "cot", "sec", "csc",
  "arcsin", "arccos", "arctan", "arccot", "arcsec", "arccsc",
  "asin", "acos", "atan", "atan2",
  "sinh", "cosh", "tanh", "coth",
  "asinh", "acosh", "atanh",
  "exp", "ln", "log", "log10", "log2", "lg",
  "sqrt", "cbrt", "abs", "sign", "floor", "ceil", "round",
  "max", "min", "hypot", "pow",
  "sinc", "erf", "gamma",
]);

/** Longest-first so `arccos` wins over `cos`. */
const LATEX_FN_REWRITE = [
  "arccos", "arcsin", "arctan", "arccot", "arcsec", "arccsc",
  "arsinh", "arcosh", "artanh", "arctanh", "arcsech", "arccsch",
  "sinh", "cosh", "tanh", "coth",
  "sin", "cos", "tan", "cot", "sec", "csc",
  "exp", "ln", "log", "max", "min", "sqrt", "abs",
];

/**
 * Turn bare typed names (`cos`, `sin x`) into LaTeX commands (`\cos`, `\sin x`)
 * so users need not type `\`. Skips names already introduced by `\`.
 * @param {string} latex
 */
function normalizeLatexFunctions(latex: string) {
  let s = String(latex ?? "");
  for (const name of LATEX_FN_REWRITE) {
    const re = new RegExp(`(^|[^\\\\A-Za-z])(${name})(?![A-Za-z])`, "gi");
    s = s.replace(re, (_, pre, n) => `${pre}\\${n.toLowerCase()}`);
  }
  return s;
}

/**
 * Compile LaTeX via parse→box. Never pass raw strings to `compile()` — that path
 * treats `cos` as a JS identifier (`_.cos`) instead of letter juxtaposition.
 * @param {string} latex
 */
function compileLatex(latex: string): CeCompileResult {
  const src = normalizeLatexFunctions(String(latex ?? "").trim());
  if (!src) {
    return { success: false, unsupported: ["empty"], run: null, freeSymbols: [] };
  }
  let box;
  try {
    box = ce.parse(src);
  } catch {
    return { success: false, unsupported: ["parse"], run: null, freeSymbols: [] };
  }
  return compile(box) as CeCompileResult;
}

/**
 * Drop reserved / known-function symbols from CE freeSymbols.
 * Also suppress `cos`-style letter runs that are only an incomplete function name.
 * @param {Iterable<unknown>} freeSymbols
 * @param {string} latex
 * @param {{ skipName?: string | null }} [opts]
 * @returns {{ freeParams: string[], usesSpace: boolean }}
 */
function collectFreeParams(
  freeSymbols: Iterable<unknown> | null | undefined,
  latex: string,
  opts: { skipName?: string | null } = {},
) {
  /** @type {string[]} */
  const ids = [];
  let usesSpace = false;
  for (const s of freeSymbols ?? []) {
    const id = String(s);
    if (SPATIAL_SYMBOLS.has(id)) usesSpace = true;
    if (RESERVED_SYMBOLS.has(id)) continue;
    if (opts.skipName && id === opts.skipName) continue;
    if (KNOWN_FUNCTION_NAMES.has(id.toLowerCase())) continue;
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id)) {
      throw new Error(`Invalid parameter “${id}” (use letters / digits)`);
    }
    ids.push(id);
  }

  // Bare typed name of a known fn (`cos` / `sin` / …) — not a slider param.
  // CE may parse these as letter products (and `sin` even eats `i` as √−1).
  const compact = String(latex ?? "").replace(/\s+/g, "");
  if (/^[A-Za-z]+$/.test(compact) && KNOWN_FUNCTION_NAMES.has(compact.toLowerCase())) {
    return { freeParams: [], usesSpace: false };
  }

  ids.sort();
  return { freeParams: ids, usesSpace };
}

/** @param {unknown} json MathJSON node */
function symbolId(json: unknown): string | null {
  if (typeof json === "string" && /^[A-Za-z][A-Za-z0-9_]*$/.test(json)) return json;
  if (Array.isArray(json) && json[0] === "Symbol" && typeof json[1] === "string") return json[1];
  return null;
}

function coerceNumber(v: unknown): number {
  if (typeof v === "number") return v;
  if (v && typeof v.valueOf === "function") return Number(v.valueOf());
  return Number(v);
}

/** World (x,y,z) → spherical / cylindrical auxiliaries for the expr scope. */
function polarFromCartesian(x: number, y: number, z: number) {
  const rho = Math.hypot(x, y);
  const r = Math.hypot(rho, z);
  const phi = Math.atan2(y, x);
  const theta = r > 1e-15 ? Math.acos(Math.min(1, Math.max(-1, z / r))) : 0;
  return { r, theta, phi, rho };
}

/**
 * MathJSON operator heads — not user function names.
 * (Incomplete on purpose; unknown heads with args are treated as calls.)
 */
const MATH_OPERATORS = new Set([
  "Add",
  "Subtract",
  "Multiply",
  "Divide",
  "Power",
  "Negate",
  "Root",
  "Sqrt",
  "Abs",
  "Exp",
  "Ln",
  "Log",
  "Sin",
  "Cos",
  "Tan",
  "ArcSin",
  "ArcCos",
  "ArcTan",
  "Sinh",
  "Cosh",
  "Tanh",
  "Max",
  "Min",
  "Delimiter",
  "List",
  "Tuple",
  "Equal",
  "Assign",
  "Colon",
  "Function",
  "Block",
  "Error",
]);

/**
 * LaTeX left of `=` looks like `f(...)` / `f\left(...\right)`.
 * Catches CE's `f(r)` → Multiply(f,r) misparse for single-arg defs.
 */
function latexLooksLikeFunctionDef(src: string) {
  const eq = String(src).search(/=/);
  if (eq < 0) return false;
  const left = String(src).slice(0, eq).trim();
  return /^[A-Za-z][A-Za-z0-9]*\s*(\\left\s*)?\(/.test(left);
}

function isUserFunctionCall(json: unknown) {
  if (!Array.isArray(json) || typeof json[0] !== "string") return false;
  const head = json[0];
  if (MATH_OPERATORS.has(head)) return false;
  if (!/^[A-Za-z][A-Za-z0-9]*$/.test(head)) return false;
  // f(x), f(x,y,z), …
  return json.length >= 2;
}

/** Parse `f(x,y)=` left-hand side into name and formal args. */
function parseFuncDefLhs(src: string, lhs: unknown) {
  if (isUserFunctionCall(lhs)) {
    const ja = jsonArr(lhs);
    const funcName = String(ja[0]);
    const funcArgs = ja
      .slice(1)
      .map((a) => symbolId(a))
      .filter((a): a is string => !!a);
    return { funcName, funcArgs };
  }

  const eq = String(src).search(/=/);
  if (eq < 0) throw new Error("Invalid function definition");
  const left = String(src).slice(0, eq).trim();
  const m = left.match(
    /^([A-Za-z][A-Za-z0-9]*)\s*(?:\\left\s*)?\(([\s\S]*?)\)(?:\\right\s*)?\s*$/,
  );
  if (!m) throw new Error("Invalid function definition");
  const funcName = m[1]!;
  const argSrc = m[2]!.trim();
  const funcArgs = argSrc
    ? argSrc
        .split(",")
        .map((s) => s.trim())
        .map((s) => symbolId(s) ?? (/^[A-Za-z][A-Za-z0-9_]*$/.test(s) ? s : null))
        .filter((a): a is string => !!a)
    : [];
  return { funcName, funcArgs };
}

function substituteFuncBody(body: unknown, formals: string[], actuals: unknown[]): unknown {
  const binding = new Map<string, unknown>();
  for (let i = 0; i < formals.length; i++) {
    binding.set(formals[i]!, actuals[i] ?? formals[i]);
  }

  function walk(node: unknown): unknown {
    const id = symbolId(node);
    if (id && binding.has(id)) return binding.get(id);

    if (Array.isArray(node)) {
      if (isUserFunctionCall(node)) {
        const ja = node as unknown[];
        return [ja[0], ...ja.slice(1).map(walk)];
      }
      return node.map((n, i) => (i === 0 && typeof n === "string" ? n : walk(n)));
    }
    return node;
  }

  return walk(body);
}

function isMultiplyFuncCall(json: unknown, registry: SymbolRegistry): { fn: string; args: unknown[] } | null {
  if (!Array.isArray(json) || json[0] !== "Multiply" || json.length < 3) return null;
  const fn = symbolId(json[1]);
  if (!fn || !registry.getFuncdef(fn)) return null;
  return { fn, args: json.slice(2) };
}

function expandJson(json: unknown, registry: SymbolRegistry): unknown {
  const mulCall = isMultiplyFuncCall(json, registry);
  if (mulCall) {
    const fd = registry.getFuncdef(mulCall.fn)!;
    const actuals = mulCall.args.map((a) => expandJson(a, registry));
    let body: unknown;
    try {
      body = ce.parse(normalizeLatexFunctions(fd.rhsLatex)).json;
    } catch {
      return json;
    }
    return expandJson(substituteFuncBody(body, fd.funcArgs ?? [], actuals), registry);
  }

  if (isUserFunctionCall(json)) {
    const ja = jsonArr(json);
    const fn = String(ja[0]);
    const fd = registry.getFuncdef(fn);
    if (fd?.funcArgs?.length) {
      const actuals = ja.slice(1).map((a) => expandJson(a, registry));
      let body: unknown;
      try {
        body = ce.parse(normalizeLatexFunctions(fd.rhsLatex)).json;
      } catch {
        return json;
      }
      return expandJson(substituteFuncBody(body, fd.funcArgs, actuals), registry);
    }
    return [ja[0], ...ja.slice(1).map((a) => expandJson(a, registry))];
  }

  const id = symbolId(json);
  if (id) {
    const alias = registry.getAlias(id);
    if (alias) {
      try {
        const body = ce.parse(normalizeLatexFunctions(alias.rhsLatex)).json;
        return expandJson(body, registry);
      } catch {
        return json;
      }
    }
    return json;
  }

  if (Array.isArray(json)) {
    return json.map((n, i) => (i === 0 && typeof n === "string" ? n : expandJson(n, registry)));
  }
  return json;
}

/** Expand alias names and user function calls using a declaration registry. */
export function expandDefinitions(raw: string, registry: SymbolRegistry): string {
  const src = normalizeLatexFunctions(String(raw ?? "").trim());
  if (!src) return src;

  let latex = src;
  for (let iter = 0; iter < 32; iter++) {
    let box;
    try {
      box = ce.parse(latex);
    } catch {
      return latex;
    }
    const j = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);
    const next = expandJson(j, registry);
    if (JSON.stringify(next) === JSON.stringify(j)) break;
    const nextBox = ce.box(next as never);
    latex = nextBox.latex || latex;
  }
  return latex;
}

/**
 * Classify input and rewrite to a numeric field for fitting.
 *
 * - parameter `a = E`      → named slider (RHS has no spatial deps)
 * - alias `T = E`          → named spatial expression (not a slider)
 * - funcdef `f(…) = E`     → reusable function (expanded at call sites)
 * - constraint `A = B`     → field A−B, isosurface at 0
 * - bare expression `E`    → field E, volume (Beer)
 *
 * @returns {{
 *   kind: "parameter" | "constraint" | "definition" | "bare",
 *   shade: "iso" | "volume" | "none",
 *   isoLevel: number,
 *   compileLatex: string,
 *   label: string,
 *   paramName?: string,
 * }}
 */
export function classifyExpr(raw: string): ClassifiedExpr {
  const src = normalizeLatexFunctions(String(raw ?? "").trim());
  if (!src) throw new Error("Empty expression");

  let box;
  try {
    box = ce.parse(src);
  } catch {
    return {
      kind: "bare",
      shade: "volume",
      isoLevel: 0,
      compileLatex: src,
      label: "expression",
    };
  }

  const j = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);
  const headRaw = Array.isArray(j) ? j[0] : box?.operator;
  const head = typeof headRaw === "string" ? headRaw.toLowerCase() : null;

  if (head === "equal" || head === "assign") {
    const ja = jsonArr(j);
    const lhs = ja[1];
    const rhs = ja[2];
    const asDef =
      latexLooksLikeFunctionDef(src) || isUserFunctionCall(lhs);

    if (asDef) {
      const { funcName, funcArgs } = parseFuncDefLhs(src, lhs);
      const rhsBox = ce.box(rhs as never);
      const rhsLatex = rhsBox.latex || src.split("=").slice(1).join("=").trim();
      if (!rhsLatex) throw new Error("Empty right-hand side");
      return {
        kind: "funcdef",
        shade: "none",
        isoLevel: 0,
        compileLatex: rhsLatex,
        label: `function ${funcName}`,
        funcName,
        funcArgs,
      };
    }

    // Bare free symbol on LHS → parameter or spatial alias.
    const lhsName = symbolId(lhs);
    if (
      lhsName &&
      !RESERVED_SYMBOLS.has(lhsName) &&
      /^[A-Za-z][A-Za-z0-9_]*$/.test(lhsName)
    ) {
      const rhsBox = ce.box(rhs as never);
      const rhsLatex = rhsBox.latex || src.split("=").slice(1).join("=").trim();
      if (!rhsLatex) throw new Error("Empty right-hand side");
      const rhsResult = compileLatex(rhsLatex);
      const { usesSpace } = collectFreeParams(rhsResult.freeSymbols, rhsLatex, {
        skipName: lhsName,
      });
      if (usesSpace) {
        return {
          kind: "alias",
          shade: "none",
          isoLevel: 0,
          compileLatex: rhsLatex,
          label: `alias ${lhsName}`,
          aliasName: lhsName,
        };
      }
      return {
        kind: "parameter",
        shade: "none",
        isoLevel: 0,
        compileLatex: rhsLatex,
        label: `parameter ${lhsName}`,
        paramName: lhsName,
      };
    }

    const diff = ce.box(["Subtract", lhs, rhs] as never);
    const constraintLatex = diff.latex;
    if (!constraintLatex) throw new Error("Could not form constraint residual");
    return {
      kind: "constraint",
      shade: "iso",
      isoLevel: 0,
      compileLatex: constraintLatex,
      label: "constraint → isosurface",
    };
  }

  return {
    kind: "bare",
    shade: "volume",
    isoLevel: 0,
    compileLatex: src,
    label: "expression → volume",
  };
}

/**
 * Compile a named parameter definition `a = <expr>` (or bare RHS for `expectedName`).
 * Free symbols besides reserved/`expectedName` are other parameters.
 *
 * @param {string} raw
 * @param {string} expectedName
 * @returns {{
 *   name: string,
 *   rhsLatex: string,
 *   freeParams: string[],
 *   isConstant: boolean,
 *   constantValue: number | null,
 *   eval: (scope?: Record<string, number>) => number,
 * }}
 */
export function compileParamLatex(raw: string, expectedName: string): CompiledParam {
  const src = normalizeLatexFunctions(String(raw ?? "").trim());
  if (!src) throw new Error("Empty parameter");

  let name = expectedName;
  let rhsLatex = src;

  let box;
  try {
    box = ce.parse(src);
  } catch {
    box = null;
  }

  const j = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);
  const headRaw = Array.isArray(j) ? j[0] : box?.operator;
  const head = typeof headRaw === "string" ? headRaw.toLowerCase() : null;
  if (head === "equal" || head === "assign") {
    const ja = jsonArr(j);
    const rhsBox = ce.box(ja[2] as never);
    rhsLatex = rhsBox?.latex || src.split("=").slice(1).join("=").trim();
  } else if (expectedName) {
    // Bare number / expr → treat as RHS for expectedName.
    rhsLatex = src;
  }

  if (!rhsLatex) throw new Error("Empty parameter right-hand side");
  // Slot name is authoritative (field may show a=… or a bare RHS).
  name = expectedName;

  const result = compileLatex(rhsLatex);
  if (!result?.success || typeof result.run !== "function") {
    const why = result?.unsupported?.length
      ? `unsupported: ${result.unsupported.join(", ")}`
      : "could not compile";
    throw new Error(`Parameter ${why}`);
  }

  const { run } = result;
  const { freeParams } = collectFreeParams(result.freeSymbols, rhsLatex, {
    skipName: name,
  });

  let constantValue = null;
  let isConstant = freeParams.length === 0;
  if (isConstant) {
    try {
      constantValue = coerceNumber(run({}));
      if (!Number.isFinite(constantValue)) {
        isConstant = false;
        constantValue = null;
      }
    } catch {
      isConstant = false;
      constantValue = null;
    }
  }

  return {
    name,
    rhsLatex,
    freeParams,
    isConstant,
    constantValue,
    eval(scope = {}) {
      return coerceNumber(run(scope));
    },
  };
}

/** Format a number for embedding in `name=<value>` LaTeX. */
export function formatParamLatexValue(v: number) {
  if (!Number.isFinite(v)) return "0";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-4)) return v.toExponential(6).replace(/e\+?/, "e");
  const s = String(Math.round(v * 1e9) / 1e9);
  return s;
}

/**
 * Compile a density / constraint expression.
 *
 * Plain 0th-order fields (no dependence on x,y,z / r,θ,φ,ρ) get `shade: "none"`
 * and are not graphed — constants and param-only expressions stay off the volume.
 *
 * @returns {{
 *   freeParams: string[],
 *   usesSpace: boolean,
 *   kind: "constraint" | "definition" | "bare",
 *   shade: "iso" | "volume" | "none",
 *   isoLevel: number,
 *   classifyLabel: string,
 *   bind: (params?: Record<string, number>) => (x: number, y: number, z: number) => number,
 * }}
 */
function bindLaplacianFromScalar(
  scalarFn: (x: number, y: number, z: number) => number,
  eps = 1e-5,
): (x: number, y: number, z: number) => number {
  const h2 = eps * eps;
  return (x: number, y: number, z: number) => {
    const c = scalarFn(x, y, z);
    const d2x =
      (scalarFn(x + eps, y, z) - 2 * c + scalarFn(x - eps, y, z)) / h2;
    const d2y =
      (scalarFn(x, y + eps, z) - 2 * c + scalarFn(x, y - eps, z)) / h2;
    const d2z =
      (scalarFn(x, y, z + eps) - 2 * c + scalarFn(x, y, z - eps)) / h2;
    return d2x + d2y + d2z;
  };
}

function bindDivergenceFromVector(
  vectorFn: (x: number, y: number, z: number) => [number, number, number],
  eps = 1e-5,
): (x: number, y: number, z: number) => number {
  return (x: number, y: number, z: number) => {
    const dVx_dx =
      (vectorFn(x + eps, y, z)[0]! - vectorFn(x - eps, y, z)[0]!) / (2 * eps);
    const dVy_dy =
      (vectorFn(x, y + eps, z)[1]! - vectorFn(x, y - eps, z)[1]!) / (2 * eps);
    const dVz_dz =
      (vectorFn(x, y, z + eps)[2]! - vectorFn(x, y, z - eps)[2]!) / (2 * eps);
    return dVx_dx + dVy_dy + dVz_dz;
  };
}

function scalarFromUnaryOpJsonLocal(json: unknown, op: string): string | null {
  return scalarFromUnaryOpJson(json, op);
}

function tripleFromUnaryOpJsonLocal(json: unknown, op: string): string[] | null {
  return tripleFromUnaryOpJson(json, op);
}

function looksLikeLaplacian(src: string, json: unknown): string | null {
  const fromJson = scalarFromUnaryOpJsonLocal(json, "laplacian");
  if (fromJson) return fromJson.trim();

  if (/\\laplacian|\\Delta|\\nabla\s*\^|\\operatorname\s*\{\s*laplacian/i.test(src)) {
    const m = src.match(
      /\\(?:operatorname\s*\{\s*laplacian\s*\}|laplacian|Delta)\s*(?:\\left)?[\{\(]?\s*([\s\S]+?)\s*(?:\\right)?[\}\)]?\s*$/i,
    );
    if (m?.[1]) return m[1].trim();
    const m2 = src.match(
      /\\nabla\s*\^\s*\{?\s*2\s*\}?\s*(?:\\left)?[\{\(]?\s*([\s\S]+?)\s*(?:\\right)?[\}\)]?\s*$/i,
    );
    if (m2?.[1]) return m2[1].trim();
  }
  return null;
}

function looksLikeDivergence(src: string, json: unknown): string[] | null {
  const fromJson = tripleFromUnaryOpJsonLocal(json, "div");
  if (fromJson?.length === 3) return fromJson;

  if (/\\div|\\operatorname\s*\{\s*div/i.test(src)) {
    const m = src.match(
      /\\(?:operatorname\s*\{\s*div\s*\}|div)\s*(?:\\left)?[\{\(]?\s*([\s\S]+?)\s*(?:\\right)?[\}\)]?\s*$/i,
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

function bindScalarFromLatex(latex: string, freeParams: string[]) {
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

function bindTupleFromLatex(parts: string[]) {
  const compiled = parts.map((latex) => {
    const r = compileLatex(latex);
    if (!r?.success || typeof r.run !== "function") {
      throw new Error(`Could not compile vector component: ${latex}`);
    }
    return r;
  });
  return (params: Record<string, number> = {}) =>
    (x: number, y: number, z: number): [number, number, number] => {
      const { r, theta, phi, rho } = polarFromCartesian(x, y, z);
      const out: [number, number, number] = [0, 0, 0];
      for (let i = 0; i < parts.length; i++) {
        const fp = collectFreeParams(compiled[i]!.freeSymbols, parts[i]!).freeParams;
        const scope: Record<string, number> = { x, y, z, r, theta, phi, rho };
        for (const name of fp) {
          const v = params[name];
          scope[name] = Number.isFinite(v) ? v : 1;
        }
        out[i] = coerceNumber(compiled[i]!.run!(scope));
      }
      return out;
    };
}

function compileLaplacianExpr(
  raw: string,
  scalarLatex: string,
  classified: ClassifiedExpr,
): CompiledExpr {
  const scalarResult = compileLatex(scalarLatex);
  if (!scalarResult?.success || typeof scalarResult.run !== "function") {
    throw new Error("Could not compile scalar inside laplacian");
  }
  const { freeParams, usesSpace } = collectFreeParams(scalarResult.freeSymbols, scalarLatex);
  if (!usesSpace) throw new Error("Laplacian field must depend on x, y, or z");
  const shade = classified.shade === "none" ? "none" : classified.shade;
  return {
    freeParams,
    usesSpace,
    kind: classified.kind as FieldKind,
    shade,
    isoLevel: classified.isoLevel,
    classifyLabel: "laplacian field",
    operator: "laplacian",
    scalarCompileLatex: scalarLatex,
    bind(params: Record<string, number> = {}) {
      const scalar = bindScalarFromLatex(scalarLatex, freeParams)(params);
      return bindLaplacianFromScalar(scalar);
    },
    bindScalar(params: Record<string, number> = {}) {
      return bindScalarFromLatex(scalarLatex, freeParams)(params);
    },
  };
}

function compileDivergenceExpr(
  raw: string,
  parts: [string, string, string],
  classified: ClassifiedExpr,
): CompiledExpr {
  const compiled = parts.map((latex) => compileLatex(latex));
  const freeSet = new Set<string>();
  let usesSpace = false;
  for (let i = 0; i < compiled.length; i++) {
    const { freeParams, usesSpace: us } = collectFreeParams(
      compiled[i]!.freeSymbols,
      parts[i]!,
    );
    for (const p of freeParams) freeSet.add(p);
    if (us) usesSpace = true;
  }
  const freeParams = [...freeSet].sort();
  if (!usesSpace) throw new Error("Divergence field must depend on x, y, or z");
  const shade = classified.shade === "none" ? "none" : classified.shade;
  return {
    freeParams,
    usesSpace,
    kind: classified.kind as FieldKind,
    shade,
    isoLevel: classified.isoLevel,
    classifyLabel: "divergence field",
    operator: "divergence",
    divergenceParts: parts,
    bind(params: Record<string, number> = {}) {
      const tuple = bindTupleFromLatex(parts)(params);
      return bindDivergenceFromVector(tuple);
    },
    bindTuple(params: Record<string, number> = {}) {
      return bindTupleFromLatex(parts)(params);
    },
  };
}

/**
 * Flow-field syntax must use compileVectorExpr / flow role — not scalar compileExpr.
 * Cheap pre-check avoids CE "Tuple + \\grad" console noise.
 */
function isLikelyFlowLatex(raw: string): boolean {
  const s = String(raw ?? "").trim();
  if (/\\curl|\\nabla\s*\\times/i.test(s)) return true;
  if (/\\nabla\s*\^|\^2|\\laplacian|\\Delta|\\div|\\nabla\s*\\cdot/i.test(s)) return false;
  if (/\\grad|\\nabla/i.test(s)) return true;
  const m = s.match(/^\(([\s\S]+)\)$/);
  if (m) {
    let depth = 0;
    let parts = 0;
    for (let i = 0; i < m[1]!.length; i++) {
      const c = m[1]![i];
      if (c === "(") depth++;
      else if (c === ")") depth--;
      else if (c === "," && depth === 0) parts++;
    }
    if (parts === 2) return true;
  }
  return false;
}

export function compileExpr(raw: string, registry?: SymbolRegistry): CompiledExpr {
  const expandedRaw = registry ? expandDefinitions(raw, registry) : raw;

  if (isLikelyFlowLatex(expandedRaw)) {
    throw new Error("Vector field — use flow role (tuple, \\grad, or \\curl)");
  }

  const normalized = normalizeCalcLatex(normalizeLatexFunctions(String(expandedRaw ?? "").trim()));
  if (normalized) {
    let box;
    try {
      box = ce.parse(normalized);
    } catch {
      box = null;
    }
    const j = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);
    const lapInner = looksLikeLaplacian(normalized, j);
    if (lapInner) {
      const classified = classifyExpr(expandedRaw);
      return compileLaplacianExpr(expandedRaw, lapInner, classified);
    }
    const divParts = looksLikeDivergence(normalized, j);
    if (divParts?.length === 3) {
      const classified = classifyExpr(expandedRaw);
      return compileDivergenceExpr(expandedRaw, divParts as [string, string, string], classified);
    }
  }

  const classified = classifyExpr(expandedRaw);
  const src = classified.compileLatex;

  const result = compileLatex(src);
  if (!result?.success || typeof result.run !== "function") {
    const why = result?.unsupported?.length
      ? `unsupported: ${result.unsupported.join(", ")}`
      : "could not compile";
    throw new Error(`Expression ${why}`);
  }

  const { run } = result;
  const { freeParams, usesSpace } = collectFreeParams(result.freeSymbols, src);

  const shade = usesSpace ? classified.shade : "none";

  return {
    freeParams,
    usesSpace,
    kind: classified.kind as FieldKind,
    shade,
    isoLevel: classified.isoLevel,
    classifyLabel: usesSpace ? classified.label : "constant (not graphed)",
    /** Bind current parameter values → f(x,y,z); injects r,θ,φ,ρ. */
    bind(params: Record<string, number> = {}) {
      return (x: number, y: number, z: number) => {
        const { r, theta, phi, rho } = polarFromCartesian(x, y, z);
        const scope: Record<string, number> = { x, y, z, r, theta, phi, rho };
        for (const name of freeParams) {
          const v = params[name];
          scope[name] = Number.isFinite(v) ? v : 1;
        }
        return coerceNumber(run(scope));
      };
    },
  };
}

/** Preset densities as LaTeX (shown in the MathLive field). */
export const PRESETS: Record<string, PresetDef> = {
  sincos: {
    label: "y = sin(x) cos(x)",
    latex: String.raw`y=\sin(x)\cos(z)`,
  },
  blob: {
    label: "Gaussian blob",
    latex: String.raw`e^{-(x^{2}+y^{2}+z^{2})}`,
  },
  soft: {
    label: "Soft ellipsoid",
    latex: String.raw`\exp(-(x^{2}+0.5y^{2}+2z^{2}))`,
  },
  two: {
    label: "Two blobs",
    latex: String.raw`\exp(-4((x-0.7)^{2}+y^{2}+z^{2}))+\exp(-4((x+0.7)^{2}+y^{2}+z^{2}))`,
  },
  shell: {
    label: "Spherical shell",
    latex: String.raw`\exp(-12(r-0.9)^{2})`,
  },
  lobe: {
    label: "Polar lobe (θ)",
    latex: String.raw`\exp(-4r^{2})\max(0,\cos\theta)^{2}`,
  },
  torus: {
    label: "Polar torus (ρ)",
    latex: String.raw`\exp(-20(\rho-0.9)^{2}-8z^{2})`,
  },
  ridge: {
    label: "Vertical ridge",
    latex: String.raw`\exp(-10x^{2})\exp(-0.4(y^{2}+z^{2}))`,
  },
  pulse: {
    label: "Pulse blob (a)",
    latex: String.raw`\exp(-r^{2}/a^{2})`,
    params: { a: { value: 1, min: 0.35, max: 1.6, animate: true } },
  },
  twist: {
    label: "Two blobs (d)",
    latex: String.raw`\exp(-4((x-d)^{2}+y^{2}+z^{2}))+\exp(-4((x+d)^{2}+y^{2}+z^{2}))`,
    params: { d: { value: 0.7, min: 0.15, max: 1.2, animate: true } },
  },
  sphere: {
    label: "Sphere (constraint)",
    latex: String.raw`x^{2}+y^{2}+z^{2}=1`,
  },
  torusSigned: {
    label: "Torus (constraint)",
    latex: String.raw`(\rho-0.9)^{2}+z^{2}=0.12`,
  },
  swirl: {
    label: "Flow swirl",
    expressions: [{ latex: String.raw`(-y, x, 0)`, role: "flow" }],
  },
  swirlGrad: {
    label: "Flow (grad r²)",
    expressions: [{ latex: String.raw`\grad(x^2+y^2+z^2)`, role: "flow" }],
  },
  lavalamp: {
    label: "Lava lamp (animated blobs)",
    expressions: [
      {
        latex: "u=0.5",
        autoParam: true,
        sliderMin: -0.85,
        sliderMax: 0.85,
        sliderSpeed: 0.12,
        sliderAnimating: true,
        sliderAnimMode: "loop",
        sliderPhase: 0.1,
      },
      {
        latex: "v=-0.4",
        autoParam: true,
        sliderMin: -0.9,
        sliderMax: 0.9,
        sliderSpeed: 0.14,
        sliderAnimating: true,
        sliderAnimMode: "loop",
        sliderPhase: 0.55,
      },
      {
        latex: "w=0.1",
        autoParam: true,
        sliderMin: -0.8,
        sliderMax: 0.8,
        sliderSpeed: 0.11,
        sliderAnimating: true,
        sliderAnimMode: "loop",
        sliderPhase: 0.85,
      },
      {
        latex: String.raw`\exp(-5((x-0.45)^2+(y-u)^2+(z+0.25)^2))`,
        color: "#ff4500",
        color2: "#ffec00",
      },
      {
        latex: String.raw`\exp(-4((x+0.35)^2+(y-v)^2+(z-0.3)^2))`,
        color: "#ff6b4a",
        color2: "#ffb0d8",
      },
      {
        latex: String.raw`\exp(-4.5((x-0.15)^2+(y-w)^2+(z-0.45)^2))`,
        color: "#ff1493",
        color2: "#7b2fff",
      },
    ],
    params: {
      u: { value: 0.5, min: -0.85, max: 0.85, animate: true, speed: 0.12, animMode: "loop", phase: 0.1 },
      v: { value: -0.4, min: -0.9, max: 0.9, animate: true, speed: 0.14, animMode: "loop", phase: 0.55 },
      w: { value: 0.1, min: -0.8, max: 0.8, animate: true, speed: 0.11, animMode: "loop", phase: 0.85 },
    },
  },
};

function fromUnit(u: number, a: number, b: number) {
  return 0.5 * (a + b) + 0.5 * (b - a) * u;
}

/**
 * DCT matrix W[i,a] = T_i(u_a) for Gauss–Chebyshev nodes.
 * Built via recurrence in O(n²).
 */
function chebWeightMatrix(n: number, uNodes: number[]) {
  const W = new Float64Array(n * n);
  for (let a = 0; a < n; a++) {
    const u = uNodes[a];
    W[a] = 1;
    if (n > 1) W[n + a] = u;
    for (let i = 2; i < n; i++) {
      W[i * n + a] = 2 * u * W[(i - 1) * n + a] - W[(i - 2) * n + a];
    }
  }
  return W;
}

/**
 * Separable 3D Chebyshev DCT of samples on the tensor Chebyshev grid.
 * Same math as the naive O(n⁶) sum, but three 1D passes → O(n⁴).
 *
 * c_ijk = (α_i α_j α_k / n³) Σ_{a,b,c} f_abc T_i(u_a) T_j(u_b) T_k(u_c)
 * with α_0 = 1, α_{>0} = 2.
 *
 * Packing: idx = x + y*n + z*n*n (same as vals / cheb elsewhere).
 */
function chebDCT3DSeparable(vals: Float64Array, n: number, uNodes: number[]) {
  const W = chebWeightMatrix(n, uNodes);
  const scale = new Float64Array(n);
  for (let i = 0; i < n; i++) scale[i] = (i === 0 ? 1 : 2) / n;

  const n2 = n * n;
  const tmp = new Float64Array(n * n * n);
  const tmp2 = new Float64Array(n * n * n);
  const out = new Float32Array(n * n * n);

  // X: vals[a,b,c] → tmp[i,b,c]
  for (let b = 0; b < n; b++) {
    for (let c = 0; c < n; c++) {
      const base = b * n + c * n2;
      for (let i = 0; i < n; i++) {
        let s = 0;
        const Wi = i * n;
        for (let a = 0; a < n; a++) s += vals[a + base] * W[Wi + a];
        tmp[i + base] = s * scale[i];
      }
    }
  }

  // Y: tmp[i,b,c] → tmp2[i,j,c]
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < n; c++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        const Wj = j * n;
        for (let b = 0; b < n; b++) s += tmp[i + b * n + c * n2] * W[Wj + b];
        tmp2[i + j * n + c * n2] = s * scale[j];
      }
    }
  }

  // Z: tmp2[i,j,c] → out[i,j,k]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const ij = i + j * n;
      for (let k = 0; k < n; k++) {
        let s = 0;
        const Wk = k * n;
        for (let c = 0; c < n; c++) s += tmp2[ij + c * n2] * W[Wk + c];
        out[ij + k * n2] = s * scale[k];
      }
    }
  }

  return out;
}

/** T_0..T_deg as monomial coeffs in u (length deg+1 arrays). */
function chebToMonoTable(deg: number) {
  const T: number[][] = [[1], [0, 1]];
  for (let n = 2; n <= deg; n++) {
    const prev = T[n - 1];
    const prev2 = T[n - 2];
    const cur = new Array(n + 1).fill(0);
    for (let i = 0; i < prev.length; i++) cur[i + 1] += 2 * prev[i];
    for (let i = 0; i < prev2.length; i++) cur[i] -= prev2[i];
    T[n] = cur;
  }
  return T;
}

/**
 * Chebyshev tensor c_ijk T_i(x/h)T_j(y/h)T_k(z/h)
 * → monomials m_abc for x^a y^b z^c (same packing).
 */
function chebToMonomial3D(chebCoeffs: ArrayLike<number>, deg: number, half: number) {
  const N = deg;
  const n = N + 1;
  const T = chebToMonoTable(N);
  const invH = 1 / half;
  const mono = new Float64Array(n * n * n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        const c = chebCoeffs[i + j * n + k * n * n];
        if (Math.abs(c) < 1e-18) continue;
        const Ti = T[i];
        const Tj = T[j];
        const Tk = T[k];
        for (let a = 0; a < Ti.length; a++) {
          if (Ti[a] === 0) continue;
          for (let b = 0; b < Tj.length; b++) {
            if (Tj[b] === 0) continue;
            for (let d = 0; d < Tk.length; d++) {
              if (Tk[d] === 0) continue;
              const scale = c * Ti[a] * Tj[b] * Tk[d] * invH ** (a + b + d);
              mono[a + b * n + d * n * n] += scale;
            }
          }
        }
      }
    }
  }
  return Float32Array.from(mono);
}

function evalMonomial3D(mono: ArrayLike<number>, deg: number, x: number, y: number, z: number) {
  const n = deg + 1;
  let s = 0;
  let xp = 1;
  for (let i = 0; i < n; i++) {
    let yp = 1;
    for (let j = 0; j < n; j++) {
      let zp = 1;
      for (let k = 0; k < n; k++) {
        s += mono[i + j * n + k * n * n] * xp * yp * zp;
        zp *= z;
      }
      yp *= y;
    }
    xp *= x;
  }
  return s;
}

/**
 * Fit f on [-half,half]^3 with tensor Chebyshev, convert to world monomials.
 * @param {{ skipL2?: boolean, skipMono?: boolean }} [opts]
 */
export function fitChebyshev3D(
  fn: (x: number, y: number, z: number) => number,
  half: number,
  deg: number,
  opts: { skipL2?: boolean; skipMono?: boolean } = {},
): ChebFitResult {
  const tAll = performance.now();
  const N = Math.max(0, Math.min(MAX_DEG, deg | 0));
  const n = N + 1;
  const uNodes = new Array(n);
  const pts = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.cos((Math.PI * (2 * i + 1)) / (2 * n));
    uNodes[i] = u;
    pts[i] = fromUnit(u, -half, half);
  }

  let t0 = performance.now();
  const vals = new Float64Array(n * n * n);
  let fMin = Infinity;
  let fMax = -Infinity;
  for (let ix = 0; ix < n; ix++) {
    for (let iy = 0; iy < n; iy++) {
      for (let iz = 0; iz < n; iz++) {
        const v = fn(pts[ix], pts[iy], pts[iz]);
        if (!Number.isFinite(v)) throw new Error(`f NaN at sample (${pts[ix]}, ${pts[iy]}, ${pts[iz]})`);
        vals[ix + iy * n + iz * n * n] = v;
        fMin = Math.min(fMin, v);
        fMax = Math.max(fMax, v);
      }
    }
  }
  const sampleMs = performance.now() - t0;

  t0 = performance.now();
  // Separable DCT: O(n⁴). Mutates vals as scratch after the X-pass buffer.
  const cheb = chebDCT3DSeparable(vals, n, uNodes);
  const chebMs = performance.now() - t0;

  let mono = null;
  let monoMs = 0;
  if (!opts.skipMono) {
    t0 = performance.now();
    mono = chebToMonomial3D(cheb, N, half);
    monoMs = performance.now() - t0;
  }

  let fitRelL2 = NaN;
  let l2Ms = 0;
  if (!opts.skipL2) {
    if (!mono) {
      t0 = performance.now();
      mono = chebToMonomial3D(cheb, N, half);
      monoMs += performance.now() - t0;
    }
    t0 = performance.now();
    const M = 10;
    let num = 0;
    let den = 0;
    for (let ix = 0; ix < M; ix++) {
      for (let iy = 0; iy < M; iy++) {
        for (let iz = 0; iz < M; iz++) {
          const x = -half + (2 * half * (ix + 0.5)) / M;
          const y = -half + (2 * half * (iy + 0.5)) / M;
          const z = -half + (2 * half * (iz + 0.5)) / M;
          const truth = fn(x, y, z);
          const approx = evalMonomial3D(mono, N, x, y, z);
          const d = approx - truth;
          num += d * d;
          den += truth * truth;
        }
      }
    }
    fitRelL2 = Math.sqrt(num) / (Math.sqrt(den) + 1e-15);
    l2Ms = performance.now() - t0;
  }

  const totalMs = performance.now() - tAll;
  return {
    cheb,
    mono,
    deg: N,
    half,
    fitRelL2,
    fMin,
    fMax,
    timing: {
      sampleMs,
      chebMs,
      monoMs,
      l2Ms,
      totalMs,
    },
  };
}

/** Bake scalar density with optional spectral laplacian / divergence operators. */
export function fitScalarField(
  compiled: CompiledExpr,
  fn: (x: number, y: number, z: number) => number,
  half: number,
  deg: number,
  opts: { skipL2?: boolean; skipMono?: boolean } = {},
): ScalarFitResult & { fitRelL2: number; timing?: ChebFitTiming } {
  const skipL2 = opts.skipL2 ?? false;
  const skipMono = opts.skipMono ?? false;

  if (compiled.operator === "laplacian" && compiled.bindScalar) {
    const fit = fitChebyshev3D(compiled.bindScalar(), half, deg, { skipL2, skipMono });
    const lap = idctChebLaplacian3D(fit.cheb, deg, deg + 1);
    const scale = (1 / half) ** 2;
    const dens = new Float32Array(lap.dens.length);
    for (let i = 0; i < dens.length; i++) dens[i] = lap.dens[i]! * scale;
    return {
      dens,
      cheb: fit.cheb,
      fitRelL2: fit.fitRelL2,
      M: lap.M,
      deg: fit.deg,
      timing: fit.timing,
    };
  }

  if (compiled.operator === "divergence" && compiled.bindTuple) {
    const tupleFn = compiled.bindTuple();
    const fitX = fitChebyshev3D((x, y, z) => tupleFn(x, y, z)[0]!, half, deg, {
      skipL2,
      skipMono,
    });
    const fitY = fitChebyshev3D((x, y, z) => tupleFn(x, y, z)[1]!, half, deg, {
      skipL2,
      skipMono,
    });
    const fitZ = fitChebyshev3D((x, y, z) => tupleFn(x, y, z)[2]!, half, deg, {
      skipL2,
      skipMono,
    });
    const div = idctChebDivergence3D(fitX.cheb, fitY.cheb, fitZ.cheb, deg, deg + 1);
    const scale = 1 / half;
    const dens = new Float32Array(div.dens.length);
    for (let i = 0; i < dens.length; i++) dens[i] = div.dens[i]! * scale;
    return {
      dens,
      cheb: fitX.cheb,
      fitRelL2: fitX.fitRelL2,
      M: div.M,
      deg: fitX.deg,
      timing: fitX.timing,
    };
  }

  const fit = fitChebyshev3D(fn, half, deg, { skipL2, skipMono });
  const idct = idctCheb3D(fit.cheb, fit.deg, fit.deg + 1);
  return {
    dens: idct.dens,
    cheb: fit.cheb,
    fitRelL2: fit.fitRelL2,
    M: idct.M,
    deg: fit.deg,
    timing: fit.timing,
  };
}
