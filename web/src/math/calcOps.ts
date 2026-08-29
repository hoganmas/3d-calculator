/** Shared vector-calculus LaTeX normalization and MathJSON tuple extraction. */

import { ComputeEngine } from "@cortex-js/compute-engine";
import type { IntegralAxisSpec } from "../types/models.js";

const ce = new ComputeEngine();

/** Alternate spellings CE does not recognize as commands. */
export function normalizeLatexAliases(latex: string): string {
  let s = String(latex ?? "").replace(/\\del/gi, "\\nabla");
  s = normalizeDegreeLatex(s);
  return s;
}

/**
 * CE parses \\deg as the symbolic `Degree` unit, which has no JavaScript lowering.
 * Rewrite to \\circ / ° forms the engine can evaluate as radians.
 */
export function normalizeDegreeLatex(latex: string): string {
  let s = String(latex ?? "");
  s = s.replace(/([\d)A-Za-z}\]])\s*\\deg(?![A-Za-z])/gi, "$1^\\circ");
  s = s.replace(/([\d)A-Za-z}\]])\s*\\degree(?![A-Za-z])/gi, "$1^\\circ");
  s = s.replace(/^\\deg(?![A-Za-z])$/gi, "°");
  return s;
}

/** True when CE embedded an Error node or other un-compilable MathJSON during typing. */
export function mathJsonHasError(json: unknown): boolean {
  if (json === "Degree") return true;
  if (!Array.isArray(json)) return false;
  if (json[0] === "Error") return true;
  for (let i = 1; i < json.length; i++) {
    if (mathJsonHasError(json[i])) return true;
  }
  return false;
}

/** Strip \\mathrm{V}, \\mathbf{V}, etc. to plain identifiers (MathLive output). */
export function unwrapLatexSymbolTokens(latex: string): string {
  let s = String(latex ?? "");
  for (let i = 0; i < 8; i++) {
    const next = s.replace(
      /(?<![A-Za-z])\\(?:mathrm|mathbf|mathit|textrm|bf|it)\s*\{([A-Za-z][A-Za-z0-9_]*)\}/g,
      "$1",
    );
    if (next === s) break;
    s = next;
  }
  return s;
}

const VECTOR_CALC_OP_NAMES = ["div", "curl", "grad", "laplacian"] as const;

/** CE / MathLive often emit \\mathrm{div} — normalize to \\operatorname before unwrap. */
function normalizeVectorCalcOperatorForms(s: string): string {
  let out = s.replace(
    /\\(?:mathrm|mathbf|mathit|textrm|bf|it)\s*\{\s*\\nabla\s*\}/gi,
    "\\nabla",
  );
  out = out.replace(/\\nabla\s*\\!\s*\\cdot/gi, "\\nabla\\cdot");
  for (const op of VECTOR_CALC_OP_NAMES) {
    out = out.replace(
      new RegExp(
        `\\\\(?:mathrm|mathbf|mathit|textrm|bf|it)\\\\s*\\\\{\\\\s*${op}\\\\s*\\\\}`,
        "gi",
      ),
      `\\operatorname{${op}}`,
    );
  }
  return out;
}

/** After unwrap, bare `div(` must not become `d*i*v(` juxtaposition. */
function normalizeBareVectorCalcOperators(s: string): string {
  let out = s;
  for (const op of ["div", "curl", "laplacian"] as const) {
    out = out.replace(
      new RegExp(`(^|[^\\\\A-Za-z])${op}(?=\\s*[({])`, "gi"),
      `$1\\operatorname{${op}}`,
    );
  }
  out = out.replace(/(^|[^\\A-Za-z])grad(?=\s*[({])/gi, "$1\\operatorname{grad}");
  return out;
}

/** Extract the argument of \\operatorname{op}… with balanced parens/braces or a bare symbol. */
export function extractOperatornameArg(src: string, op: string): string | null {
  const re = new RegExp(`\\\\operatorname\\s*\\{\\s*${op}\\s*\\}`, "i");
  const m = re.exec(src);
  if (!m) return null;
  let rest = src.slice(m.index + m[0].length).replace(/^\s*\\left\s*/, "").trimStart();
  if (!rest) return null;

  const open = rest[0];
  if (open === "(" || open === "{") {
    const close = open === "(" ? ")" : "}";
    let depth = 0;
    for (let i = 0; i < rest.length; i++) {
      const c = rest[i]!;
      if (c === open) depth++;
      else if (c === close) {
        depth--;
        if (depth === 0) return rest.slice(1, i).trim();
      }
    }
    return null;
  }

  const bare = rest.match(
    /^((?:\\(?:mathrm|mathbf|mathit|textrm|bf|it)\s*\{[^{}]+\}|[A-Za-z][A-Za-z0-9_]*(?:\([^()]*\))?))/,
  );
  return bare?.[1]?.trim() ?? null;
}

export type ChebAxis = 0 | 1 | 2;

const PARTIAL_OP_NAMES = ["partial_x", "partial_y", "partial_z"] as const;

function axisFromVar(v: string): ChebAxis | null {
  const c = v.toLowerCase();
  if (c === "x") return 0;
  if (c === "y") return 1;
  if (c === "z") return 2;
  return null;
}

function partialOpName(axis: ChebAxis): string {
  return PARTIAL_OP_NAMES[axis]!;
}

function extractUnaryTail(rest: string): string | null {
  let s = rest.replace(/^\s*\\left\s*/, "").trimStart();
  if (!s) return null;

  const open = s[0];
  if (open === "(" || open === "{") {
    const close = open === "(" ? ")" : "}";
    let depth = 0;
    for (let i = 0; i < s.length; i++) {
      const c = s[i]!;
      if (c === open) depth++;
      else if (c === close) {
        depth--;
        if (depth === 0) return s.slice(1, i).trim();
      }
    }
    return null;
  }

  return s.trim();
}

function wrapPartial(axis: ChebAxis, inner: string): string {
  return `\\operatorname{${partialOpName(axis)}}{${inner.trim()}}`;
}

/** Normalize partial-derivative LaTeX before nabla→grad rules. */
export function normalizePartialForms(latex: string): string {
  let s = String(latex ?? "").trim();

  const subscript = s.match(
    /^\\(?:partial|grad)\s*_\s*\{?\s*([xyz])\s*\}?\s*([\s\S]+)$/i,
  );
  if (subscript?.[1] && subscript[2]) {
    const axis = axisFromVar(subscript[1]);
    const inner = extractUnaryTail(subscript[2]);
    if (axis != null && inner) return wrapPartial(axis, inner);
  }

  const fracInner = s.match(
    /^\\frac\s*\{\s*\\partial\s+([\s\S]+?)\s*\}\s*\{\s*\\partial\s*([xyz])\s*\}$/i,
  );
  if (fracInner?.[1] && fracInner[2]) {
    const axis = axisFromVar(fracInner[2]);
    if (axis != null) return wrapPartial(axis, fracInner[1]);
  }

  const fracPrefix = s.match(
    /^\\frac\s*\{\s*\\partial\s*\}\s*\{\s*\\partial\s*([xyz])\s*\}\s*([\s\S]+)$/i,
  );
  if (fracPrefix?.[1] && fracPrefix[2]) {
    const axis = axisFromVar(fracPrefix[1]);
    const inner = extractUnaryTail(fracPrefix[2]);
    if (axis != null && inner) return wrapPartial(axis, inner);
  }

  return s;
}

export interface PeeledDefiniteIntegral {
  inner: string;
  axes: IntegralAxisSpec[];
}

/** Peel chained \\int_{a}^{b} … dvar from the outside in. */
export function peelDefiniteIntegrals(latex: string): PeeledDefiniteIntegral | null {
  let s = String(latex ?? "").trim();
  const bounds: { a: string; b: string }[] = [];

  for (;;) {
    const m = s.match(/^\\int\s*(?:_\s*\{([^}]*)\})?\s*(?:\^\s*\{([^}]*)\})?\s*/);
    if (!m) break;
    bounds.push({ a: (m[1] ?? "").trim(), b: (m[2] ?? "").trim() });
    s = s.slice(m[0].length).trim();
  }

  if (bounds.length === 0) return null;

  const axes: IntegralAxisSpec[] = [];
  for (let i = 0; i < bounds.length; i++) {
    const dm = s.match(/\s*(?:\\,)?\s*d\s*([xyz])\s*$/i);
    if (!dm) return null;
    const axis = axisFromVar(dm[1]!);
    if (axis == null) return null;
    s = s.slice(0, dm.index).trim();
    axes.push({
      axis,
      aLatex: bounds[i]!.a,
      bLatex: bounds[i]!.b,
    });
  }

  const inner = s.trim();
  if (!inner) return null;
  // Innermost integral last in the array (application order).
  axes.reverse();
  return { inner, axes };
}

export function normalizeCalcLatex(latex: string): string {
  let s = String(latex ?? "").trim();
  s = s.replace(/\\left\s*/g, "");
  s = s.replace(/\\right\s*/g, "");
  s = normalizePartialForms(s);
  // MathLive: \grad\mathrm{f}, \nabla\mathrm{f}
  s = s.replace(
    /\\grad\\(?:mathrm|mathbf|mathit)\s*\{([A-Za-z][A-Za-z0-9_]*)\}/g,
    "\\operatorname{grad}{$1}",
  );
  s = s.replace(
    /\\nabla\\(?:mathrm|mathbf|mathit)\s*\{([A-Za-z][A-Za-z0-9_]*)\}/g,
    "\\operatorname{grad}{$1}",
  );
  s = normalizeVectorCalcOperatorForms(s);
  s = unwrapLatexSymbolTokens(s);
  s = normalizeLatexAliases(s);
  s = s.replace(/\\laplacian/gi, "\\operatorname{laplacian}");
  s = s.replace(/\\Delta\s*\(/g, "\\operatorname{laplacian}(");
  s = s.replace(/\\Delta\s*\{/g, "\\operatorname{laplacian}{");
  s = s.replace(/\\Delta\s+/g, "\\operatorname{laplacian} ");
  s = s.replace(/\\nabla\s*\^\s*\{?\s*2\s*\}?\s*/gi, "\\operatorname{laplacian} ");
  // \nabla\cdot / \nabla\times before plain \nabla → grad.
  s = s.replace(/\\nabla\s*\\cdot\s*/gi, "\\operatorname{div}");
  s = s.replace(/\\nabla\s*\\times\s*/gi, "\\operatorname{curl}");
  s = s.replace(/\\div/gi, "\\operatorname{div}");
  s = s.replace(/\\curl/gi, "\\operatorname{curl}");
  s = normalizeBareVectorCalcOperators(s);
  s = s.replace(/\\operatorname\s*\{\s*grad\s*\}/gi, "\\grad");
  s = s.replace(/\\grad\s*\{/g, "\\operatorname{grad}{");
  s = s.replace(/\\grad\s*\(/g, "\\operatorname{grad}(");
  s = s.replace(/\\grad\s+(?=[A-Za-z(\\{])/g, "\\operatorname{grad} ");
  s = s.replace(/\\nabla\s+/g, "\\operatorname{grad} ");
  s = s.replace(/\\nabla\s*\(/g, "\\operatorname{grad}(");
  s = s.replace(/\\nabla\s*\{/g, "\\operatorname{grad}{");
  return s;
}

export function extractTriple(json: unknown): string[] | null {
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

export function scalarFromUnaryOpJson(json: unknown, op: string): string | null {
  if (!Array.isArray(json)) return null;
  const head = String(json[0]);
  const opLower = op.toLowerCase();
  if (head === "Multiply" && String(json[1]).toLowerCase() === opLower && json[2] != null) {
    return ce.box(json[2] as never).latex;
  }
  if (head.toLowerCase() === opLower && json[1] != null) {
    return ce.box(json[1] as never).latex;
  }
  return null;
}

export function scalarFromGradJson(json: unknown): string | null {
  if (!Array.isArray(json)) return null;
  const head = String(json[0]);
  const inner = scalarFromUnaryOpJson(json, "grad");
  if (inner) return inner;
  if ((head === "grad" || head === "Gradient" || head === "nabla") && json[1] != null) {
    return ce.box(json[1] as never).latex;
  }
  return null;
}

/** True when LaTeX denotes the Cartesian position vector (x,y,z). */
export function isPositionVectorLatex(latex: string): boolean {
  const t = String(latex ?? "")
    .replace(/\\left\s*/g, "")
    .replace(/\\right\s*/g, "")
    .trim();
  return /^(?:\\(?:mathbf|mathrm|mathit|bf)\s*\{\s*r\s*\}|r)$/i.test(t);
}

export const POSITION_VECTOR_TRIPLE: [string, string, string] = ["x", "y", "z"];

export type DivergenceMatch =
  | { mode: "triple"; parts: [string, string, string]; scale?: number }
  | { mode: "laplacian"; inner: string; scale?: number }
  | { mode: "constant"; value: number; scale?: number };

export type TupleBinaryMatch = {
  left: [string, string, string];
  right: [string, string, string];
};

function extractDivArgumentLatex(src: string): string | null {
  const m = src.match(
    /(?:\\(?:operatorname\s*\{\s*div\s*\}|div)|(?<![A-Za-z\\])div)\s*(.*)$/is,
  );
  if (!m?.[1]) return null;
  let tail = m[1].trim();
  if (!tail) return null;
  const wrapped = extractUnaryTail(tail);
  if (wrapped != null) return wrapped;
  if (tail.endsWith(")") && !tail.includes("(")) {
    tail = tail.slice(0, -1).trim();
  }
  return tail || null;
}

function applyDivergenceScale(match: DivergenceMatch, scale: number): DivergenceMatch {
  if (scale === 1) return match;
  return { ...match, scale };
}

/** Parse `(Fx,Fy,Fz)` into three LaTeX component strings. */
export function parseTupleLatexComponents(src: string): [string, string, string] | null {
  let s = String(src ?? "").trim();
  if (s.startsWith("(") && s.endsWith(")")) s = s.slice(1, -1).trim();
  if (!s) return null;

  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i <= s.length; i++) {
    const c = s[i];
    if (c === "(" || c === "{") depth++;
    else if (c === ")" || c === "}") depth--;
    else if ((c === "," || i === s.length) && depth === 0) {
      const chunk = s.slice(start, c === "," ? i : i).trim();
      if (chunk) parts.push(chunk);
      start = i + 1;
    }
  }
  if (parts.length !== 3 || parts.some((p) => !p)) return null;
  return parts as [string, string, string];
}

function splitTopLevelInfix(src: string, op: "\\times" | "\\cdot"): [string, string] | null {
  let depth = 0;
  for (let i = 0; i < src.length; i++) {
    const c = src[i]!;
    if (c === "(" || c === "{") depth++;
    else if (c === ")" || c === "}") depth--;
    else if (depth === 0 && src.startsWith(op, i)) {
      const left = src.slice(0, i).trim();
      const right = src.slice(i + op.length).trim();
      if (left && right) return [left, right];
      return null;
    }
  }
  return null;
}

function parseTupleBinaryOp(src: string, op: "\\times" | "\\cdot"): TupleBinaryMatch | null {
  const split = splitTopLevelInfix(src, op);
  if (!split) return null;
  const left = parseTupleLatexComponents(split[0]);
  const right = parseTupleLatexComponents(split[1]);
  if (!left || !right) return null;
  return { left, right };
}

/** Cross product of two 3-tuples written `(a,b,c)\\times(d,e,f)`. */
export function parseTupleCrossMatch(src: string): TupleBinaryMatch | null {
  return parseTupleBinaryOp(src, "\\times");
}

/** Dot product of two 3-tuples written `(a,b,c)\\cdot(d,e,f)`. */
export function parseTupleDotMatch(src: string): TupleBinaryMatch | null {
  return parseTupleBinaryOp(src, "\\cdot");
}

/** Inner scalar of `\grad(f)`, `\nabla(f)`, etc. Requires balanced parens/brackets. */
export function extractGradOperand(part: string): string | null {
  const trimmed = String(part ?? "").trim();
  const m = trimmed.match(
    /^\\(?:operatorname\s*\{\s*grad\s*\}|grad|nabla)\s*(?:\\left)?[\{\(]\s*([\s\S]+?)\s*(?:\\right)?[\}\)]\s*$/i,
  );
  if (m?.[1]) return m[1].trim();
  const m2 = trimmed.match(
    /^\\(?:operatorname\s*\{\s*grad\s*\}|grad|nabla)\s+(?![_\^\\])([\s\S]+)$/i,
  );
  return m2?.[1]?.trim() ?? null;
}

/** Scalar `\grad f \cdot \grad g` (top-level `\cdot` between two gradients). */
export function parseGradDotMatch(src: string): { left: string; right: string } | null {
  const split = splitTopLevelInfix(src, "\\cdot");
  if (!split) return null;
  const left = extractGradOperand(split[0]);
  const right = extractGradOperand(split[1]);
  if (!left || !right) return null;
  return { left, right };
}

/** Numeric scale × 3-tuple from CE `Multiply` MathJSON. */
export function extractScaledTriple(
  json: unknown,
): { scale: number; parts: [string, string, string] } | null {
  if (!Array.isArray(json)) return null;
  if (String(json[0]) === "Multiply") {
    let scale = 1;
    let parts: string[] | null = null;
    for (const f of json.slice(1)) {
      const triple = extractTriple(f);
      if (triple?.length === 3) {
        parts = triple;
        continue;
      }
      const n = numericExponent(f);
      if (n != null) scale *= n;
    }
    if (parts) return { scale, parts: parts as [string, string, string] };
    return null;
  }
  const triple = extractTriple(json);
  if (triple?.length === 3) return { scale: 1, parts: triple as [string, string, string] };
  return null;
}

function numericExponent(json: unknown): number | null {
  if (typeof json === "number" && Number.isFinite(json)) return json;
  if (typeof json === "bigint") return Number(json);
  if (typeof json === "string") {
    const n = Number(json);
    return Number.isFinite(n) ? n : null;
  }
  if (Array.isArray(json)) {
    const head = String(json[0]);
    if (head === "Negate" && json[1] != null) {
      const n = numericExponent(json[1]);
      return n == null ? null : -n;
    }
    if (head === "Rational" && json[1] != null && json[2] != null) {
      const a = numericExponent(json[1]);
      const b = numericExponent(json[2]);
      return a != null && b != null && b !== 0 ? a / b : null;
    }
  }
  try {
    const n = Number(ce.box(json as never).value);
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

function parseDivPoweredForm(json: unknown): DivergenceMatch | null {
  if (!Array.isArray(json) || String(json[0]) !== "Power") return null;
  const base = json[1];
  const exp = numericExponent(json[2]);
  if (exp == null) return null;
  if (!Array.isArray(base) || String(base[0]) !== "Multiply") return null;
  if (String(base[1]).toLowerCase() !== "div" || base[2] == null) return null;
  const innerLatex = ce.box(base[2] as never).latex;
  if (!isPositionVectorLatex(innerLatex)) return null;
  return { mode: "constant", value: Math.pow(3, exp) };
}

function parseDivMultiplyForm(json: unknown): DivergenceMatch | null {
  if (!Array.isArray(json) || String(json[0]) !== "Multiply") return null;
  const factors = json.slice(1);
  let scale = 1;
  let innerJson: unknown = null;
  let pastDiv = false;
  for (const f of factors) {
    if (typeof f === "string" && f.toLowerCase() === "div") {
      pastDiv = true;
      continue;
    }
    if (!pastDiv) {
      const n = numericExponent(f);
      if (n != null) {
        scale *= n;
        continue;
      }
    }
    if (pastDiv && innerJson == null) {
      innerJson = f;
      continue;
    }
    if (pastDiv && innerJson != null) {
      const n = numericExponent(f);
      if (n != null) scale *= n;
    }
  }
  if (!pastDiv || innerJson == null) return null;
  const innerLatex = ce.box(innerJson as never).latex;
  const match = divergenceMatchFromInner(innerLatex, innerJson);
  return match ? applyDivergenceScale(match, scale) : null;
}

function divergenceMatchFromInner(innerLatex: string, innerJson: unknown): DivergenceMatch | null {
  const trimmed = innerLatex.trim();
  if (!trimmed) return null;
  if (isPositionVectorLatex(trimmed)) {
    return { mode: "triple", parts: [...POSITION_VECTOR_TRIPLE] };
  }

  const gradInner = scalarFromGradJson(innerJson);
  if (gradInner) return { mode: "laplacian", inner: gradInner.trim() };

  const triple = extractTriple(innerJson);
  if (triple?.length === 3) {
    return { mode: "triple", parts: triple as [string, string, string] };
  }

  // Bare scalar f: treat ∇·f as ∇·(∇f) = ∇²f (common shorthand when f is a scalar potential).
  return { mode: "laplacian", inner: trimmed };
}

/** Detect divergence / div LaTeX and map to vector components or an inner Laplacian. */
export function parseDivergenceMatch(src: string, json: unknown): DivergenceMatch | null {
  const powered = parseDivPoweredForm(json);
  if (powered) return powered;

  const juxtaposed = parseDivMultiplyForm(json);
  if (juxtaposed) return juxtaposed;

  if (Array.isArray(json) && String(json[0]).toLowerCase() === "div") {
    if (json.length >= 4 && json[1] != null && json[2] != null && json[3] != null) {
      const parts = [json[1], json[2], json[3]].map((p) => ce.box(p as never).latex.trim());
      if (parts.every(Boolean)) {
        return { mode: "triple", parts: parts as [string, string, string] };
      }
    }
    if (json[1] != null) {
      const innerLatex = ce.box(json[1] as never).latex;
      const match = divergenceMatchFromInner(innerLatex, json[1]);
      if (match) return match;
    }
  }

  const fromJson = tripleFromUnaryOpJson(json, "div");
  if (fromJson?.length === 3) {
    return { mode: "triple", parts: fromJson as [string, string, string] };
  }

  if (/\\(?:operatorname\s*\{\s*div\s*\}|\\div\b|(?<![A-Za-z\\])div\s*[({])/i.test(src)) {
    const innerLatex = extractDivArgumentLatex(src);
    if (innerLatex) {
      try {
        const inner = ce.parse(innerLatex);
        const j = inner?.json ?? (typeof inner?.toJSON === "function" ? inner.toJSON() : null);
        const match = divergenceMatchFromInner(innerLatex, j);
        if (match) return match;
      } catch {
        return divergenceMatchFromInner(innerLatex, null);
      }
    }
  }

  return null;
}

export function tripleFromUnaryOpJson(json: unknown, op: string): string[] | null {
  if (!Array.isArray(json)) return null;
  const head = String(json[0]);
  const opLower = op.toLowerCase();
  if (head === "Multiply" && String(json[1]).toLowerCase() === opLower && json[2] != null) {
    return extractTriple(json[2]);
  }
  if (head.toLowerCase() === opLower && json[1] != null) {
    return extractTriple(json[1]);
  }
  return null;
}

export function tripleFromOpLatex(src: string, opPattern: RegExp, innerLatex: string): string[] | null {
  try {
    const inner = ce.parse(innerLatex.trim());
    const j = inner?.json ?? (typeof inner?.toJSON === "function" ? inner.toJSON() : null);
    const triple = extractTriple(j);
    if (triple?.length === 3) return triple;
  } catch {
    /* fall through */
  }
  void opPattern;
  void src;
  return null;
}

export interface PartialMatch {
  axis: ChebAxis;
  inner: string;
}

/** Detect \\operatorname{partial_x}{f} and equivalent normalized forms. */
export function looksLikePartial(src: string, json: unknown): PartialMatch | null {
  for (let axis = 0 as ChebAxis; axis <= 2; axis = (axis + 1) as ChebAxis) {
    const op = partialOpName(axis);
    const fromJson = scalarFromUnaryOpJson(json, op);
    if (fromJson) return { axis, inner: fromJson.trim() };

    const inner = extractOperatornameArg(src, op);
    if (inner) return { axis, inner };
  }

  // CE Derivative / PartialDerivative fallback
  if (Array.isArray(json)) {
    const head = String(json[0]);
    if (head === "Derivative" || head === "PartialDerivative") {
      const expr = json[1];
      const wrt = json[2];
      const varName =
        typeof wrt === "string"
          ? wrt
          : Array.isArray(wrt) && wrt[0] === "Symbol"
            ? String(wrt[1])
            : null;
      const axis = varName ? axisFromVar(varName) : null;
      if (axis != null && expr != null) {
        const inner = ce.box(expr as never).latex;
        if (inner) return { axis, inner: inner.trim() };
      }
    }
  }

  return null;
}
