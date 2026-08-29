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
