/** Shared vector-calculus LaTeX normalization and MathJSON tuple extraction. */

import { ComputeEngine } from "@cortex-js/compute-engine";

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

export function normalizeCalcLatex(latex: string): string {
  let s = String(latex ?? "").trim();
  s = s.replace(/\\left\s*/g, "");
  s = s.replace(/\\right\s*/g, "");
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
