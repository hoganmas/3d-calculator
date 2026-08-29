/** Shared vector-calculus LaTeX normalization and MathJSON tuple extraction. */

import { ComputeEngine } from "@cortex-js/compute-engine";

const ce = new ComputeEngine();

export function normalizeCalcLatex(latex: string): string {
  let s = String(latex ?? "").trim();
  s = s.replace(/\\left\s*/g, "");
  s = s.replace(/\\right\s*/g, "");
  // \del is an alternate spelling of \nabla — normalize before operator rules.
  s = s.replace(/\\del/gi, "\\nabla");
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
