import {
  decodeCompactFragment,
  FRAGMENT_RE,
} from "../../web/src/app/persistence/exprShareCodec.js";
import type { ExprItem } from "../../web/src/types/models.js";

export function normalizeSharePayload(input: string): string {
  let s = input.trim();
  if (s.startsWith("#")) s = s.slice(1);
  if (s.startsWith("e=")) s = s.slice(2);
  return s;
}

export function fragmentFromSharePayload(payload: string): string {
  return `e=${normalizeSharePayload(payload)}`;
}

export function validateSharePayload(payload: string): string {
  const body = normalizeSharePayload(payload);
  if (!body || !FRAGMENT_RE.test(`e=${body}`)) {
    throw new Error("Invalid share payload");
  }
  return body;
}

export async function decodeSharePayload(payload: string): Promise<Partial<ExprItem>[]> {
  const body = validateSharePayload(payload);
  const decoded = await decodeCompactFragment(`e=${body}`);
  if (!decoded?.rows?.length) throw new Error("Could not decode share payload");
  return decoded.rows;
}

export function sharePanelsFromRows(rows: Partial<ExprItem>[], max = 3) {
  // Parameter rows (e.g. `t=0.35`) aren't graphed themselves, but a visual
  // row can reference them (animated/parametric expressions are common) —
  // each panel is rendered as an isolated single-expression scene, so it
  // needs these carried along or the compiler sees `t` as undefined.
  const paramRows = rows.filter((row) => row.autoParam && String(row.latex ?? "").trim());

  return rows
    .filter((row) => {
      const latex = String(row.latex ?? "").trim();
      if (!latex) return false;
      // autoParam is the classifier-driven signal for "this is a slider
      // parameter, not a graphed surface" (e.g. `a=1`). A plain name=…
      // regex here would also catch legitimate graphed expressions like
      // `z=f(x,y)` or aliases (`T=x^2+y^2`) that still render a layer.
      if (row.autoParam) return false;
      return true;
    })
    .slice(0, max)
    .map((row, index) => ({
      latex: String(row.latex ?? ""),
      palette: index,
      label: latexToPlainLabel(String(row.latex ?? "")),
      paramRows,
    }));
}

function latexToPlainLabel(latex: string): string {
  return latex
    .replace(/\\left/g, "")
    .replace(/\\right/g, "")
    .replace(/\\cos/g, "cos")
    .replace(/\\sin/g, "sin")
    .replace(/\\abs/g, "|")
    .replace(/\\pi/g, "π")
    .replace(/\^{([^}]+)}/g, "^$1")
    .replace(/_{([^}]+)}/g, "_$1")
    .replace(/\s+/g, " ")
    .trim();
}

export function siteUrl(): string {
  const url = process.env.SITE_URL ?? process.env.VERCEL_URL;
  if (!url) return "http://127.0.0.1:4173";
  if (url.startsWith("http")) return url.replace(/\/$/, "");
  return `https://${url.replace(/\/$/, "")}`;
}
