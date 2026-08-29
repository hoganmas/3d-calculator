/**
 * Structured expression / parameter error reporting for UI and MCP.
 */
import { listExpressions, listExprWarnings } from "../model/expressions.js";
import { listParamNames, getParam } from "../model/params.js";

export type ExpressionErrorKind = "expression" | "parameter" | "global";

export interface ExpressionErrorItem {
  kind: ExpressionErrorKind;
  id?: string;
  row?: number;
  latex?: string;
  name?: string;
  message: string;
}

export interface ExpressionErrorReport {
  compileOk: boolean;
  globalError: string | null;
  errors: ExpressionErrorItem[];
  expressionCount: number;
  errorCount: number;
}

function rowNumberForExprId(id: string): number | undefined {
  const rows = listExpressions().filter((e) => String(e.latex || "").trim());
  const idx = rows.findIndex((e) => e.id === id);
  return idx >= 0 ? idx + 1 : undefined;
}

/** Gather per-row warnings and param errors (call after compile sync). */
export function collectExpressionErrors(
  compileOk: boolean,
  globalError: string | null = null,
): ExpressionErrorReport {
  const errors: ExpressionErrorItem[] = [];

  if (globalError?.trim()) {
    errors.push({ kind: "global", message: globalError.trim() });
  }

  for (const row of listExprWarnings()) {
    errors.push({
      kind: "expression",
      id: row.id,
      row: rowNumberForExprId(row.id),
      latex: row.latex,
      message: row.message,
    });
  }

  for (const name of listParamNames()) {
    const p = getParam(name);
    if (!p?.error) continue;
    errors.push({
      kind: "parameter",
      id: p.exprId ?? undefined,
      name,
      row: p.exprId ? rowNumberForExprId(p.exprId) : undefined,
      latex: p.latex,
      message: p.error,
    });
  }

  return {
    compileOk,
    globalError: globalError?.trim() || null,
    errors,
    expressionCount: listExpressions().filter((e) => String(e.latex || "").trim()).length,
    errorCount: errors.length,
  };
}

export function formatExpressionErrors(report: ExpressionErrorReport): string {
  if (!report.errors.length) return "";
  const lines: string[] = [];
  for (const e of report.errors) {
    if (e.kind === "global") {
      lines.push(e.message);
      continue;
    }
    if (e.kind === "parameter") {
      const where = e.row != null ? `row ${e.row}` : `param ${e.name}`;
      lines.push(`${where} · ${e.name}: ${e.message}`);
      continue;
    }
    const where = e.row != null ? `row ${e.row}` : e.id ?? "expression";
    lines.push(`${where}: ${e.message}`);
  }
  return lines.join("\n");
}

/** Console-friendly multi-line dump. */
export function logExpressionErrors(report: ExpressionErrorReport) {
  if (!report.errors.length) {
    console.info("[expressions] no errors");
    return;
  }
  console.group(`[expressions] ${report.errorCount} issue(s)`);
  for (const e of report.errors) {
    const prefix =
      e.kind === "global"
        ? "global"
        : e.kind === "parameter"
          ? `param ${e.name}`
          : `row ${e.row ?? e.id}`;
    console.warn(`${prefix}: ${e.message}`, e.latex ? { latex: e.latex } : "");
  }
  console.groupEnd();
}
