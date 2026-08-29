import {
  classifyExpr,
  compileExpr,
  compileParamLatex,
} from "../math/fit.js";
import { compileVectorExpr } from "../math/fitVector.js";
import {
  listExpressions,
  resolveExprRole,
  insertExprAt,
} from "../model/expressions.js";
import { formatParamLatexValue } from "../math/fit.js";
import { SymbolRegistry } from "../model/symbols.js";
import type { ClassifiedExpr, ExprItem } from "../types/models.js";
import { state } from "./state.js";

function listNonemptyExprs() {
  return listExpressions().filter((e) => String(e.latex || "").trim());
}

function symbolEntryFromClassified(item: ExprItem, classified: ClassifiedExpr) {
  if (classified.kind === "parameter" && classified.paramName) {
    return {
      kind: "parameter" as const,
      name: classified.paramName,
      rhsLatex: classified.compileLatex,
      latex: item.latex,
      exprId: item.id,
    };
  }
  if (classified.kind === "alias" && classified.aliasName) {
    return {
      kind: "alias" as const,
      name: classified.aliasName,
      rhsLatex: classified.compileLatex,
      latex: item.latex,
      exprId: item.id,
    };
  }
  if (classified.kind === "funcdef" && classified.funcName) {
    return {
      kind: "funcdef" as const,
      name: classified.funcName,
      rhsLatex: classified.compileLatex,
      latex: item.latex,
      exprId: item.id,
      funcArgs: classified.funcArgs ?? [],
    };
  }
  return null;
}

function buildSymbolRegistry(items: ExprItem[]) {
  const registry = new SymbolRegistry();
  for (const item of items) {
    try {
      const classified = classifyExpr(item.latex);
      const entry = symbolEntryFromClassified(item, classified);
      if (!entry) continue;
      registry.tryAdd(entry);
    } catch {
      /* skip invalid rows */
    }
  }
  return registry;
}

/** Parameter names that have an owning `name=…` declaration row. */
export function collectDefinedParamNames(items = listNonemptyExprs()): Set<string> {
  const registry = buildSymbolRegistry(items);
  const defined = new Set<string>();
  for (const item of items) {
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind !== "parameter" || !classified.paramName) continue;
      const entry = registry.getParam(classified.paramName);
      if (entry && entry.exprId === item.id) defined.add(classified.paramName);
    } catch {
      /* skip */
    }
  }
  return defined;
}

/** Free parameter symbols referenced by one row that lack a declaration. */
export function collectPendingParamsForExpr(item: ExprItem): string[] {
  const latex = String(item.latex || "").trim();
  if (!latex || !item.enabled) return [];

  const allItems = listNonemptyExprs();
  const defined = collectDefinedParamNames(allItems);
  const registry = buildSymbolRegistry(allItems);
  const pending = new Set<string>();

  try {
    const classified = classifyExpr(latex);
    if (classified.kind === "parameter" && classified.paramName) {
      const compiled = compileParamLatex(latex, classified.paramName);
      for (const name of compiled.freeParams) {
        if (!defined.has(name)) pending.add(name);
      }
    } else {
      const fieldLatex = classified.compileLatex;
      const role = resolveExprRole(item.role, classified.kind, fieldLatex, registry);
      const compiled =
        role === "flow"
          ? compileVectorExpr(fieldLatex, registry)
          : compileExpr(fieldLatex, registry);
      for (const name of compiled.freeParams) {
        if (!defined.has(name)) pending.add(name);
      }
    }
  } catch {
    return [];
  }

  return [...pending].sort();
}

export function formatPendingParamLabel(names: string[]): string {
  if (!names.length) return "";
  if (names.length === 1) return names[0]!;
  if (names.length <= 3) return names.join(", ");
  return `${names.slice(0, 2).join(", ")} +${names.length - 2}`;
}

export function pendingParamErrorMessage(names: string[]): string | null {
  if (!names.length) return null;
  const label = formatPendingParamLabel(names);
  const noun = names.length === 1 ? "parameter" : "parameters";
  return `Undefined ${noun}: ${label}. Press Tab or click below to create.`;
}

/** Insert missing `name=value` expression rows at the bottom (sorted) without UI notify. */
export function ensureParamExprRows(names: string[]) {
  const missing = [...new Set(names)].filter(Boolean).sort();
  if (!missing.length) return false;
  for (const name of missing) {
    const seed = state.pendingParamSeed[name] ?? {};
    const value = Number.isFinite(seed.value) ? seed.value! : 1;
    insertExprAt(listExpressions().length, {
      latex: `${name}=${formatParamLatexValue(value)}`,
      sliderMin: seed.min,
      sliderMax: seed.max,
      sliderSpeed: seed.speed,
      sliderAnimating: !!(seed.animate ?? seed.animating),
      sliderPhase: seed.phase,
      sliderAnimMode: seed.animMode === "loop" ? "loop" : "pingpong",
      autoParam: true,
    });
  }
  return true;
}

/** Insert `name=value` rows for undefined parameters. Returns true if any row was added. */
export function createParamRows(names: string[]): boolean {
  const defined = collectDefinedParamNames();
  const toCreate = [...new Set(names)].filter((n) => n && !defined.has(n)).sort();
  if (!toCreate.length) return false;
  ensureParamExprRows(toCreate);
  return true;
}
