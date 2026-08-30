import { compileExpr, classifyExpr, PRESETS, compileParamLatex } from "../math/fit.js";
import { compileVectorExpr } from "../math/fitVector.js";
import {
  syncParamsFromDefinitions,
  applyParamSeed,
  getParamValues,
  getParam,
  recompileAllParams,
  evalParamEquations,
  ensureParamAnimationFromExprs,
} from "../model/params.js";
import { SymbolRegistry, type SymbolEntry } from "../model/symbols.js";
import {
  listExpressions,
  setExpressions,
  clearExpressions,
  updateExprSilent,
  hexToRgb01,
  inferLayerRole,
  replaceExprWarnings,
  removeExprSilent,
  resolveExprGradient,
} from "../model/expressions.js";
import { closeSettingsDialog, els } from "./dom.js";
import { state } from "./state.js";
import {
  collectPendingParamsForExpr,
  pendingParamErrorMessage,
  ensureParamExprRows,
} from "./pendingParams.js";
import type { ClassifiedExpr, CompileAllResult, CompileLayerResult, ExprItem } from "../types/models.js";

function symbolEntryFromClassified(item: ExprItem, classified: ClassifiedExpr): SymbolEntry | null {
  if (classified.kind === "parameter" && classified.paramName) {
    return {
      kind: "parameter",
      name: classified.paramName,
      rhsLatex: classified.compileLatex,
      latex: item.latex,
      exprId: item.id,
    };
  }
  if (classified.kind === "alias" && classified.aliasName) {
    return {
      kind: "alias",
      name: classified.aliasName,
      rhsLatex: classified.compileLatex,
      latex: item.latex,
      exprId: item.id,
    };
  }
  if (classified.kind === "funcdef" && classified.funcName) {
    return {
      kind: "funcdef",
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
  const warnings: [string, string][] = [];
  for (const item of items) {
    let classified: ClassifiedExpr;
    try {
      classified = classifyExpr(item.latex);
    } catch (e) {
      warnings.push([
        item.id,
        e instanceof Error ? e.message : "Invalid expression",
      ]);
      continue;
    }
    const entry = symbolEntryFromClassified(item, classified);
    if (!entry) continue;
    const err = registry.tryAdd(entry);
    if (err) warnings.push([item.id, err]);
  }
  return { registry, warnings };
}

export function fmtParamNum(v: number) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1000 || a < 0.01)) return v.toPrecision(3);
  return String(Math.round(v * 1000) / 1000);
}

/** Re-exported for presets and tests. */
export { ensureParamExprRows, createParamRows } from "./pendingParams.js";

function listNonemptyExprs() {
  return listExpressions().filter((e) => String(e.latex || "").trim());
}

/** Names referenced by field free-symbols or parameter RHS deps. */
export function collectParamReferences() {
  const allItems = listNonemptyExprs();
  const enabledItems = allItems.filter((e) => e.enabled);
  const { registry } = buildSymbolRegistry(allItems);
  const refs = new Set<string>();
  for (const item of enabledItems) {
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind === "parameter") {
        const compiled = compileParamLatex(item.latex, classified.paramName!);
        for (const p of compiled.freeParams) refs.add(p);
        continue;
      }
      const fieldLatex = classified.compileLatex;
      const role = inferLayerRole(classified.kind, fieldLatex, registry);
      const compiled =
        role === "flow"
          ? compileVectorExpr(fieldLatex, registry)
          : compileExpr(fieldLatex, registry);
      for (const p of compiled.freeParams) refs.add(p);
    } catch {
      /* ignore */
    }
  }
  return refs;
}

/** Drop ephemeral auto-param rows that are no longer referenced (typing undo). */
export function pruneUnusedAutoParams() {
  const refs = collectParamReferences();
  let removed = false;
  for (const item of listExpressions()) {
    if (!item.autoParam) continue;
    let name: string | null = null;
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind === "parameter") name = classified.paramName ?? null;
    } catch {
      continue;
    }
    if (!name || refs.has(name)) continue;
    if (removeExprSilent(item.id)) removed = true;
  }
  return removed;
}

interface CompileOpts {
  rebuildUi?: boolean;
  _afterEnsure?: boolean;
}

/**
 * Compile all expressions: parameter rows feed shared values; field rows become layers.
 */
export function compileAllExprs(opts: CompileOpts = {}): CompileAllResult {
  const rebuildUi = opts.rebuildUi !== false;
  const allItems = listNonemptyExprs();
  const enabledItems = allItems.filter((e) => e.enabled);

  // Declarations stay in scope when hidden — visibility only skips rendering layers.
  const { registry, warnings: registryWarnings } = buildSymbolRegistry(allItems);
  const paramRows: { item: ExprItem; name: string }[] = [];
  const layers: CompileLayerResult[] = [];
  const freeSet = new Set<string>();
  const definedParams = new Set<string>();
  const warnings: [string, string][] = [...registryWarnings];
  replaceExprWarnings([]);

  for (const item of allItems) {
    let classified;
    try {
      classified = classifyExpr(item.latex);
    } catch (e) {
      warnings.push([
        item.id,
        e instanceof Error ? e.message : "Invalid expression",
      ]);
      continue;
    }
    if (classified.kind === "parameter") {
      const name = classified.paramName;
      if (!name) continue;
      const entry = registry.getParam(name);
      if (!entry || entry.exprId !== item.id) continue;
      definedParams.add(name);
      paramRows.push({ item, name });
      const pendingMsg = pendingParamErrorMessage(collectPendingParamsForExpr(item));
      if (pendingMsg) warnings.push([item.id, pendingMsg]);
      continue;
    }
    if (!item.enabled) continue;

    const pendingMsg = pendingParamErrorMessage(collectPendingParamsForExpr(item));
    if (pendingMsg) {
      warnings.push([item.id, pendingMsg]);
      continue;
    }

    const fieldLatex = classified.compileLatex;
    const role = inferLayerRole(classified.kind, fieldLatex, registry);

    if (role === "flow") {
      try {
        const rowWarnings: string[] = [];
        const vectorCompiled = compileVectorExpr(fieldLatex, registry, rowWarnings);
        for (const w of rowWarnings) warnings.push([item.id, w]);
        for (const p of vectorCompiled.freeParams) freeSet.add(p);
        if (!vectorCompiled.usesSpace) continue;
        layers.push({
          item,
          vectorCompiled,
          role: "flow",
          vectorFn: vectorCompiled.bind(getParamValues()),
        });
      } catch (e) {
        warnings.push([
          item.id,
          e instanceof Error ? e.message : "Invalid flow field",
        ]);
      }
      continue;
    }

    try {
      const rowWarnings: string[] = [];
      const compiled = compileExpr(fieldLatex, registry, rowWarnings);
      for (const w of rowWarnings) warnings.push([item.id, w]);
      for (const p of compiled.freeParams) freeSet.add(p);
      if (!compiled.usesSpace || compiled.shade === "none") continue;
      layers.push({
        item,
        compiled,
        role,
        fn: compiled.bind(getParamValues()),
      });
    } catch (e) {
      warnings.push([
        item.id,
        e instanceof Error ? e.message : "Invalid expression",
      ]);
    }
  }

  replaceExprWarnings(warnings);

  const defs = paramRows.map(({ item, name }) => ({
    name,
    latex: item.latex,
    exprId: item.id,
    min: item.sliderMin,
    max: item.sliderMax,
    speed: item.sliderSpeed,
    animating: item.sliderAnimating,
    phase: item.sliderPhase,
    animMode: item.sliderAnimMode,
  }));

  syncParamsFromDefinitions(defs, state.pendingParamSeed);

  recompileAllParams();

  const pruned = !opts._afterEnsure && pruneUnusedAutoParams();
  if (pruned && !opts._afterEnsure) {
    return compileAllExprs({ ...opts, _afterEnsure: true });
  }

  if (Object.keys(state.pendingParamSeed).length) {
    applyParamSeed(state.pendingParamSeed);
    for (const { item, name } of paramRows) {
      const p = getParam(name);
      if (!p) continue;
      updateExprSilent(item.id, {
        latex: p.latex,
        sliderMin: p.min,
        sliderMax: p.max,
        sliderSpeed: p.speed,
        sliderAnimating: p.animating,
        sliderPhase: p.phase,
        sliderAnimMode: p.animMode,
      });
    }
    state.pendingParamSeed = {};
  }

  evalParamEquations();
  const params = getParamValues();
  for (const L of layers) {
    if (L.role === "flow" && L.vectorCompiled) {
      L.vectorFn = L.vectorCompiled.bind(params);
    } else if (L.compiled) {
      L.fn = L.compiled.bind(params);
    }
  }

  if (rebuildUi) {
    if (!state.exprListApi?.syncParamChrome?.()) {
      state.exprListApi?.render();
    }
  }

  const nIso = layers.filter((L) => L.role === "isosurface").length;
  const nCloud = layers.filter((L) => L.role === "cloud").length;
  const nFlow = layers.filter((L) => L.role === "flow").length;
  state.lastExprMeta = {
    kind: nIso && nCloud ? "mixed" : nIso ? "constraint" : nFlow ? "bare" : "bare",
    shade: nIso && !nCloud && !nFlow ? "iso" : "volume",
    isoLevel: 0,
    label: `${nCloud} cloud · ${nIso} isosurface · ${nFlow} flow`,
  };

  return {
    freeParams: [...freeSet].sort(),
    layers,
    warnings: warnings.map(([, msg]) => msg),
  };
}

export function applyPreset(key: string) {
  const p = PRESETS[key] ?? PRESETS.sincos;
  els.preset.value = key in PRESETS ? key : "sincos";
  closeSettingsDialog();
  state.pendingParamSeed = p.params ?? {};
  if (Array.isArray(p.expressions) && p.expressions.length) {
    setExpressions(p.expressions);
  } else {
    setExpressions([{ latex: p.latex ?? "" }]);
  }
  const seedNames = Object.keys(p.params ?? {});
  if (seedNames.length) ensureParamExprRows(seedNames);
  // Insert any auto-param rows before a single UI rebuild.
  try {
    compileAllExprs({ rebuildUi: false });
  } catch {
    /* uploadFit / syncExprCompileState will surface errors */
  }
  ensureParamAnimationFromExprs();
  state.exprListApi?.render();
}

/** Clear all expression rows and recompile (empty scene). */
export function clearAllExprs() {
  clearExpressions();
  state.pendingParamSeed = {};
  replaceExprWarnings([]);
  try {
    compileAllExprs({ rebuildUi: false });
  } catch {
    /* syncExprCompileState will surface errors */
  }
  state.exprListApi?.render();
}

export function layerRgbFromItem(item: ExprItem) {
  const grad = resolveExprGradient(item);
  const colors = grad.colors.map((hex) => hexToRgb01(hex));
  return {
    color: colors[0],
    color2: colors[colors.length - 1],
    colors,
  };
}

export function initCompile() {
  applyPreset("sincos");
  if (!listExpressions().length) {
    const p = PRESETS.sincos;
    if (Array.isArray(p.expressions) && p.expressions.length) {
      setExpressions(p.expressions);
    } else {
      setExpressions([{ latex: p.latex ?? "" }]);
    }
  }
}
