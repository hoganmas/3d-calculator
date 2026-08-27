import { compileExpr, classifyExpr, PRESETS, formatParamLatexValue, compileParamLatex } from "../math/fit.js";
import { compileVectorExpr } from "../math/fitVector.js";
import {
  syncParamsFromDefinitions,
  applyParamSeed,
  getParamValues,
  getParam,
  recompileAllParams,
  evalParamEquations,
} from "../model/params.js";
import {
  listExpressions,
  setExpressions,
  updateExprSilent,
  hexToRgb01,
  resolveExprRole,
  replaceExprWarnings,
  insertExprAt,
  removeExprSilent,
  resolveExprGradient,
} from "../model/expressions.js";
import { els } from "./dom.js";
import { state } from "./state.js";
import type { CompileAllResult, CompileLayerResult, ExprItem } from "../types/models.js";

export function fmtParamNum(v: number) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1000 || a < 0.01)) return v.toPrecision(3);
  return String(Math.round(v * 1000) / 1000);
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

/** Names referenced by field free-symbols or parameter RHS deps. */
export function collectParamReferences() {
  const refs = new Set<string>();
  for (const item of listExpressions()) {
    if (!item.enabled || !String(item.latex || "").trim()) continue;
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind === "parameter") {
        const compiled = compileParamLatex(item.latex, classified.paramName!);
        for (const p of compiled.freeParams) refs.add(p);
      } else {
        const role = resolveExprRole(item.role, classified.kind, item.latex);
        const compiled =
          role === "flow" ? compileVectorExpr(item.latex) : compileExpr(item.latex);
        for (const p of compiled.freeParams) refs.add(p);
      }
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

export function shouldDeferAutoParamRows() {
  const ae = document.activeElement;
  if (!ae) return false;
  const mf =
    (ae.closest && ae.closest("math-field")) ||
    (ae.tagName === "MATH-FIELD" ? ae : null);
  if (!mf) return false;
  const row = mf.closest?.(".expr-row");
  if (!row || row.classList.contains("is-param-def")) return false;
  return true;
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
  const items = listExpressions().filter((e) => e.enabled && String(e.latex || "").trim());

  const paramRows: { item: ExprItem; name: string }[] = [];
  const layers: CompileLayerResult[] = [];
  const freeSet = new Set<string>();
  const definedParams = new Set<string>();
  const warnings: [string, string][] = [];
  replaceExprWarnings([]);

  for (const item of items) {
    let classified;
    try {
      classified = classifyExpr(item.latex);
    } catch {
      continue;
    }
    if (classified.kind === "parameter") {
      const name = classified.paramName;
      if (!name) continue;
      if (definedParams.has(name)) {
        warnings.push([item.id, `Variable “${name}” is already declared`]);
        continue;
      }
      definedParams.add(name);
      paramRows.push({ item, name });
      continue;
    }

    const role = resolveExprRole(item.role, classified.kind, item.latex);

    if (role === "flow") {
      try {
        const vectorCompiled = compileVectorExpr(item.latex);
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

    const compiled = compileExpr(item.latex);
    for (const p of compiled.freeParams) freeSet.add(p);
    if (!compiled.usesSpace || compiled.shade === "none") continue;
    layers.push({
      item,
      compiled,
      role,
      fn: compiled.bind(getParamValues()),
    });
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

  const depNames = recompileAllParams();
  const known = new Set(defs.map((d) => d.name));
  const needRows = [
    ...[...freeSet].filter((n) => !definedParams.has(n)),
    ...[...new Set(depNames)].filter((n) => !known.has(n) && !definedParams.has(n)),
  ];

  const deferAuto = shouldDeferAutoParamRows();
  const pruned = !opts._afterEnsure && !deferAuto && pruneUnusedAutoParams();
  const toCreate = deferAuto ? [] : needRows;
  if ((pruned || toCreate.length) && !opts._afterEnsure) {
    if (toCreate.length) ensureParamExprRows(toCreate);
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
  state.pendingParamSeed = p.params ?? {};
  if (Array.isArray(p.expressions) && p.expressions.length) {
    setExpressions(p.expressions);
  } else {
    setExpressions([{ latex: p.latex ?? "" }]);
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
    setExpressions([{ latex: PRESETS.sincos.latex ?? "" }]);
  }
}
