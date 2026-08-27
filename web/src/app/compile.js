import { compileExpr, classifyExpr, PRESETS, formatParamLatexValue, compileParamLatex } from "../math/fit.js";
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

export function fmtParamNum(v) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1000 || a < 0.01)) return v.toPrecision(3);
  return String(Math.round(v * 1000) / 1000);
}

/**
 * Insert missing `name=value` expression rows at the bottom (sorted) without UI notify.
 * @param {string[]} names
 */
export function ensureParamExprRows(names) {
  const missing = [...new Set(names)].filter(Boolean).sort();
  if (!missing.length) return false;
  for (const name of missing) {
    const seed = state.pendingParamSeed[name] ?? {};
    const value = Number.isFinite(seed.value) ? seed.value : 1;
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
  /** @type {Set<string>} */
  const refs = new Set();
  for (const item of listExpressions()) {
    if (!item.enabled || !String(item.latex || "").trim()) continue;
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind === "parameter") {
        const compiled = compileParamLatex(item.latex, classified.paramName);
        for (const p of compiled.freeParams) refs.add(p);
      } else {
        const compiled = compileExpr(item.latex);
        for (const p of compiled.freeParams) refs.add(p);
      }
    } catch {
      /* ignore */
    }
  }
  return refs;
}

/**
 * Drop ephemeral auto-param rows that are no longer referenced (typing undo).
 * Committed rows (after blur) are kept.
 */
export function pruneUnusedAutoParams() {
  const refs = collectParamReferences();
  let removed = false;
  for (const item of listExpressions()) {
    if (!item.autoParam) continue;
    let name = null;
    try {
      const classified = classifyExpr(item.latex);
      if (classified.kind === "parameter") name = classified.paramName;
    } catch {
      continue;
    }
    if (!name || refs.has(name)) continue;
    if (removeExprSilent(item.id)) removed = true;
  }
  return removed;
}

/**
 * While typing in a field expression, skip inserting/pruning auto-param rows.
 * Otherwise each letter (`c`→`co`→`cos`) recreates the math-field and kills
 * MathLive's inline shortcuts (`cos` → `\cos`).
 */
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

/**
 * Compile all expressions: parameter rows feed shared values; field rows become layers.
 * Free symbols without a dedicated `a=…` row get an auto-created parameter line.
 * @param {{ rebuildUi?: boolean, _afterEnsure?: boolean }} [opts]
 */
export function compileAllExprs(opts = {}) {
  const rebuildUi = opts.rebuildUi !== false;
  const items = listExpressions().filter((e) => e.enabled && String(e.latex || "").trim());

  /** @type {{ item: any, name: string }[]} */
  const paramRows = [];
  /** @type {{ item: any, compiled: any, fn: Function, role: string }[]} */
  const layers = [];
  const freeSet = new Set();
  const definedParams = new Set();
  /** @type {[string, string][]} */
  const warnings = [];
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
    const compiled = compileExpr(item.latex);
    for (const p of compiled.freeParams) freeSet.add(p);
    // Spatially constant (0th-order) fields: do not graph.
    if (!compiled.usesSpace || compiled.shade === "none") continue;
    const role = resolveExprRole(item.role, compiled.kind);
    layers.push({
      item,
      compiled,
      role,
      fn: compiled.bind(getParamValues()),
    });
  }

  replaceExprWarnings(warnings);

  /** @type {Parameters<typeof syncParamsFromDefinitions>[0]} */
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

  // Param-equation deps (a=b+1) without their own row.
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
  for (const L of layers) L.fn = L.compiled.bind(params);

  if (rebuildUi) {
    // Row inserts or kind flips: prefer full render when chrome can't sync in place.
    if (!state.exprListApi?.syncParamChrome?.()) {
      state.exprListApi?.render();
    }
  }

  const nCons = layers.filter((L) => L.role === "constraint").length;
  const nDens = layers.filter((L) => L.role === "density").length;
  state.lastExprMeta = {
    kind: nCons && nDens ? "mixed" : nCons ? "constraint" : "bare",
    shade: nCons && !nDens ? "iso" : "volume",
    isoLevel: 0,
    label: `${nDens} density · ${nCons} manifold`,
  };

  return {
    freeParams: [...freeSet].sort(),
    layers,
    warnings: warnings.map(([, msg]) => msg),
  };
}

export function applyPreset(key) {
  const p = PRESETS[key] ?? PRESETS.sincos;
  els.preset.value = key in PRESETS ? key : "sincos";
  state.pendingParamSeed = p.params ?? {};
  if (Array.isArray(p.expressions) && p.expressions.length) {
    setExpressions(p.expressions);
  } else {
    setExpressions([{ latex: p.latex }]);
  }
  state.exprListApi?.render();
}

export function layerRgbFromItem(item) {
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
    setExpressions([{ latex: PRESETS.sincos.latex }]);
  }
}
