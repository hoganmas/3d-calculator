/**
 * Runtime values for named parameters defined in the expression list (`a = …`).
 * RHS may reference other params.
 */

import { compileParamLatex, formatParamLatexValue } from "../math/fit.js";

/**
 * @typedef {"pingpong" | "loop"} AnimMode
 * @typedef {{
 *   value: number,
 *   min: number,
 *   max: number,
 *   step: number,
 *   animating: boolean,
 *   speed: number,
 *   phase: number,
 *   animMode: AnimMode,
 *   latex: string,
 *   exprId: string | null,
 *   driven: boolean,
 *   freeParams: string[],
 *   error: string | null,
 * }} ParamState
 */

/** @type {Map<string, ParamState>} */
const params = new Map();

const DEFAULT_MIN = -10;
const DEFAULT_MAX = 10;
const DEFAULT_VALUE = 1;
const DEFAULT_SPEED = 0.35;
/** @type {AnimMode} */
const DEFAULT_ANIM_MODE = "pingpong";

/**
 * @param {unknown} mode
 * @returns {AnimMode}
 */
export function normalizeAnimMode(mode) {
  return mode === "loop" ? "loop" : "pingpong";
}

/**
 * Phase so `tickParamAnimation(timeSec)` yields the current value.
 * @param {Pick<ParamState, "value" | "min" | "max" | "speed" | "animMode">} p
 * @param {number} timeSec
 */
export function phaseForValue(p, timeSec) {
  const span = Math.max(1e-9, p.max - p.min);
  const u = Math.min(1, Math.max(0, (p.value - p.min) / span));
  if (normalizeAnimMode(p.animMode) === "loop") {
    return Math.min(0.999999, u) - timeSec * p.speed;
  }
  const s = Math.min(1, Math.max(-1, (u - 0.5) / 0.5));
  return Math.asin(s) / (2 * Math.PI) - timeSec * p.speed;
}

/**
 * @param {string} name
 * @param {Partial<ParamState> & { animate?: boolean }} [init]
 * @returns {ParamState}
 */
function makeParam(name, init = {}) {
  let min = Number.isFinite(init.min) ? init.min : DEFAULT_MIN;
  let max = Number.isFinite(init.max) ? init.max : DEFAULT_MAX;
  if (max < min) [min, max] = [max, min];
  let value = Number.isFinite(init.value) ? init.value : DEFAULT_VALUE;
  value = Math.min(max, Math.max(min, value));
  const span = Math.max(1e-9, max - min);
  const step =
    Number.isFinite(init.step) && init.step > 0
      ? init.step
      : Math.max(0.01, Number((span / 200).toPrecision(2)));
  const latex =
    typeof init.latex === "string" && init.latex.trim()
      ? init.latex.trim()
      : `${name}=${formatParamLatexValue(value)}`;
  return {
    value,
    min,
    max,
    step,
    animating: !!(init.animating ?? init.animate),
    speed: Number.isFinite(init.speed) && init.speed > 0 ? init.speed : DEFAULT_SPEED,
    phase: Number.isFinite(init.phase) ? init.phase : Math.random(),
    animMode: normalizeAnimMode(init.animMode),
    latex,
    exprId: typeof init.exprId === "string" ? init.exprId : null,
    driven: false,
    freeParams: [],
    error: null,
  };
}

/** @returns {string[]} */
export function listParamNames() {
  return [...params.keys()].sort();
}

/** @returns {Record<string, number>} */
export function getParamValues() {
  /** @type {Record<string, number>} */
  const out = {};
  for (const [name, p] of params) out[name] = p.value;
  return out;
}

/** @param {string} name */
export function getParam(name) {
  return params.get(name) ?? null;
}

export function anyParamAnimating() {
  for (const p of params.values()) if (p.animating) return true;
  return false;
}

/**
 * Params that may have changed this animation tick, plus anything that
 * transitively depends on them via `a = f(b,…)`.
 * @returns {Set<string>}
 */
export function collectAnimDirtyParams() {
  /** @type {Set<string>} */
  const dirty = new Set();
  for (const [name, p] of params) {
    if (p.animating && !p.driven) dirty.add(name);
  }
  let grew = true;
  while (grew) {
    grew = false;
    for (const [name, p] of params) {
      if (dirty.has(name) || !p.driven) continue;
      for (const dep of p.freeParams) {
        if (dirty.has(dep)) {
          dirty.add(name);
          grew = true;
          break;
        }
      }
    }
  }
  return dirty;
}

/**
 * Replace param map from expression-list parameter rows.
 * @param {{
 *   name: string,
 *   latex: string,
 *   exprId: string,
 *   min?: number,
 *   max?: number,
 *   speed?: number,
 *   animating?: boolean,
 *   phase?: number,
 *   animMode?: AnimMode,
 *   value?: number,
 * }[]} defs
 * @param {Record<string, Partial<ParamState> & { animate?: boolean }>} [seed]
 */
export function syncParamsFromDefinitions(defs, seed = {}) {
  const keep = new Set(defs.map((d) => d.name));
  for (const name of [...params.keys()]) {
    if (!keep.has(name)) params.delete(name);
  }
  for (const d of defs) {
    const s = seed[d.name] ?? {};
    const cur = params.get(d.name);
    if (!cur) {
      params.set(
        d.name,
        makeParam(d.name, {
          ...s,
          latex: d.latex,
          exprId: d.exprId,
          min: d.min ?? s.min,
          max: d.max ?? s.max,
          speed: d.speed ?? s.speed,
          animating: d.animating ?? s.animating ?? s.animate,
          phase: d.phase ?? s.phase,
          animMode: d.animMode ?? s.animMode,
          value: d.value ?? s.value,
        }),
      );
      continue;
    }
    params.set(d.name, {
      ...cur,
      latex: d.latex,
      exprId: d.exprId,
      min: Number.isFinite(d.min) ? d.min : cur.min,
      max: Number.isFinite(d.max) ? d.max : cur.max,
      speed: Number.isFinite(d.speed) && d.speed > 0 ? d.speed : cur.speed,
      animating: typeof d.animating === "boolean" ? d.animating : cur.animating,
      phase: Number.isFinite(d.phase) ? d.phase : cur.phase,
      animMode: d.animMode != null ? normalizeAnimMode(d.animMode) : cur.animMode,
    });
  }
}

/**
 * @param {Record<string, Partial<ParamState> & { animate?: boolean }}>} seed
 */
export function applyParamSeed(seed) {
  for (const [name, init] of Object.entries(seed ?? {})) {
    const cur = params.get(name);
    if (!cur) continue;
    const next = makeParam(name, { ...cur, ...init, latex: cur.latex, exprId: cur.exprId });
    if (Number.isFinite(init.value) && (typeof init.latex !== "string" || !init.latex.trim())) {
      next.latex = `${name}=${formatParamLatexValue(init.value)}`;
      next.value = Math.min(next.max, Math.max(next.min, init.value));
    }
    if (init.animating === undefined && init.animate === undefined) next.animating = cur.animating;
    if (!Number.isFinite(init.phase)) next.phase = cur.phase;
    params.set(name, next);
  }
}

/**
 * @param {string} name
 * @param {Partial<ParamState>} patch
 */
export function updateParam(name, patch) {
  const cur = params.get(name);
  if (!cur) return null;
  let min = Number.isFinite(patch.min) ? patch.min : cur.min;
  let max = Number.isFinite(patch.max) ? patch.max : cur.max;
  if (max < min) [min, max] = [max, min];
  let value = Number.isFinite(patch.value) ? patch.value : cur.value;
  value = Math.min(max, Math.max(min, value));
  let latex = typeof patch.latex === "string" ? patch.latex : cur.latex;
  if (Number.isFinite(patch.value) && patch.latex === undefined) {
    latex = `${name}=${formatParamLatexValue(value)}`;
  }
  const next = {
    ...cur,
    ...patch,
    min,
    max,
    value,
    latex,
    step: Number.isFinite(patch.step) && patch.step > 0 ? patch.step : cur.step,
    speed: Number.isFinite(patch.speed) && patch.speed > 0 ? patch.speed : cur.speed,
    animMode: patch.animMode != null ? normalizeAnimMode(patch.animMode) : cur.animMode,
  };
  params.set(name, next);
  return next;
}

export function setParamValue(name, value, { stopAnim = true, rewriteLatex = true } = {}) {
  const cur = params.get(name);
  if (!cur) return null;
  const v = Math.min(cur.max, Math.max(cur.min, Number(value)));
  if (!Number.isFinite(v)) return cur;
  const next = {
    ...cur,
    value: v,
    animating: stopAnim ? false : cur.animating,
    latex: rewriteLatex ? `${name}=${formatParamLatexValue(v)}` : cur.latex,
    driven: rewriteLatex ? false : cur.driven,
    freeParams: rewriteLatex ? [] : cur.freeParams,
    error: rewriteLatex ? null : cur.error,
  };
  params.set(name, next);
  return next;
}

/** Recompile one param from its latex. */
export function recompileParam(name) {
  const cur = params.get(name);
  if (!cur) return false;
  try {
    const compiled = compileParamLatex(cur.latex, name);
    const driven = compiled.freeParams.length > 0;
    /** @type {ParamState} */
    const next = {
      ...cur,
      freeParams: compiled.freeParams,
      driven,
      error: null,
      animating: driven ? false : cur.animating,
    };
    if (compiled.isConstant && compiled.constantValue != null) {
      next.value = Math.min(next.max, Math.max(next.min, compiled.constantValue));
    }
    params.set(name, next);
    return true;
  } catch (err) {
    params.set(name, {
      ...cur,
      error: err instanceof Error ? err.message : String(err),
      driven: false,
      freeParams: [],
    });
    return false;
  }
}

export function recompileAllParams() {
  for (const name of listParamNames()) recompileParam(name);
  return listParamNames().flatMap((name) => params.get(name)?.freeParams ?? []);
}

/**
 * Evaluate driven parameter equations `a = f(b,…)`.
 * @returns {boolean}
 */
export function evalParamEquations() {
  const names = listParamNames();
  if (!names.length) return false;

  /** @type {Map<string, ReturnType<typeof compileParamLatex>>} */
  const compiled = new Map();
  for (const name of names) {
    const p = params.get(name);
    if (!p || p.error || !p.driven) continue;
    try {
      compiled.set(name, compileParamLatex(p.latex, name));
    } catch {
      /* keep */
    }
  }
  if (!compiled.size) return false;

  /** @type {Record<string, number>} */
  const scope = {};
  for (const [n, p] of params) scope[n] = p.value;

  let changed = false;
  for (let pass = 0; pass < names.length + 2; pass++) {
    let passChanged = false;
    for (const [name, c] of compiled) {
      const p = params.get(name);
      if (!p) continue;
      try {
        const v = c.eval(scope);
        if (!Number.isFinite(v)) continue;
        const clamped = Math.min(p.max, Math.max(p.min, v));
        scope[name] = clamped;
        if (Math.abs(clamped - p.value) > 1e-12) {
          params.set(name, { ...p, value: clamped });
          changed = true;
          passChanged = true;
        }
      } catch {
        /* skip */
      }
    }
    if (!passChanged) break;
  }
  return changed;
}

export function toggleParamAnimate(name, timeSec = performance.now() / 1000) {
  const cur = params.get(name);
  if (!cur || cur.driven) return cur;
  const next = { ...cur, animating: !cur.animating };
  if (next.animating) {
    next.phase = phaseForValue(next, timeSec);
  }
  params.set(name, next);
  return next;
}

/**
 * Permanently stop parameter animation (e.g. when editing its LaTeX).
 * @param {string} name
 * @returns {ParamState | null}
 */
export function stopParamAnimation(name) {
  const cur = params.get(name);
  if (!cur || !cur.animating) return cur;
  const next = { ...cur, animating: false };
  params.set(name, next);
  return next;
}

/**
 * @param {number} timeSec
 * @returns {boolean}
 */
export function tickParamAnimation(timeSec) {
  let changed = false;
  for (const [name, p] of params) {
    if (!p.animating || p.driven) continue;
    const tau = timeSec * p.speed + p.phase;
    let u;
    if (normalizeAnimMode(p.animMode) === "loop") {
      u = ((tau % 1) + 1) % 1;
    } else {
      u = 0.5 + 0.5 * Math.sin(2 * Math.PI * tau);
    }
    const value = p.min + (p.max - p.min) * u;
    if (Math.abs(value - p.value) > 1e-12) {
      params.set(name, {
        ...p,
        value,
        latex: `${name}=${formatParamLatexValue(value)}`,
      });
      changed = true;
    }
  }
  return changed;
}
