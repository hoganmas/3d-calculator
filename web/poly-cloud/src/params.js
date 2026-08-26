/**
 * Name parameters for the density expression.
 * Each free symbol (besides spatial x,y,z / r,θ,φ,ρ) gets a slider; optional cosine ping-pong animation.
 */

/**
 * @typedef {{
 *   value: number,
 *   min: number,
 *   max: number,
 *   step: number,
 *   animating: boolean,
 *   speed: number,
 *   phase: number,
 * }} ParamState
 */

/** @type {Map<string, ParamState>} */
const params = new Map();

/** @type {((name: string, p: ParamState) => void) | null} */
let onChange = null;

const DEFAULT_MIN = 0;
const DEFAULT_MAX = 2;
const DEFAULT_VALUE = 1;
const DEFAULT_SPEED = 0.35; // cycles / second

/**
 * @param {Partial<ParamState> & { animate?: boolean }} [init]
 * @returns {ParamState}
 */
function makeParam(init = {}) {
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
  return {
    value,
    min,
    max,
    step,
    animating: !!(init.animating ?? init.animate),
    speed: Number.isFinite(init.speed) && init.speed > 0 ? init.speed : DEFAULT_SPEED,
    phase: Number.isFinite(init.phase) ? init.phase : Math.random(),
  };
}

export function setParamsOnChange(fn) {
  onChange = fn;
}

function emit(name) {
  const p = params.get(name);
  if (p && onChange) onChange(name, p);
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
 * Keep only symbols present in the expression; preserve values when names match.
 * @param {string[]} freeParams
 * @param {Record<string, Partial<ParamState> & { animate?: boolean }>} [seed]
 */
export function syncParamsFromSymbols(freeParams, seed = {}) {
  const keep = new Set(freeParams);
  for (const name of [...params.keys()]) {
    if (!keep.has(name)) params.delete(name);
  }
  for (const name of freeParams) {
    if (params.has(name)) continue;
    params.set(name, makeParam(seed[name] ?? {}));
  }
}

/**
 * Apply preset defaults (value/min/max/animate) for known names; create missing.
 * @param {Record<string, Partial<ParamState> & { animate?: boolean }>} seed
 */
export function applyParamSeed(seed) {
  for (const [name, init] of Object.entries(seed ?? {})) {
    const cur = params.get(name);
    if (!cur) {
      params.set(name, makeParam(init));
      continue;
    }
    const next = makeParam({ ...cur, ...init });
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
  const next = {
    ...cur,
    ...patch,
    min,
    max,
    value,
    step:
      Number.isFinite(patch.step) && patch.step > 0 ? patch.step : cur.step,
    speed:
      Number.isFinite(patch.speed) && patch.speed > 0 ? patch.speed : cur.speed,
  };
  params.set(name, next);
  emit(name);
  return next;
}

export function setParamValue(name, value, { stopAnim = true } = {}) {
  const cur = params.get(name);
  if (!cur) return null;
  const v = Math.min(cur.max, Math.max(cur.min, Number(value)));
  if (!Number.isFinite(v)) return cur;
  const next = {
    ...cur,
    value: v,
    animating: stopAnim ? false : cur.animating,
  };
  params.set(name, next);
  emit(name);
  return next;
}

export function toggleParamAnimate(name, timeSec = performance.now() / 1000) {
  const cur = params.get(name);
  if (!cur) return null;
  const next = { ...cur, animating: !cur.animating };
  if (next.animating) {
    const u = (cur.value - cur.min) / Math.max(1e-9, cur.max - cur.min);
    const s = Math.min(1, Math.max(-1, (u - 0.5) / 0.5));
    next.phase = Math.asin(s) / (2 * Math.PI) - timeSec * next.speed;
  }
  params.set(name, next);
  emit(name);
  return next;
}

/**
 * Advance animated params. Cosine ping-pong in [min, max].
 * @param {number} timeSec absolute time (performance.now()/1000)
 * @returns {boolean} true if any value changed
 */
export function tickParamAnimation(timeSec) {
  let changed = false;
  for (const [name, p] of params) {
    if (!p.animating) continue;
    const u = 0.5 + 0.5 * Math.sin(2 * Math.PI * (timeSec * p.speed + p.phase));
    const value = p.min + (p.max - p.min) * u;
    if (Math.abs(value - p.value) > 1e-12) {
      params.set(name, { ...p, value });
      changed = true;
    }
  }
  return changed;
}

export function clearParams() {
  params.clear();
}
