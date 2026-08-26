/**
 * Dens/iso keyframe cache for animated free parameters.
 * Cold: fit K volumes over [min,max]. Hot: lerp adjacent frames (no DCT).
 */

import { fitChebyshev3D } from "./fit.js";
import { idctCheb3D, idctChebGrad3D } from "./chebIdct.js";
import { getParam } from "./params.js";

export const DEFAULT_KEYFRAME_K = 8;

/**
 * @typedef {{
 *   dens: Float32Array,
 *   gx?: Float32Array,
 *   gy?: Float32Array,
 *   gz?: Float32Array,
 *   cheb?: Float32Array,
 *   fitRel?: number,
 * }} KeyframeFrame
 *
 * @typedef {{
 *   paramName: string,
 *   min: number,
 *   max: number,
 *   K: number,
 *   deg: number,
 *   half: number,
 *   role: "density" | "constraint",
 *   isoLevel: number,
 *   latex: string,
 *   frames: KeyframeFrame[],
 *   scratch: KeyframeFrame,
 * }} LayerKeyframeCache
 */

/** @type {Map<string, LayerKeyframeCache>} */
const caches = new Map();

let lastBakeMs = 0;
let lastLerpMs = 0;
let lastKfLayers = 0;
let bakeMsAcc = 0;
let lerpMsAcc = 0;

export function getKeyframeMetrics() {
  return {
    bakeMs: lastBakeMs,
    lerpMs: lastLerpMs,
    layers: lastKfLayers,
    K: DEFAULT_KEYFRAME_K,
  };
}

export function clearKeyframeCaches() {
  caches.clear();
  lastBakeMs = 0;
  lastLerpMs = 0;
  lastKfLayers = 0;
  bakeMsAcc = 0;
  lerpMsAcc = 0;
}

/** @param {string} id */
export function invalidateKeyframeCache(id) {
  caches.delete(id);
}

/**
 * Eligible when dirty free-params collapse to exactly one cosine-animated slider
 * (not driven / time-equation).
 * @param {string[]} freeParams
 * @param {Set<string>} dirty
 * @returns {string | null} param name or null
 */
export function keyframeAnimParam(freeParams, dirty) {
  if (!dirty?.size || !freeParams?.length) return null;
  /** @type {string[]} */
  const hit = [];
  for (const p of freeParams) {
    if (dirty.has(p)) hit.push(p);
  }
  if (hit.length !== 1) return null;
  const name = hit[0];
  const st = getParam(name);
  if (!st || st.driven || !st.animating) return null;
  return name;
}

/**
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} t
 * @param {Float32Array} out
 */
export function lerpFloat32(a, b, t, out) {
  const n = out.length;
  const u = 1 - t;
  for (let i = 0; i < n; i++) out[i] = u * a[i] + t * b[i];
  return out;
}

/**
 * @param {LayerKeyframeCache} cache
 * @param {number} value
 * @returns {{ i0: number, i1: number, t: number }}
 */
function segmentForValue(cache, value) {
  const span = Math.max(1e-12, cache.max - cache.min);
  const u = Math.min(1, Math.max(0, (value - cache.min) / span));
  const K = cache.K;
  if (K <= 1) return { i0: 0, i1: 0, t: 0 };
  const x = u * (K - 1);
  const i0 = Math.min(K - 2, Math.max(0, Math.floor(x)));
  const i1 = i0 + 1;
  return { i0, i1, t: x - i0 };
}

/**
 * @param {LayerKeyframeCache} cache
 * @returns {boolean}
 */
function cacheMatches(cache, { paramName, min, max, K, deg, half, role, latex, isoLevel }) {
  return (
    cache.paramName === paramName &&
    cache.role === role &&
    cache.K === K &&
    cache.deg === deg &&
    Math.abs(cache.half - half) < 1e-12 &&
    Math.abs(cache.min - min) < 1e-12 &&
    Math.abs(cache.max - max) < 1e-12 &&
    Math.abs(cache.isoLevel - isoLevel) < 1e-12 &&
    cache.latex === latex &&
    cache.frames.length === K
  );
}

/**
 * Bake or reuse keyframes for one layer; return lerped dens (± grads) at `value`.
 *
 * @param {{
 *   layerId: string,
 *   latex: string,
 *   role: "density" | "constraint",
 *   isoLevel: number,
 *   paramName: string,
 *   compiled: { bind: (params: Record<string, number>) => (x:number,y:number,z:number)=>number },
 *   baseParams: Record<string, number>,
 *   half: number,
 *   deg: number,
 *   K?: number,
 * }} opts
 * @returns {{
 *   dens: Float32Array,
 *   gx?: Float32Array,
 *   gy?: Float32Array,
 *   gz?: Float32Array,
 *   cheb?: Float32Array,
 *   fitRel?: number,
 *   M: number,
 *   baked: boolean,
 * }}
 */
export function sampleLayerKeyframes(opts) {
  const K = Math.max(2, opts.K ?? DEFAULT_KEYFRAME_K);
  const st = getParam(opts.paramName);
  if (!st) throw new Error(`Unknown param “${opts.paramName}” for keyframes`);
  const min = st.min;
  const max = st.max;
  const value = st.value;
  const key = opts.layerId;

  let cache = caches.get(key);
  const meta = {
    paramName: opts.paramName,
    min,
    max,
    K,
    deg: opts.deg,
    half: opts.half,
    role: opts.role,
    latex: opts.latex,
    isoLevel: opts.isoLevel ?? 0,
  };

  let baked = false;
  if (!cache || !cacheMatches(cache, meta)) {
    const t0 = performance.now();
    const frames = [];
    let firstCheb = null;
    let firstRel = NaN;
    for (let k = 0; k < K; k++) {
      const a = min + ((max - min) * k) / (K - 1);
      const params = { ...opts.baseParams, [opts.paramName]: a };
      const fn = opts.compiled.bind(params);
      const fit = fitChebyshev3D(fn, opts.half, opts.deg, { skipL2: true });
      const idct = idctCheb3D(fit.cheb, fit.deg, fit.deg + 1);
      /** @type {KeyframeFrame} */
      const frame = { dens: idct.dens, cheb: fit.cheb, fitRel: fit.fitRelL2 };
      if (opts.role === "constraint") {
        const grad = idctChebGrad3D(fit.cheb, fit.deg, fit.deg + 1);
        frame.gx = grad.gx;
        frame.gy = grad.gy;
        frame.gz = grad.gz;
      }
      if (!firstCheb) {
        firstCheb = fit.cheb;
        firstRel = fit.fitRelL2;
      }
      frames.push(frame);
    }
    const n = frames[0].dens.length;
    /** @type {KeyframeFrame} */
    const scratch = { dens: new Float32Array(n) };
    if (opts.role === "constraint") {
      scratch.gx = new Float32Array(n);
      scratch.gy = new Float32Array(n);
      scratch.gz = new Float32Array(n);
    }
    cache = {
      ...meta,
      frames,
      scratch,
    };
    caches.set(key, cache);
    bakeMsAcc += performance.now() - t0;
    lastBakeMs = bakeMsAcc;
    baked = true;
  }

  const tLerp = performance.now();
  const { i0, i1, t } = segmentForValue(cache, value);
  const a = cache.frames[i0];
  const b = cache.frames[i1];
  const out = cache.scratch;
  lerpFloat32(a.dens, b.dens, t, out.dens);
  if (cache.role === "constraint" && out.gx && a.gx && b.gx) {
    lerpFloat32(a.gx, b.gx, t, out.gx);
    lerpFloat32(a.gy, b.gy, t, out.gy);
    lerpFloat32(a.gz, b.gz, t, out.gz);
  }
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  return {
    dens: out.dens,
    gx: out.gx,
    gy: out.gy,
    gz: out.gz,
    cheb: t < 0.5 ? a.cheb : b.cheb,
    fitRel: t < 0.5 ? a.fitRel : b.fitRel,
    M: Math.round(Math.cbrt(out.dens.length)),
    baked,
  };
}

/** Call once per uploadFit anim pass to reset per-frame layer counter. */
export function beginKeyframePass() {
  lastKfLayers = 0;
  bakeMsAcc = 0;
  lerpMsAcc = 0;
  lastBakeMs = 0;
  lastLerpMs = 0;
}

export function noteKeyframeLayer() {
  lastKfLayers++;
}
