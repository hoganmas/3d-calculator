/**
 * Dens/iso keyframe cache for animated free parameters.
 */

import { fitChebyshev3D } from "../math/fit.js";
import { idctCheb3D, idctChebGrad3D } from "../math/idct.js";
import { getParam } from "./params.js";
import type { CompiledExpr, KeyframeFrame } from "../types/models.js";

export const DEFAULT_KEYFRAME_K = 8;

interface BakeStages {
  sampleMs: number;
  chebMs: number;
  idctMs: number;
  gradMs: number;
}

interface BakeFrameOpts {
  layerId: string;
  latex: string;
  role: "density" | "constraint";
  isoLevel: number;
  paramName: string;
  compiled: { bind: CompiledExpr["bind"] };
  baseParams: Record<string, number>;
  half: number;
  deg: number;
  K?: number;
  min?: number;
  max?: number;
}

interface LayerKeyframeCache {
  paramName: string;
  min: number;
  max: number;
  K: number;
  deg: number;
  half: number;
  role: "density" | "constraint";
  isoLevel: number;
  latex: string;
  frames: (KeyframeFrame | null)[];
  scratch: KeyframeFrame;
  gen: number;
  readyCount: number;
  bakeOpts: BakeFrameOpts;
}

interface KeyframeProgressInfo {
  layerId: string;
  index: number;
  frame: KeyframeFrame;
  readyCount: number;
  K: number;
  done: boolean;
}

interface EnsureKeyframesOpts {
  layerId: string;
  latex: string;
  role: "density" | "constraint";
  isoLevel: number;
  paramName: string;
  compiled: { bind: CompiledExpr["bind"] };
  baseParams: Record<string, number>;
  half: number;
  deg: number;
  K?: number;
}

interface BakeDetail extends BakeStages {
  frames: number;
  deg: number;
  M: number;
  role: string;
  layerId: string;
  paramName: string;
  bakeMs: number;
  async: boolean;
  index?: number;
  syncPair?: [number, number];
}

const caches = new Map<string, LayerKeyframeCache>();

const asyncJobs = new Map<string, { gen: number; timer: number; cancelled: boolean }>();

let onKeyframeProgress: ((info: KeyframeProgressInfo) => void) | null = null;

let cacheGen = 0;
let lastBakeMs = 0;
let lastLerpMs = 0;
let lastKfLayers = 0;
let bakeMsAcc = 0;
let lerpMsAcc = 0;
let lastBakeDetails: BakeDetail[] = [];

export function setKeyframeProgressHandler(cb: typeof onKeyframeProgress) {
  onKeyframeProgress = cb;
}

export function getKeyframeMetrics() {
  const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
  for (const d of lastBakeDetails) {
    stages.sampleMs += d.sampleMs || 0;
    stages.chebMs += d.chebMs || 0;
    stages.idctMs += d.idctMs || 0;
    stages.gradMs += d.gradMs || 0;
  }
  return {
    bakeMs: lastBakeMs,
    lerpMs: lastLerpMs,
    layers: lastKfLayers,
    K: DEFAULT_KEYFRAME_K,
    stages,
    details: lastBakeDetails.slice(),
  };
}

function cancelAsyncJob(layerId: string) {
  const job = asyncJobs.get(layerId);
  if (!job) return;
  job.cancelled = true;
  if (job.timer) clearTimeout(job.timer);
  asyncJobs.delete(layerId);
}

export function clearKeyframeCaches() {
  for (const id of [...asyncJobs.keys()]) cancelAsyncJob(id);
  caches.clear();
  lastBakeMs = 0;
  lastLerpMs = 0;
  lastKfLayers = 0;
  bakeMsAcc = 0;
  lerpMsAcc = 0;
  lastBakeDetails = [];
}

/**
 * Eligible when dirty free-params collapse to exactly one animated slider
 * (not driven by another equation).
 * @param {string[]} freeParams
 * @param {Set<string>} dirty
 * @returns {string | null} param name or null
 */
export function keyframeAnimParam(freeParams: string[], dirty: Set<string>) {
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
function lerpFloat32(a: Float32Array, b: Float32Array, t: number, out: Float32Array) {
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
function segmentForValue(cache: LayerKeyframeCache, value: number) {
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
 * Bake priority: current blend pair first, then expand outward.
 * @param {number} K
 * @param {number} i0
 * @param {number} i1
 * @returns {number[]}
 */
function bakeOrder(K: number, i0: number, i1: number) {
  const order: number[] = [];
  const seen = new Set<number>();
  const push = (i: number) => {
    const k = i | 0;
    if (k < 0 || k >= K || seen.has(k)) return;
    seen.add(k);
    order.push(k);
  };
  push(i0);
  push(i1);
  for (let d = 1; d < K; d++) {
    push(i0 - d);
    push(i1 + d);
    push(i0 + d);
    push(i1 - d);
  }
  for (let k = 0; k < K; k++) push(k);
  return order;
}

/**
 * @param {LayerKeyframeCache} cache
 * @returns {boolean}
 */
function cacheMatches(
  cache: LayerKeyframeCache,
  meta: Pick<
    LayerKeyframeCache,
    "paramName" | "min" | "max" | "K" | "deg" | "half" | "role" | "latex" | "isoLevel"
  >,
) {
  return (
    cache.paramName === meta.paramName &&
    cache.role === meta.role &&
    cache.K === meta.K &&
    cache.deg === meta.deg &&
    Math.abs(cache.half - meta.half) < 1e-12 &&
    Math.abs(cache.min - meta.min) < 1e-12 &&
    Math.abs(cache.max - meta.max) < 1e-12 &&
    Math.abs(cache.isoLevel - meta.isoLevel) < 1e-12 &&
    cache.latex === meta.latex &&
    cache.frames.length === meta.K
  );
}

/**
 * @param {object} opts
 * @param {number} k
 * @param {{ sampleMs: number, chebMs: number, idctMs: number, gradMs: number }} stages
 * @returns {KeyframeFrame}
 */
function bakeFrameAt(opts: BakeFrameOpts, k: number, stages: BakeStages | null): KeyframeFrame {
  const st = getParam(opts.paramName);
  const min = st?.min ?? opts.min ?? 0;
  const max = st?.max ?? opts.max ?? 1;
  const K = opts.K ?? DEFAULT_KEYFRAME_K;
  const a = min + ((max - min) * k) / Math.max(1, K - 1);
  const params = { ...opts.baseParams, [opts.paramName]: a };
  const fn = opts.compiled.bind(params);
  const fit = fitChebyshev3D(fn, opts.half, opts.deg, { skipL2: true, skipMono: true });
  if (fit.timing && stages) {
    stages.sampleMs += fit.timing.sampleMs;
    stages.chebMs += fit.timing.chebMs;
  }
  let tStage = performance.now();
  const idct = idctCheb3D(fit.cheb, fit.deg, fit.deg + 1);
  if (stages) stages.idctMs += performance.now() - tStage;
  const frame: KeyframeFrame = { dens: idct.dens, cheb: fit.cheb, fitRel: fit.fitRelL2 };
  if (opts.role === "constraint") {
    tStage = performance.now();
    const grad = idctChebGrad3D(fit.cheb, fit.deg, fit.deg + 1);
    if (stages) stages.gradMs += performance.now() - tStage;
    frame.gx = grad.gx;
    frame.gy = grad.gy;
    frame.gz = grad.gz;
  }
  return frame;
}

/**
 * @param {LayerKeyframeCache} cache
 * @param {number} k
 * @param {KeyframeFrame} frame
 */
function storeFrame(cache: LayerKeyframeCache, k: number, frame: KeyframeFrame) {
  if (cache.frames[k]) return;
  cache.frames[k] = frame;
  cache.readyCount++;
}

/**
 * Frames array for upload/GPU: null slots filled with nearest ready frame.
 * @param {(KeyframeFrame | null)[]} frames
 * @returns {KeyframeFrame[]}
 */
function materializeKeyframeFrames(frames: (KeyframeFrame | null)[]): KeyframeFrame[] {
  let last: KeyframeFrame | null = null;
  for (const fr of frames) {
    if (fr) {
      last = fr;
      break;
    }
  }
  if (!last) throw new Error("no keyframes ready");
  const out: KeyframeFrame[] = new Array(frames.length);
  for (let i = 0; i < frames.length; i++) {
    if (frames[i]) {
      last = frames[i];
      out[i] = frames[i]!;
    } else {
      // Look ahead for a ready frame if we haven't seen one yet in this pass.
      let fwd = last;
      for (let j = i + 1; j < frames.length; j++) {
        if (frames[j]) {
          fwd = frames[j];
          break;
        }
      }
      out[i] = (fwd || last)!;
    }
  }
  return out;
}

/**
 * @param {string} layerId
 * @param {LayerKeyframeCache} cache
 */
function scheduleAsyncFill(layerId: string, cache: LayerKeyframeCache) {
  cancelAsyncJob(layerId);
  const gen = cache.gen;
  const job = { gen, timer: 0, cancelled: false };
  asyncJobs.set(layerId, job);

  const pump = () => {
    if (job.cancelled || job.gen !== cache.gen) return;
    const live = caches.get(layerId);
    if (!live || live.gen !== gen) return;

    const st = getParam(live.paramName);
    const value = st?.value ?? live.min;
    const { i0, i1 } = segmentForValue(live, value);
    const order = bakeOrder(live.K, i0, i1);
    const next = order.find((k) => !live.frames[k]);
    if (next == null) {
      asyncJobs.delete(layerId);
      onKeyframeProgress?.({
        layerId,
        index: -1,
        frame: live.frames[0]!,
        readyCount: live.readyCount,
        K: live.K,
        done: true,
      });
      return;
    }

    const t0 = performance.now();
    const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
    const frame = bakeFrameAt({ ...live.bakeOpts, K: live.K, min: live.min, max: live.max }, next, stages);
    if (job.cancelled || caches.get(layerId)?.gen !== gen) return;
    storeFrame(live, next, frame);
    const bakeMs = performance.now() - t0;
    bakeMsAcc += bakeMs;
    lastBakeMs = bakeMsAcc;
    lastBakeDetails.push({
      ...stages,
      frames: 1,
      deg: live.deg,
      M: Math.round(Math.cbrt(frame.dens.length)),
      role: live.role,
      layerId,
      paramName: live.paramName,
      bakeMs,
      async: true,
      index: next,
    });
    onKeyframeProgress?.({
      layerId,
      index: next,
      frame,
      readyCount: live.readyCount,
      K: live.K,
      done: live.readyCount >= live.K,
    });

    if (live.readyCount >= live.K) {
      asyncJobs.delete(layerId);
      return;
    }
    // Yield so RAF / input stay responsive; one keyframe per tick.
    job.timer = window.setTimeout(pump, 0);
  };

  job.timer = window.setTimeout(pump, 0);
}

/**
 * Bake or reuse keyframes; return GPU blend indices (no dens lerp).
 * Sync-bakes only the current blend pair; remaining frames fill asynchronously.
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
 *   frames: KeyframeFrame[],
 *   rawFrames: (KeyframeFrame | null)[],
 *   blend: { i0: number, i1: number, t: number },
 *   cheb?: Float32Array,
 *   fitRel?: number,
 *   M: number,
 *   baked: boolean,
 *   readyCount: number,
 *   complete: boolean,
 * }}
 */
export function ensureLayerKeyframes(opts: EnsureKeyframesOpts) {
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
    cancelAsyncJob(key);
    cacheGen++;
    cache = {
      ...meta,
      frames: new Array(K).fill(null),
      scratch: { dens: new Float32Array(0) },
      gen: cacheGen,
      readyCount: 0,
      bakeOpts: {
        layerId: opts.layerId,
        latex: opts.latex,
        role: opts.role,
        isoLevel: opts.isoLevel ?? 0,
        paramName: opts.paramName,
        compiled: opts.compiled,
        baseParams: { ...opts.baseParams },
        half: opts.half,
        deg: opts.deg,
        K,
        min,
        max,
      },
    };
    caches.set(key, cache);
    baked = true;
  } else {
    // Keep baseParams fresh for async fills (other sliders may have moved).
    cache.bakeOpts = {
      ...cache.bakeOpts,
      baseParams: { ...opts.baseParams },
      compiled: opts.compiled,
    };
  }

  const blend = segmentForValue(cache, value);
  const t0 = performance.now();
  const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
  let syncCount = 0;
  for (const k of [blend.i0, blend.i1]) {
    if (cache.frames[k]) continue;
    const frame = bakeFrameAt({ ...cache.bakeOpts, K, min, max }, k, stages);
    storeFrame(cache, k, frame);
    syncCount++;
    baked = true;
  }

  // Allocate scratch once we know volume size.
  const proto = cache.frames[blend.i0] || cache.frames[blend.i1];
  if (proto && (!cache.scratch.dens || cache.scratch.dens.length !== proto.dens.length)) {
    const n = proto.dens.length;
    cache.scratch = { dens: new Float32Array(n) };
    if (opts.role === "constraint") {
      cache.scratch.gx = new Float32Array(n);
      cache.scratch.gy = new Float32Array(n);
      cache.scratch.gz = new Float32Array(n);
    }
  }

  if (syncCount > 0) {
    const bakeMs = performance.now() - t0;
    bakeMsAcc += bakeMs;
    lastBakeMs = bakeMsAcc;
    lastBakeDetails.push({
      ...stages,
      frames: syncCount,
      deg: opts.deg,
      M: Math.round(Math.cbrt(proto?.dens.length || 0)),
      role: opts.role,
      layerId: opts.layerId,
      paramName: opts.paramName,
      bakeMs,
      async: false,
      syncPair: [blend.i0, blend.i1],
    });
  }

  if (cache.readyCount < cache.K) {
    scheduleAsyncFill(key, cache);
  }

  const frames = materializeKeyframeFrames(cache.frames);
  const a = frames[blend.i0];
  const b = frames[blend.i1];
  return {
    frames,
    rawFrames: cache.frames.slice(),
    blend,
    cheb: blend.t < 0.5 ? a.cheb : b.cheb,
    fitRel: blend.t < 0.5 ? a.fitRel : b.fitRel,
    M: Math.round(Math.cbrt(a.dens.length)),
    baked,
    readyCount: cache.readyCount,
    complete: cache.readyCount >= cache.K,
  };
}

/**
 * Bake or reuse keyframes; CPU-lerp dens (± grads) at current param value.
 *
 * @param {Parameters<typeof ensureLayerKeyframes>[0]} opts
 */
export function sampleLayerKeyframes(opts: EnsureKeyframesOpts) {
  const ensured = ensureLayerKeyframes(opts);
  const cache = caches.get(opts.layerId);
  if (!cache) throw new Error("keyframe cache missing after ensure");
  const tLerp = performance.now();
  const { i0, i1, t } = ensured.blend;
  const a = cache.frames[i0];
  const b = cache.frames[i1];
  if (!a || !b) throw new Error("sync keyframe pair missing");
  const out = cache.scratch;
  lerpFloat32(a.dens, b.dens, t, out.dens);
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  return {
    dens: out.dens,
    cheb: ensured.cheb,
    fitRel: ensured.fitRel,
    M: ensured.M,
    baked: ensured.baked,
    readyCount: ensured.readyCount,
    complete: ensured.complete,
    frames: ensured.frames,
  };
}

/** Call once per uploadFit anim pass to reset per-frame layer counter. */
export function beginKeyframePass() {
  lastKfLayers = 0;
  bakeMsAcc = 0;
  lerpMsAcc = 0;
  lastBakeMs = 0;
  lastLerpMs = 0;
  lastBakeDetails = [];
}

/**
 * Pretty-print the last keyframe bake to the console (no-op if nothing baked).
 * @param {string} [reason]
 */
export function logKeyframeBake(reason = "bake") {
  if (!lastBakeDetails.length) return;
  const kf = getKeyframeMetrics();
  const s = kf.stages;
  const stageTotal = s.sampleMs + s.chebMs + s.idctMs + s.gradMs;
  const pct = (ms: number) => (stageTotal > 0 ? ((100 * ms) / stageTotal).toFixed(0) : "0");
  const syncFrames = lastBakeDetails.reduce((n, d) => n + (d.async ? 0 : d.frames || 0), 0);
  console.log(
    `[keyframes] ${reason}: ${kf.bakeMs.toFixed(1)}ms sync · ` +
      `${syncFrames} frame(s) now · rest async · K=${kf.K} · ` +
      `sample ${s.sampleMs.toFixed(1)}ms (${pct(s.sampleMs)}%) · ` +
      `dct ${s.chebMs.toFixed(1)}ms (${pct(s.chebMs)}%) · ` +
      `idct ${s.idctMs.toFixed(1)}ms (${pct(s.idctMs)}%) · ` +
      `grad ${s.gradMs.toFixed(1)}ms (${pct(s.gradMs)}%)`,
    lastBakeDetails,
  );
}

export function noteKeyframeLayer() {
  lastKfLayers++;
}

/**
 * Current GPU blend segment for a cached layer (no bake / no lerp).
 * If the needed pair isn't ready yet, returns null (caller should ensure()).
 * @param {string} layerId
 * @returns {{ id: string, i0: number, i1: number, t: number } | null}
 */
export function peekKeyframeBlend(layerId: string) {
  const cache = caches.get(layerId);
  if (!cache) return null;
  const st = getParam(cache.paramName);
  if (!st) return null;
  const blend = segmentForValue(cache, st.value);
  if (!cache.frames[blend.i0] || !cache.frames[blend.i1]) return null;
  return { id: layerId, i0: blend.i0, i1: blend.i1, t: blend.t };
}
