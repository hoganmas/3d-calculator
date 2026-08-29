/**
 * Dens/iso/flow keyframe cache for animated free parameters.
 */

import { fitVectorField } from "../math/fitVector.js";
import {
  bakeScalarKeyframeFrame,
  lobattoLadderDegrees,
  startLadderDeg,
  type LobattoFitState,
} from "../math/chebLobatto.js";
import { getParam } from "./params.js";
import type { CompiledExpr, CompiledVectorExpr, KeyframeFrame } from "../types/models.js";

export const DEFAULT_KEYFRAME_K = 8;

type KeyframeRole = "cloud" | "isosurface" | "flow";

interface BakeStages {
  sampleMs: number;
  chebMs: number;
  idctMs: number;
  gradMs: number;
}

interface BakeFrameOpts {
  layerId: string;
  latex: string;
  role: KeyframeRole;
  isoLevel: number;
  paramName: string;
  compiled?: { bind: CompiledExpr["bind"] };
  vectorCompiled?: CompiledVectorExpr;
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
  targetDeg: number;
  half: number;
  role: KeyframeRole;
  isoLevel: number;
  latex: string;
  frames: (KeyframeFrame | null)[];
  frameDeg: number[];
  lobattoByK: Map<number, LobattoFitState>;
  scratch: KeyframeFrame;
  gen: number;
  readyCount: number;
  bakeOpts: BakeFrameOpts;
}

type KeyframeWork =
  | { kind: "coarse"; k: number }
  | { kind: "refine"; k: number; nextDeg: number };

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
  role: KeyframeRole;
  isoLevel?: number;
  paramName: string;
  compiled?: { bind: CompiledExpr["bind"] };
  vectorCompiled?: CompiledVectorExpr;
  baseParams: Record<string, number>;
  half: number;
  deg: number;
  K?: number;
  /** Skip sync blend bake; async pump fills after render (anim ticks). */
  deferSyncBake?: boolean;
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

const asyncJobs = new Map<string, { gen: number; cancelled: boolean }>();
const pendingPumpLayers = new Set<string>();

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
  asyncJobs.delete(layerId);
  pendingPumpLayers.delete(layerId);
}

export function clearKeyframeCaches() {
  for (const id of [...asyncJobs.keys()]) cancelAsyncJob(id);
  pendingPumpLayers.clear();
  caches.clear();
  lastBakeMs = 0;
  lastLerpMs = 0;
  lastKfLayers = 0;
  bakeMsAcc = 0;
  lerpMsAcc = 0;
  lastBakeDetails = [];
}

export function hasActiveKeyframeCaches() {
  return caches.size > 0;
}

export function allKeyframesComplete() {
  if (!caches.size) return true;
  for (const cache of caches.values()) {
    if (cache.readyCount < cache.K) return false;
  }
  return true;
}

/** Test / debug: per-slot degree progress for a cached layer. */
export function getKeyframeProgress(layerId: string) {
  const cache = caches.get(layerId);
  if (!cache) return null;
  return {
    frameDeg: cache.frameDeg.slice(),
    targetDeg: cache.targetDeg,
    readyCount: cache.readyCount,
    K: cache.K,
  };
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
  for (let i = 0; i < n; i++) out[i] = u * a[i]! + t * b[i]!;
  return out;
}

function frameVolumeN(frame: KeyframeFrame): number {
  if (frame.dens?.length) return frame.dens.length;
  if (frame.fx?.length) return frame.fx.length;
  return 0;
}

function gridMFromFrame(frame: KeyframeFrame): number {
  const n = frameVolumeN(frame);
  return n > 0 ? Math.round(Math.cbrt(n)) : 0;
}

function allocScratch(role: KeyframeRole, n: number): KeyframeFrame {
  if (role === "flow") {
    return {
      fx: new Float32Array(n),
      fy: new Float32Array(n),
      fz: new Float32Array(n),
    };
  }
  const scratch: KeyframeFrame = { dens: new Float32Array(n) };
  if (role === "isosurface") {
    scratch.gx = new Float32Array(n);
    scratch.gy = new Float32Array(n);
    scratch.gz = new Float32Array(n);
  }
  return scratch;
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

function paramValueForK(opts: BakeFrameOpts, k: number): number {
  const st = getParam(opts.paramName);
  const min = st?.min ?? opts.min ?? 0;
  const max = st?.max ?? opts.max ?? 1;
  const K = opts.K ?? DEFAULT_KEYFRAME_K;
  return min + ((max - min) * k) / Math.max(1, K - 1);
}

function bakeFlowFrameAtDeg(
  opts: BakeFrameOpts,
  k: number,
  deg: number,
): KeyframeFrame {
  if (!opts.vectorCompiled) throw new Error("flow keyframes require vectorCompiled");
  const params = { ...opts.baseParams, [opts.paramName]: paramValueForK(opts, k) };
  const vectorFn = opts.vectorCompiled.bind(params);
  const fit = fitVectorField(opts.vectorCompiled, vectorFn, opts.half, deg, {
    skipL2: true,
    params,
  });
  return {
    fx: fit.fx,
    fy: fit.fy,
    fz: fit.fz,
    cheb: fit.cheb,
    fitRel: fit.fitRel,
  };
}

function bakeFrameAtDeg(
  cache: LayerKeyframeCache,
  k: number,
  deg: number,
  stages: BakeStages | null,
): KeyframeFrame {
  const opts = cache.bakeOpts;
  if (opts.role === "flow") {
    return bakeFlowFrameAtDeg(opts, k, deg);
  }
  if (!opts.compiled) throw new Error("scalar keyframes require compiled");
  const params = { ...opts.baseParams, [opts.paramName]: paramValueForK(opts, k) };
  const fn = opts.compiled.bind(params);
  const role = opts.role === "isosurface" ? "isosurface" : "cloud";
  const baked = bakeScalarKeyframeFrame(
    fn,
    opts.half,
    deg,
    role,
    cache.lobattoByK.get(k) ?? null,
    stages,
  );
  cache.lobattoByK.set(k, baked.lobatto);
  const frame: KeyframeFrame = {
    dens: baked.frame.dens,
    cheb: baked.frame.cheb,
    fitRel: baked.frame.fitRel,
  };
  if (role === "isosurface") {
    frame.gx = baked.frame.gx;
    frame.gy = baked.frame.gy;
    frame.gz = baked.frame.gz;
  }
  return frame;
}

function updateFrame(cache: LayerKeyframeCache, k: number, frame: KeyframeFrame, deg: number) {
  const prevDeg = cache.frameDeg[k] ?? 0;
  cache.frames[k] = frame;
  cache.frameDeg[k] = deg;
  if (prevDeg === cache.targetDeg && deg !== cache.targetDeg) cache.readyCount--;
  if (prevDeg !== cache.targetDeg && deg === cache.targetDeg) cache.readyCount++;
}

function pickNextKeyframeWork(cache: LayerKeyframeCache): KeyframeWork[] {
  const st = getParam(cache.paramName);
  const value = st?.value ?? cache.min;
  const { i0, i1 } = segmentForValue(cache, value);
  const target = cache.targetDeg;
  const ladder = lobattoLadderDegrees(target);
  const start = ladder[0] ?? startLadderDeg(target);
  const order = bakeOrder(cache.K, i0, i1);
  const degAt = (k: number) => cache.frameDeg[k] ?? 0;

  const workForSlot = (k: number, phaseDeg: number): KeyframeWork | null => {
    const d = degAt(k);
    if (d === 0) {
      if (phaseDeg !== start) return null;
      return { kind: "coarse", k };
    }
    if (d < phaseDeg) return { kind: "refine", k, nextDeg: phaseDeg };
    return null;
  };

  // Outer loop: degree ladder. Inner loop: K slots (blend pair first via bakeOrder).
  for (const phaseDeg of ladder) {
    const blendWorks: KeyframeWork[] = [];
    for (const k of [i0, i1]) {
      const w = workForSlot(k, phaseDeg);
      if (w) blendWorks.push(w);
    }
    if (blendWorks.length) return [blendWorks[0]!];

    for (const k of order) {
      if (k === i0 || k === i1) continue;
      const w = workForSlot(k, phaseDeg);
      if (w) return [w];
    }
  }

  return [];
}

function executeKeyframeWork(
  cache: LayerKeyframeCache,
  work: KeyframeWork,
  stages: BakeStages | null,
): { frame: KeyframeFrame; k: number; deg: number } {
  const deg = work.kind === "coarse" ? startLadderDeg(cache.targetDeg) : work.nextDeg;
  const frame = bakeFrameAtDeg(cache, work.k, deg, stages);
  updateFrame(cache, work.k, frame, deg);
  return { frame, k: work.k, deg };
}

function keyframesFullyReady(cache: LayerKeyframeCache): boolean {
  return cache.readyCount >= cache.K;
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

function runOneKeyframeWork(): boolean {
  for (const layerId of pendingPumpLayers) {
    const cache = caches.get(layerId);
    const job = asyncJobs.get(layerId);
    if (!cache || !job || job.cancelled || job.gen !== cache.gen) {
      pendingPumpLayers.delete(layerId);
      asyncJobs.delete(layerId);
      continue;
    }

    const works = pickNextKeyframeWork(cache);
    if (!works.length) {
      pendingPumpLayers.delete(layerId);
      asyncJobs.delete(layerId);
      onKeyframeProgress?.({
        layerId,
        index: -1,
        frame: cache.frames.find(Boolean) ?? cache.frames[0]!,
        readyCount: cache.readyCount,
        K: cache.K,
        done: keyframesFullyReady(cache),
      });
      continue;
    }

    const work = works[0]!;
    const t0 = performance.now();
    const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
    const { frame, k, deg } = executeKeyframeWork(cache, work, stages);
    onKeyframeProgress?.({
      layerId,
      index: k,
      frame,
      readyCount: cache.readyCount,
      K: cache.K,
      done: keyframesFullyReady(cache),
    });
    const bakeMs = performance.now() - t0;
    bakeMsAcc += bakeMs;
    lastBakeMs = bakeMsAcc;
    lastBakeDetails.push({
      ...stages,
      frames: 1,
      deg,
      M: Math.round(Math.cbrt(frameVolumeN(frame))),
      role: cache.role,
      layerId,
      paramName: cache.paramName,
      bakeMs,
      async: true,
      index: k,
    });

    if (keyframesFullyReady(cache)) {
      pendingPumpLayers.delete(layerId);
      asyncJobs.delete(layerId);
      onKeyframeProgress?.({
        layerId,
        index: -1,
        frame,
        readyCount: cache.readyCount,
        K: cache.K,
        done: true,
      });
    }
    return true;
  }
  return false;
}

/** Run async keyframe work after render; default one unit per frame. */
export function tickKeyframePump(maxWorks = 1): number {
  let done = 0;
  while (done < maxWorks && runOneKeyframeWork()) done++;
  return done;
}

/**
 * @param {string} layerId
 * @param {LayerKeyframeCache} cache
 */
function scheduleAsyncFill(layerId: string, cache: LayerKeyframeCache) {
  cancelAsyncJob(layerId);
  const gen = cache.gen;
  asyncJobs.set(layerId, { gen, cancelled: false });
  pendingPumpLayers.add(layerId);
}

/**
 * Bake or reuse keyframes; return GPU blend indices (no dens lerp).
 * Sync-bakes only the current blend pair; remaining frames fill asynchronously.
 *
 * @param {{
 *   layerId: string,
 *   latex: string,
 *   role: "cloud" | "isosurface",
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
      targetDeg: opts.deg,
      frames: new Array(K).fill(null),
      frameDeg: new Array(K).fill(0),
      lobattoByK: new Map(),
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
        vectorCompiled: opts.vectorCompiled,
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
      vectorCompiled: opts.vectorCompiled,
    };
  }

  const blend = segmentForValue(cache, value);
  const t0 = performance.now();
  const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
  const startDeg = startLadderDeg(cache.targetDeg);
  let syncCount = 0;
  if (!opts.deferSyncBake) {
    for (const k of [blend.i0, blend.i1]) {
      if (cache.frameDeg[k]! > 0) continue;
      const frame = bakeFrameAtDeg(cache, k, startDeg, stages);
      updateFrame(cache, k, frame, startDeg);
      syncCount++;
      baked = true;
    }
  }

  // Allocate scratch once we know volume size.
  const proto = cache.frames[blend.i0] || cache.frames[blend.i1];
  if (proto) {
    const n = frameVolumeN(proto);
    const scratchN = frameVolumeN(cache.scratch);
    if (n > 0 && scratchN !== n) {
      cache.scratch = allocScratch(opts.role, n);
    }
  }

  if (syncCount > 0) {
    const bakeMs = performance.now() - t0;
    bakeMsAcc += bakeMs;
    lastBakeMs = bakeMsAcc;
    lastBakeDetails.push({
      ...stages,
      frames: syncCount,
      deg: startDeg,
      M: gridMFromFrame(proto ?? { dens: new Float32Array(0) }),
      role: opts.role,
      layerId: opts.layerId,
      paramName: opts.paramName,
      bakeMs,
      async: false,
      syncPair: [blend.i0, blend.i1],
    });
  }

  if (!keyframesFullyReady(cache)) {
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
    M: gridMFromFrame(a),
    baked,
    readyCount: cache.readyCount,
    complete: keyframesFullyReady(cache),
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
  if (!a.dens || !b.dens) throw new Error("scalar keyframe pair missing dens");
  const out = cache.scratch;
  lerpFloat32(a.dens, b.dens, t, out.dens!);
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  return {
    dens: out.dens!,
    cheb: ensured.cheb,
    fitRel: ensured.fitRel,
    M: ensured.M,
    baked: ensured.baked,
    readyCount: ensured.readyCount,
    complete: ensured.complete,
    frames: ensured.frames,
  };
}

/** CPU-lerp fx/fy/fz velocity grids at the current param value. */
export function sampleFlowLayerKeyframes(opts: EnsureKeyframesOpts) {
  if (opts.role !== "flow") throw new Error("sampleFlowLayerKeyframes requires role flow");
  const ensured = ensureLayerKeyframes(opts);
  const cache = caches.get(opts.layerId);
  if (!cache) throw new Error("keyframe cache missing after ensure");
  const tLerp = performance.now();
  const { i0, i1, t } = ensured.blend;
  const a = cache.frames[i0];
  const b = cache.frames[i1];
  if (!a || !b) throw new Error("sync keyframe pair missing");
  if (!a.fx || !a.fy || !a.fz || !b.fx || !b.fy || !b.fz) {
    throw new Error("flow keyframe pair missing velocity grids");
  }
  const out = cache.scratch;
  lerpFloat32(a.fx, b.fx, t, out.fx!);
  lerpFloat32(a.fy, b.fy, t, out.fy!);
  lerpFloat32(a.fz, b.fz, t, out.fz!);
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  return {
    fx: out.fx!.slice(),
    fy: out.fy!.slice(),
    fz: out.fz!.slice(),
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
