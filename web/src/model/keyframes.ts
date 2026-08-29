/**
 * Dens/iso/flow keyframe cache for animated free parameters.
 */

import { fitVectorField } from "../math/fitVector.js";
import {
  bakeScalarKeyframeFrameChunked,
  lobattoLadderDegrees,
  startLadderDeg,
  type LobattoBuildJob,
  type LobattoFinalizeJob,
  type LobattoFinalizePhase,
  type LobattoFitState,
} from "../math/chebLobatto.js";
import { getParam } from "./params.js";
import { gridMFromDens, tearLog, tearLogOnce } from "../app/tearDebug.js";
import type { CompiledExpr, CompiledVectorExpr, KeyframeFrame } from "../types/models.js";

export const DEFAULT_KEYFRAME_K = 8;
/** Max Lobatto work (sample + IDCT + grad) per render frame during async fill. */
export const KEYFRAME_LOBATTO_BUDGET_MS = 3;

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
  /** Baked frames waiting for blend partner to match degree before display. */
  stagingFrames: (KeyframeFrame | null)[];
  stagingDeg: number[];
  lobattoByK: Map<number, LobattoFitState>;
  lobattoJobByK: Map<number, LobattoBuildJob>;
  lobattoFinalizeByK: Map<number, LobattoFinalizeJob>;
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
  /** Blend-pair promote: update all listed slots then full GPU upload (iso). */
  promoted?: number[];
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

export function getKeyframeLayerRole(layerId: string): KeyframeRole | null {
  return caches.get(layerId)?.role ?? null;
}

/**
 * Copy display keyframes into lastSceneBake at the active blend pair's grid M.
 * Off-blend slots at other resolutions are filled from the nearest same-M frame.
 */
export function syncIsoKeyframesToSceneBake(
  layerId: string,
  sceneBake: { isosurfaceLayers: { id?: string; keyframes?: KeyframeFrame[]; blend?: { i0: number; i1: number; t: number } }[]; M: number },
  uploadM?: number,
): number {
  const cache = caches.get(layerId);
  if (!cache || cache.role !== "isosurface") return sceneBake.M;
  const layer = sceneBake.isosurfaceLayers.find((x) => x.id === layerId);
  if (!layer) return sceneBake.M;
  const st = getParam(cache.paramName);
  const value = st?.value ?? cache.min;
  const blend = displayBlendForValue(cache, value);
  layer.blend = { i0: blend.i0, i1: blend.i1, t: blend.t };
  const blendM = isoBlendSceneM(cache, blend.i0, blend.i1, blend.t);
  const M = uploadM ?? Math.max(sceneBake.M, blendM);
  layer.keyframes = materializeKeyframeFramesAtM(cache.frames, M);
  sceneBake.M = M;
  return blendM;
}

/** Grid M for the current iso blend pair (matches displayBlendForValue snap semantics). */
export function getIsoBlendSceneM(layerId: string): number {
  const cache = caches.get(layerId);
  if (!cache || cache.role !== "isosurface") return 0;
  const st = getParam(cache.paramName);
  const value = st?.value ?? cache.min;
  const blend = displayBlendForValue(cache, value);
  return isoBlendSceneM(cache, blend.i0, blend.i1, blend.t);
}

/** Promote staged pair when both slots share a staging degree (display path is read-only). */
export function refreshIsoBlendDisplay(layerId: string): number[] {
  const cache = caches.get(layerId);
  if (!cache || cache.role !== "isosurface") return [];
  const st = getParam(cache.paramName);
  if (!st) return [];
  const { i0, i1 } = segmentForValue(cache, st.value);
  return reconcileStaging(cache, i0, i1);
}

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

/** Layers kept for re-enable reuse; excluded from load bar / completion / splash. */
const parkedKeyframeLayers = new Set<string>();

function dropLayerKeyframeCache(layerId: string) {
  cancelAsyncJob(layerId);
  caches.delete(layerId);
  parkedKeyframeLayers.delete(layerId);
}

function parkLayerKeyframeCache(layerId: string) {
  cancelAsyncJob(layerId);
  if (caches.has(layerId)) parkedKeyframeLayers.add(layerId);
}

function unparkLayerKeyframeCache(layerId: string) {
  parkedKeyframeLayers.delete(layerId);
  // Resume bake only via ensureLayerKeyframes (anim / first need), not on visibility alone.
}

function isParkedKeyframeLayer(layerId: string) {
  return parkedKeyframeLayers.has(layerId);
}

export function clearKeyframeCaches() {
  for (const id of [...asyncJobs.keys()]) cancelAsyncJob(id);
  pendingPumpLayers.clear();
  caches.clear();
  parkedKeyframeLayers.clear();
  lastBakeMs = 0;
  lastLerpMs = 0;
  lastKfLayers = 0;
  bakeMsAcc = 0;
  lerpMsAcc = 0;
  lastBakeDetails = [];
}

/**
 * Keep keyframe caches across visibility toggles / structural refits.
 * - Deleted or latex/deg/half-changed expressions: discard (no bake until enabled + needed).
 * - Disabled but unchanged: park (pause async; reuse on re-enable).
 * - Enabled and unchanged: unpark (resume bake only when ensureLayerKeyframes runs).
 *
 * Callers must pass the **UI target** degree (not a progressive ladder step).
 * Progressive dens refits after pause would otherwise wipe animation caches.
 */
export function syncKeyframeCachesWithExpressions(
  items: ReadonlyArray<{ id: string; latex: string; enabled: boolean }>,
  scene: { deg: number; half: number },
) {
  const byId = new Map(items.map((i) => [i.id, i]));
  for (const layerId of [...caches.keys()]) {
    const item = byId.get(layerId);
    const cache = caches.get(layerId);
    if (!cache) continue;
    if (!item || !String(item.latex ?? "").trim()) {
      dropLayerKeyframeCache(layerId);
      continue;
    }
    const stale =
      cache.latex !== item.latex ||
      cache.deg !== scene.deg ||
      Math.abs(cache.half - scene.half) >= 1e-12;
    if (stale) {
      dropLayerKeyframeCache(layerId);
      continue;
    }
    if (!item.enabled) parkLayerKeyframeCache(layerId);
    else unparkLayerKeyframeCache(layerId);
  }
}

export function hasActiveKeyframeCaches() {
  for (const id of caches.keys()) {
    if (!isParkedKeyframeLayer(id)) return true;
  }
  return false;
}

export function allKeyframesComplete() {
  if (!hasActiveKeyframeCaches()) return true;
  for (const [id, cache] of caches) {
    if (isParkedKeyframeLayer(id)) continue;
    if (cache.readyCount < cache.K) return false;
  }
  return true;
}

/** Coarse blend pair ready for every active (enabled) cached layer (splash / first frame). */
export function keyframesSplashReady(): boolean {
  if (!hasActiveKeyframeCaches()) return true;
  for (const [id, cache] of caches) {
    if (isParkedKeyframeLayer(id)) continue;
    if (peekKeyframeBlend(cache.bakeOpts.layerId) == null) return false;
  }
  return true;
}

/** Test / debug: per-slot degree progress for a cached layer. */
export function getKeyframeProgress(layerId: string) {
  const cache = caches.get(layerId);
  if (!cache) return null;
  return {
    /** max(display, staging) — used by scheduler; can look "done" while display lags. */
    frameDeg: cache.frameDeg.map((_, k) => bakedDegAt(cache, k)),
    displayDeg: cache.frameDeg.slice(),
    stagingDeg: cache.stagingDeg.slice(),
    targetDeg: cache.targetDeg,
    readyCount: cache.readyCount,
    K: cache.K,
  };
}

export interface KeyframeStallSlotDiag {
  k: number;
  displayDeg: number;
  stagingDeg: number;
  schedDeg: number;
  inFlight: boolean;
  finalizePhase?: string;
}

export interface KeyframeStallDiag {
  layerId: string;
  role: KeyframeRole;
  targetDeg: number;
  readyCount: number;
  K: number;
  parked: boolean;
  blend: { i0: number; i1: number; value: number };
  pending: boolean;
  worksQueued: number;
  slots: KeyframeStallSlotDiag[];
  /** True when pump would stop but display is not fully ready. */
  stalled: boolean;
}

/** Debug: dump why a layer may never reach complete. */
export function diagnoseKeyframeCaches(): KeyframeStallDiag[] {
  const out: KeyframeStallDiag[] = [];
  for (const [layerId, cache] of caches) {
    const parked = isParkedKeyframeLayer(layerId);
    const st = getParam(cache.paramName);
    const value = st?.value ?? cache.min;
    const { i0, i1 } = segmentForValue(cache, value);
    const works = parked ? [] : peekPickNextWork(cache);
    const slots: KeyframeStallSlotDiag[] = [];
    for (let k = 0; k < cache.K; k++) {
      const fin = cache.lobattoFinalizeByK.get(k);
      slots.push({
        k,
        displayDeg: cache.frameDeg[k] ?? 0,
        stagingDeg: cache.stagingDeg[k] ?? 0,
        schedDeg: bakedDegAt(cache, k),
        inFlight: cache.lobattoJobByK.has(k) || cache.lobattoFinalizeByK.has(k),
        finalizePhase: fin?.phase,
      });
    }
    const ready = keyframesFullyReady(cache);
    out.push({
      layerId,
      role: cache.role,
      targetDeg: cache.targetDeg,
      readyCount: cache.readyCount,
      K: cache.K,
      parked,
      blend: { i0, i1, value },
      pending: pendingPumpLayers.has(layerId),
      worksQueued: works.length,
      slots,
      stalled: !parked && !ready && works.length === 0,
    });
  }
  return out;
}

/** Same as pickNext but does not mutate (reconcile is idempotent / safe). */
function peekPickNextWork(cache: LayerKeyframeCache): KeyframeWork[] {
  return pickNextKeyframeWork(cache);
}

export interface KeyframeLoadSummary {
  /** 0–1 aggregate load across all cached layers and K slots. */
  fraction: number;
  complete: boolean;
  /** True when keyframe caches exist and fill is not finished. */
  active: boolean;
  slotsAtTarget: number;
  slotsTotal: number;
  layerCount: number;
  label: string;
}

function slotLoadFraction(cache: LayerKeyframeCache, k: number): number {
  const target = cache.targetDeg;
  if (target <= 0) return 1;
  const d = cache.frameDeg[k] ?? 0;
  if (d >= target) return 1;
  if (d <= 0) return 0;
  const ladder = lobattoLadderDegrees(target);
  let rung = -1;
  for (let i = 0; i < ladder.length; i++) {
    if (d >= ladder[i]!) rung = i;
  }
  if (rung < 0) return 0;
  return (rung + 1) / ladder.length;
}

/** Aggregate animation keyframe load for UI progress bars. */
export function getKeyframeLoadSummary(): KeyframeLoadSummary {
  if (!hasActiveKeyframeCaches()) {
    return {
      fraction: 1,
      complete: true,
      active: false,
      slotsAtTarget: 0,
      slotsTotal: 0,
      layerCount: 0,
      label: "",
    };
  }
  let sum = 0;
  let slotsTotal = 0;
  let slotsAtTarget = 0;
  let layerCount = 0;
  for (const [id, cache] of caches) {
    if (isParkedKeyframeLayer(id)) continue;
    layerCount++;
    slotsTotal += cache.K;
    slotsAtTarget += cache.readyCount;
    for (let k = 0; k < cache.K; k++) sum += slotLoadFraction(cache, k);
  }
  const fraction = slotsTotal > 0 ? sum / slotsTotal : 1;
  const complete = allKeyframesComplete();
  const pct = Math.min(100, Math.max(0, Math.round(fraction * 100)));
  const label = complete
    ? ""
    : `Loading animation · ${pct}% (${slotsAtTarget}/${slotsTotal} at full quality)`;
  return {
    fraction,
    complete,
    active: !complete,
    slotsAtTarget,
    slotsTotal,
    layerCount,
    label,
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
  if (a.length !== b.length) {
    const src = t < 0.5 ? a : b;
    const n = Math.min(src.length, out.length);
    out.fill(0);
    if (n > 0) out.set(src.subarray(0, n));
    return out;
  }
  const n = Math.min(a.length, b.length, out.length);
  const u = 1 - t;
  for (let i = 0; i < n; i++) out[i] = u * a[i]! + t * b[i]!;
  return out;
}

function ensureScratchVolume(cache: LayerKeyframeCache, n: number) {
  if (n <= 0) return;
  if (frameVolumeN(cache.scratch) !== n) {
    cache.scratch = allocScratch(cache.role, n);
  }
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
  lobattoBudgetMs: number | null = KEYFRAME_LOBATTO_BUDGET_MS,
): { frame: KeyframeFrame | null; complete: boolean; phase?: LobattoFinalizePhase | "sample" } {
  const opts = cache.bakeOpts;
  if (opts.role === "flow") {
    return { frame: bakeFlowFrameAtDeg(opts, k, deg), complete: true };
  }
  if (!opts.compiled) throw new Error("scalar keyframes require compiled");
  const params = { ...opts.baseParams, [opts.paramName]: paramValueForK(opts, k) };
  const fn = opts.compiled.bind(params);
  const role = opts.role === "isosurface" ? "isosurface" : "cloud";
  const existingJob = cache.lobattoJobByK.get(k) ?? null;
  const existingFinalize = cache.lobattoFinalizeByK.get(k) ?? null;
  const baked = bakeScalarKeyframeFrameChunked(
    fn,
    opts.half,
    deg,
    role,
    cache.lobattoByK.get(k) ?? null,
    existingJob,
    stages,
    lobattoBudgetMs,
    existingFinalize,
  );
  if (baked.job) cache.lobattoJobByK.set(k, baked.job);
  else cache.lobattoJobByK.delete(k);
  if (baked.finalizeJob) {
    cache.lobattoFinalizeByK.set(k, baked.finalizeJob);
    if (!existingFinalize) cache.lobattoByK.set(k, baked.finalizeJob.lob);
  } else cache.lobattoFinalizeByK.delete(k);
  if (!baked.complete || !baked.result) {
    return {
      frame: cache.frames[k],
      complete: false,
      phase: baked.finalizePhase ?? (baked.job ? "sample" : undefined),
    };
  }
  cache.lobattoByK.set(k, baked.result.lobatto);
  const frame: KeyframeFrame = {
    dens: baked.result.frame.dens,
    cheb: baked.result.frame.cheb,
    fitRel: baked.result.frame.fitRel,
  };
  if (role === "isosurface") {
    frame.gx = baked.result.frame.gx;
    frame.gy = baked.result.frame.gy;
    frame.gz = baked.result.frame.gz;
  }
  return { frame, complete: true };
}

function stageOffBlendFrame(
  cache: LayerKeyframeCache,
  k: number,
  frame: KeyframeFrame,
  deg: number,
) {
  const displayDeg = cache.frameDeg[k] ?? 0;
  if (deg <= displayDeg) return;
  applyDisplayFrame(cache, k, frame, deg);
}

function clearCoarseStaging(cache: LayerKeyframeCache, k: number) {
  const sd = cache.stagingDeg[k] ?? 0;
  const dd = cache.frameDeg[k] ?? 0;
  if (sd > 0 && sd < dd) {
    cache.stagingFrames[k] = null;
    cache.stagingDeg[k] = 0;
  }
}

function isoBlendSceneM(
  cache: LayerKeyframeCache,
  i0: number,
  i1: number,
  t: number,
): number {
  const d0 = cache.frameDeg[i0] ?? 0;
  const d1 = cache.frameDeg[i1] ?? 0;
  const M0 = gridMFromFrame(cache.frames[i0] ?? {});
  const M1 = gridMFromFrame(cache.frames[i1] ?? {});
  if (d0 === d1 && M0 > 0 && M0 === M1) return M0;
  if (d0 > d1 || (d0 > 0 && d1 <= 0)) return M0 || M1 || 2;
  if (d1 > d0 || (d1 > 0 && d0 <= 0)) return M1 || M0 || 2;
  if (M0 > 0 && M0 === M1) return M0;
  return t < 0.5 ? M0 || M1 || 2 : M1 || M0 || 2;
}

/** Promote when both blend slots share the same staging degree. */
function tryPromoteStagingPair(cache: LayerKeyframeCache, i0: number, i1: number): number[] {
  const sd0 = cache.stagingDeg[i0] ?? 0;
  const sd1 = cache.stagingDeg[i1] ?? 0;
  if (sd0 <= 0 || sd0 !== sd1 || !cache.stagingFrames[i0] || !cache.stagingFrames[i1]) {
    return [];
  }
  const promoted: number[] = [];
  for (const slot of [i0, i1]) {
    if ((cache.frameDeg[slot] ?? 0) < sd0) {
      applyDisplayFrame(cache, slot, cache.stagingFrames[slot]!, sd0);
      cache.stagingFrames[slot] = null;
      cache.stagingDeg[slot] = 0;
      promoted.push(slot);
    }
  }
  if (promoted.length) {
    tearLog("promote-staging-pair", {
      layerId: cache.bakeOpts.layerId,
      i0,
      i1,
      deg: sd0,
      promoted,
    });
  }
  return promoted;
}

/**
 * Flush staging that can never lockstep-promote:
 * - Off-blend slots: staging → display immediately (no tear risk).
 * - Blend slot whose partner already displays the same degree: solo promote.
 * Without this, schedDeg looks done while readyCount never reaches K (pump stalls).
 */
function reconcileStaging(cache: LayerKeyframeCache, i0: number, i1: number): number[] {
  const promoted: number[] = [];
  for (let k = 0; k < cache.K; k++) {
    if (k === i0 || k === i1) continue;
    const sd = cache.stagingDeg[k] ?? 0;
    const fr = cache.stagingFrames[k];
    if (sd > (cache.frameDeg[k] ?? 0) && fr) {
      applyDisplayFrame(cache, k, fr, sd);
      promoted.push(k);
    }
  }
  for (const [k, partner] of [
    [i0, i1],
    [i1, i0],
  ] as const) {
    const sd = cache.stagingDeg[k] ?? 0;
    const fr = cache.stagingFrames[k];
    if (!fr || sd <= (cache.frameDeg[k] ?? 0)) continue;
    if ((cache.frameDeg[partner] ?? 0) === sd) {
      applyDisplayFrame(cache, k, fr, sd);
      tearLog("reconcile-solo-promote", {
        layerId: cache.bakeOpts.layerId,
        k,
        partner,
        deg: sd,
      });
      promoted.push(k);
    }
  }
  promoted.push(...tryPromoteStagingPair(cache, i0, i1));
  return promoted;
}

function bakedDegAt(cache: LayerKeyframeCache, k: number): number {
  return Math.max(cache.frameDeg[k] ?? 0, cache.stagingDeg[k] ?? 0);
}

function slotSummary(cache: LayerKeyframeCache, k: number) {
  const fr = cache.frames[k];
  const st = cache.stagingFrames[k];
  return {
    k,
    displayDeg: cache.frameDeg[k] ?? 0,
    stagingDeg: cache.stagingDeg[k] ?? 0,
    displayM: gridMFromDens(fr?.dens),
    stagingM: gridMFromDens(st?.dens),
  };
}

function applyDisplayFrame(
  cache: LayerKeyframeCache,
  k: number,
  frame: KeyframeFrame,
  deg: number,
) {
  const prevDeg = cache.frameDeg[k] ?? 0;
  if (deg < prevDeg) {
    tearLog("display-regression-blocked", {
      layerId: cache.bakeOpts.layerId,
      k,
      deg,
      prevDeg,
    });
    return;
  }
  if (deg === prevDeg) return;
  cache.frames[k] = frame;
  cache.frameDeg[k] = deg;
  cache.stagingFrames[k] = null;
  cache.stagingDeg[k] = 0;
  if (prevDeg !== cache.targetDeg && deg === cache.targetDeg) cache.readyCount++;
  tearLog("display-frame", {
    layerId: cache.bakeOpts.layerId,
    ...slotSummary(cache, k),
    deg,
    prevDeg,
  });
}

/** Commit a baked frame; blend-pair slots promote to display only when both match degree. */
function commitFrame(
  cache: LayerKeyframeCache,
  k: number,
  frame: KeyframeFrame,
  deg: number,
): number[] {
  const st = getParam(cache.paramName);
  const value = st?.value ?? cache.min;
  const { i0, i1 } = segmentForValue(cache, value);

  if (k !== i0 && k !== i1) {
    if (deg <= (cache.frameDeg[k] ?? 0)) {
      tearLog("commit-stale-skip", {
        layerId: cache.bakeOpts.layerId,
        k,
        deg,
        displayDeg: cache.frameDeg[k] ?? 0,
      });
      return [];
    }
    stageOffBlendFrame(cache, k, frame, deg);
    tearLog("commit-off-blend-display", {
      layerId: cache.bakeOpts.layerId,
      k,
      deg,
      blend: { i0, i1 },
      M: gridMFromDens(frame.dens),
    });
    return [];
  }

  if (deg <= (cache.frameDeg[k] ?? 0)) {
    tearLog("commit-stale-skip", {
      layerId: cache.bakeOpts.layerId,
      k,
      deg,
      displayDeg: cache.frameDeg[k] ?? 0,
    });
    return [];
  }
  clearCoarseStaging(cache, i0);
  clearCoarseStaging(cache, i1);

  cache.stagingFrames[k] = frame;
  cache.stagingDeg[k] = deg;
  const partner = k === i0 ? i1 : i0;
  const maxDisplay = Math.max(cache.frameDeg[i0] ?? 0, cache.frameDeg[i1] ?? 0);
  if (cache.stagingDeg[partner] === deg && cache.stagingFrames[partner]) {
    if (deg < maxDisplay) {
      cache.stagingFrames[k] = null;
      cache.stagingDeg[k] = 0;
      tearLog("commit-promote-blocked", {
        layerId: cache.bakeOpts.layerId,
        k,
        deg,
        maxDisplay,
        slots: [i0, i1].map((s) => slotSummary(cache, s)),
      });
      return [];
    }
    const promoted: number[] = [];
    for (const slot of [i0, i1]) {
      applyDisplayFrame(cache, slot, cache.stagingFrames[slot]!, deg);
      cache.stagingFrames[slot] = null;
      cache.stagingDeg[slot] = 0;
      promoted.push(slot);
    }
    tearLog("commit-promote", {
      layerId: cache.bakeOpts.layerId,
      k,
      deg,
      promoted,
      slots: [i0, i1].map((s) => slotSummary(cache, s)),
    });
    return promoted;
  }
  if ((cache.frameDeg[partner] ?? 0) === deg && cache.frames[partner] && deg >= (cache.frameDeg[k] ?? 0)) {
    applyDisplayFrame(cache, k, frame, deg);
    tearLog("commit-promote-solo", {
      layerId: cache.bakeOpts.layerId,
      k,
      deg,
      partner,
      M: gridMFromDens(frame.dens),
    });
    return [k];
  }
  tearLog("commit-staged", {
    layerId: cache.bakeOpts.layerId,
    k,
    deg,
    partner,
    partnerStagingDeg: cache.stagingDeg[partner] ?? 0,
    slots: [i0, i1].map((s) => slotSummary(cache, s)),
  });
  return [];
}

/**
 * Blend segment for display/lerp — only interpolate when both slots share degree
 * and grid resolution; otherwise snap to the ready slot to avoid tearing.
 */
function displayBlendForValue(cache: LayerKeyframeCache, value: number) {
  const seg = segmentForValue(cache, value);
  const { i0, i1, t } = seg;
  const d0 = cache.frameDeg[i0] ?? 0;
  const d1 = cache.frameDeg[i1] ?? 0;
  const a = cache.frames[i0];
  const b = cache.frames[i1];
  if (d0 !== d1 || !a || !b) {
    const snap =
      d0 > 0 && d1 <= 0 ? { i0, i1, t: 0 } :
      d1 > 0 && d0 <= 0 ? { i0, i1, t: 1 } :
      d0 > d1 ? { i0, i1, t: 0 } :
      d1 > d0 ? { i0, i1, t: 1 } :
      seg;
    tearLogOnce(
      `blend-snap-deg-${cache.bakeOpts.layerId}-${d0}-${d1}`,
      "blend-snap",
      {
        layerId: cache.bakeOpts.layerId,
        reason: "deg-or-missing",
        d0,
        d1,
        M0: gridMFromDens(a?.dens),
        M1: gridMFromDens(b?.dens),
        segT: t,
        snapT: snap.t,
      },
    );
    return snap;
  }
  if (a.dens && b.dens && a.dens.length !== b.dens.length) {
    const snap = t < 0.5 ? { i0, i1, t: 0 } : { i0, i1, t: 1 };
    tearLogOnce(
      `blend-snap-len-${cache.bakeOpts.layerId}-${a.dens.length}-${b.dens.length}`,
      "blend-snap",
      {
        layerId: cache.bakeOpts.layerId,
        reason: "dens-length",
        d0,
        d1,
        M0: gridMFromDens(a.dens),
        M1: gridMFromDens(b.dens),
        segT: t,
        snapT: snap.t,
      },
    );
    return snap;
  }
  if (a.fx && b.fx && a.fx.length !== b.fx.length) {
    const snap = t < 0.5 ? { i0, i1, t: 0 } : { i0, i1, t: 1 };
    tearLogOnce(
      `blend-snap-fx-${cache.bakeOpts.layerId}`,
      "blend-snap",
      { layerId: cache.bakeOpts.layerId, reason: "fx-length", segT: t, snapT: snap.t },
    );
    return snap;
  }
  return seg;
}

function pickNextKeyframeWork(cache: LayerKeyframeCache): KeyframeWork[] {
  const st = getParam(cache.paramName);
  const value = st?.value ?? cache.min;
  const { i0, i1 } = segmentForValue(cache, value);
  reconcileStaging(cache, i0, i1);
  const target = cache.targetDeg;
  const ladder = lobattoLadderDegrees(target);
  const start = ladder[0] ?? startLadderDeg(target);
  const order = bakeOrder(cache.K, i0, i1);
  /** Display degree drives UI; staging counts for blend-pair scheduling only. */
  const schedDegAt = (k: number) =>
    Math.max(cache.frameDeg[k] ?? 0, cache.stagingDeg[k] ?? 0);

  for (const k of order) {
    if (cache.lobattoFinalizeByK.has(k)) {
      const fin = cache.lobattoFinalizeByK.get(k)!;
      return [{ kind: "refine", k, nextDeg: fin.lob.deg }];
    }
    if (cache.lobattoJobByK.has(k)) {
      const job = cache.lobattoJobByK.get(k)!;
      return [{ kind: "refine", k, nextDeg: job.targetDeg }];
    }
  }

  const workForSlot = (k: number, phaseDeg: number): KeyframeWork | null => {
    const d = schedDegAt(k);
    if (d === 0) {
      if (phaseDeg !== start) return null;
      return { kind: "coarse", k };
    }
    if (d < phaseDeg) return { kind: "refine", k, nextDeg: phaseDeg };
    return null;
  };

  // Degree-first globally: every slot reaches phaseDeg before any slot advances further.
  for (const phaseDeg of ladder) {
    for (const k of order) {
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
  lobattoBudgetMs: number | null = KEYFRAME_LOBATTO_BUDGET_MS,
): { frame: KeyframeFrame | null; k: number; deg: number; complete: boolean; promoted?: number[] } {
  const deg = work.kind === "coarse" ? startLadderDeg(cache.targetDeg) : work.nextDeg;
  const { frame, complete } = bakeFrameAtDeg(cache, work.k, deg, stages, lobattoBudgetMs);
  if (!complete || !frame) {
    return {
      frame: cache.frames[work.k],
      k: work.k,
      deg: cache.frameDeg[work.k] ?? 0,
      complete: false,
    };
  }
  const promoted = commitFrame(cache, work.k, frame, deg);
  return { frame, k: work.k, deg, complete: true, promoted };
}

function keyframesFullyReady(cache: LayerKeyframeCache): boolean {
  return cache.readyCount >= cache.K;
}

/** True when an in-memory keyframe cache exists (not parked / disabled). */
export function hasLayerKeyframeCache(layerId: string): boolean {
  return caches.has(layerId) && !isParkedKeyframeLayer(layerId);
}

type EnsureKeyframesResult = {
  frames: KeyframeFrame[];
  rawFrames: (KeyframeFrame | null)[];
  blend: { i0: number; i1: number; t: number };
  cheb?: Float32Array;
  fitRel?: number;
  M: number;
  baked: boolean;
  gpuUploadNeeded: boolean;
  readyCount: number;
  complete: boolean;
};

function buildLayerKeyframeResult(
  cache: LayerKeyframeCache,
  value: number,
  opts: EnsureKeyframesOpts,
  baked: boolean,
  syncPromoted: number[] = [],
): EnsureKeyframesResult {
  const displayBlend = displayBlendForValue(cache, value);
  const frames =
    cache.role === "isosurface"
      ? materializeKeyframeFramesAtM(
          cache.frames,
          isoBlendSceneM(cache, displayBlend.i0, displayBlend.i1, displayBlend.t),
        )
      : materializeKeyframeFrames(cache.frames);
  const a = frames[displayBlend.i0]!;
  const b = frames[displayBlend.i1]!;
  return {
    frames,
    rawFrames: cache.frames.slice(),
    blend: displayBlend,
    cheb: displayBlend.t < 0.5 ? a.cheb : b.cheb,
    fitRel: displayBlend.t < 0.5 ? a.fitRel : b.fitRel,
    M: gridMFromFrame(a),
    baked,
    gpuUploadNeeded:
      opts.role === "isosurface" &&
      syncPromoted.length > 0 &&
      syncPromoted.includes(displayBlend.i0) &&
      syncPromoted.includes(displayBlend.i1),
    readyCount: cache.readyCount,
    complete: keyframesFullyReady(cache),
  };
}

function maybeLogBlendPairReady(
  cache: LayerKeyframeCache,
  layerId: string,
  i0: number,
  i1: number,
  deg: number,
  bakeMs: number,
  mode: "sync" | "async",
  slot: number,
) {
  const d0 = cache.frameDeg[i0] ?? 0;
  const d1 = cache.frameDeg[i1] ?? 0;
  if (d0 !== deg || d1 !== deg) return;
  const atTarget = deg === cache.targetDeg;
  console.log(
    `[keyframes] blend pair ready · ${layerId} · ${cache.paramName} · ` +
      `(${i0},${i1}) deg ${deg}${atTarget ? " · target" : ""} · ` +
      `${mode} · slot ${slot} · ${bakeMs.toFixed(1)}ms`,
  );
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
  const sceneM = gridMFromFrame(last) || 2;
  return materializeKeyframeFramesAtM(frames, sceneM);
}

function frameAtGridM(fr: KeyframeFrame | null | undefined, sceneM: number): boolean {
  if (!fr) return false;
  return gridMFromFrame(fr) === sceneM;
}

/** GPU upload: every slot gets a buffer sized for sceneM (nearest same-M neighbor fill). */
function materializeKeyframeFramesAtM(
  frames: (KeyframeFrame | null)[],
  sceneM: number,
): KeyframeFrame[] {
  let fallback: KeyframeFrame | null = null;
  for (const fr of frames) {
    if (frameAtGridM(fr, sceneM)) {
      fallback = fr!;
      break;
    }
  }
  if (!fallback) throw new Error(`no keyframes at grid M=${sceneM}`);
  const out: KeyframeFrame[] = new Array(frames.length);
  let lastAtM: KeyframeFrame = fallback;
  for (let i = 0; i < frames.length; i++) {
    if (frameAtGridM(frames[i], sceneM)) {
      lastAtM = frames[i]!;
      out[i] = frames[i]!;
    } else {
      let fwd = lastAtM;
      for (let j = i + 1; j < frames.length; j++) {
        if (frameAtGridM(frames[j], sceneM)) {
          fwd = frames[j]!;
          break;
        }
      }
      out[i] = fwd;
    }
  }
  return out;
}

function runOneKeyframeWork(): boolean {
  for (const layerId of pendingPumpLayers) {
    if (isParkedKeyframeLayer(layerId)) {
      pendingPumpLayers.delete(layerId);
      continue;
    }
    const cache = caches.get(layerId);
    const job = asyncJobs.get(layerId);
    if (!cache || !job || job.cancelled || job.gen !== cache.gen) {
      pendingPumpLayers.delete(layerId);
      asyncJobs.delete(layerId);
      continue;
    }

    const works = pickNextKeyframeWork(cache);
    if (!works.length) {
      if (!keyframesFullyReady(cache)) {
        tearLog("keyframe-pump-stall", {
          layerId,
          diag: diagnoseKeyframeCaches().find((d) => d.layerId === layerId),
        });
        // Stay pending: segment/param changes + reconcile may unlock work.
        continue;
      }
      pendingPumpLayers.delete(layerId);
      asyncJobs.delete(layerId);
      onKeyframeProgress?.({
        layerId,
        index: -1,
        frame: cache.frames.find(Boolean) ?? cache.frames[0]!,
        readyCount: cache.readyCount,
        K: cache.K,
        done: true,
      });
      continue;
    }

    const work = works[0]!;
    const t0 = performance.now();
    const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
    const { frame, k, deg, complete, promoted } = executeKeyframeWork(cache, work, stages);
    const bakeMs = performance.now() - t0;
    bakeMsAcc += bakeMs;
    lastBakeMs = bakeMsAcc;
    if (complete && frame && promoted?.length) {
      const first = promoted[0]!;
      const shown = cache.frames[first];
      if (shown) {
        onKeyframeProgress?.({
          layerId,
          index: first,
          frame: shown,
          promoted,
          readyCount: cache.readyCount,
          K: cache.K,
          done: keyframesFullyReady(cache),
        });
      }
      for (const idx of promoted) {
        lastBakeDetails.push({
          ...stages,
          frames: 1,
          deg: cache.frameDeg[idx] ?? deg,
          M: Math.round(Math.cbrt(frameVolumeN(cache.frames[idx]!))),
          role: cache.role,
          layerId,
          paramName: cache.paramName,
          bakeMs: bakeMs / promoted.length,
          async: true,
          index: idx,
        });
      }
      const blend = segmentForValue(cache, getParam(cache.paramName)?.value ?? cache.min);
      if (
        promoted.includes(blend.i0) &&
        promoted.includes(blend.i1) &&
        cache.frameDeg[blend.i0] === cache.frameDeg[blend.i1]
      ) {
        const d = cache.frameDeg[blend.i0] ?? deg;
        maybeLogBlendPairReady(cache, layerId, blend.i0, blend.i1, d, bakeMs, "async", k);
      }
    } else if (complete && frame) {
      onKeyframeProgress?.({
        layerId,
        index: k,
        frame,
        readyCount: cache.readyCount,
        K: cache.K,
        done: keyframesFullyReady(cache),
      });
    } else if (!complete) {
      if (cache.lobattoFinalizeByK.has(k)) {
        const fin = cache.lobattoFinalizeByK.get(k)!;
        console.log(
          `[keyframes] finalize chunk · ${layerId} · slot ${k} · ${fin.phase} · ${bakeMs.toFixed(1)}ms`,
        );
      } else if (cache.lobattoJobByK.has(k)) {
        const job = cache.lobattoJobByK.get(k)!;
        const cur = bakedDegAt(cache, k);
        console.log(
          `[keyframes] lobatto chunk · ${layerId} · slot ${k} · ` +
            `${cur}→${job.targetDeg} · ${job.mode} · ${bakeMs.toFixed(1)}ms`,
        );
      }
    }

    if (keyframesFullyReady(cache)) {
      pendingPumpLayers.delete(layerId);
      asyncJobs.delete(layerId);
      if (frame) {
        onKeyframeProgress?.({
          layerId,
          index: -1,
          frame,
          readyCount: cache.readyCount,
          K: cache.K,
          done: true,
        });
      }
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

  // Ensuring means the layer is enabled and needed — leave parked state.
  parkedKeyframeLayers.delete(key);

  let baked = false;
  if (!cache || !cacheMatches(cache, meta)) {
    cancelAsyncJob(key);
    cacheGen++;
    cache = {
      ...meta,
      targetDeg: opts.deg,
      frames: new Array(K).fill(null),
      frameDeg: new Array(K).fill(0),
      stagingFrames: new Array(K).fill(null),
      stagingDeg: new Array(K).fill(0),
      lobattoByK: new Map(),
      lobattoJobByK: new Map(),
      lobattoFinalizeByK: new Map(),
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
    // Anim restart / replay: skip sync + async when expression unchanged and K slots ready.
    if (keyframesFullyReady(cache)) {
      return buildLayerKeyframeResult(cache, value, opts, false);
    }
  }

  const blend = segmentForValue(cache, value);
  const t0 = performance.now();
  const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
  const startDeg = startLadderDeg(cache.targetDeg);
  let syncCount = 0;
  let lastSyncK = blend.i0;
  let syncPromoted: number[] = [];
  if (!opts.deferSyncBake) {
    for (const k of [blend.i0, blend.i1]) {
      if (cache.frameDeg[k]! > 0) continue;
      const { frame, complete } = bakeFrameAtDeg(cache, k, startDeg, stages, null);
      if (!complete || !frame) throw new Error("sync keyframe bake incomplete");
      syncPromoted = commitFrame(cache, k, frame, startDeg);
      syncCount++;
      lastSyncK = k;
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
    if (
      syncPromoted.includes(blend.i0) &&
      syncPromoted.includes(blend.i1)
    ) {
      maybeLogBlendPairReady(
        cache,
        opts.layerId,
        blend.i0,
        blend.i1,
        startDeg,
        bakeMs,
        "sync",
        lastSyncK,
      );
    }
  }

  const displayBlend = displayBlendForValue(cache, value);
  if (!keyframesFullyReady(cache)) {
    scheduleAsyncFill(key, cache);
  }

  return buildLayerKeyframeResult(cache, value, opts, baked, syncPromoted);
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
  const st = getParam(cache.paramName);
  const tLerp = performance.now();
  const { i0, i1, t } = displayBlendForValue(cache, st?.value ?? cache.min);
  const a = cache.frames[i0];
  const b = cache.frames[i1];
  if (!a || !b) throw new Error("sync keyframe pair missing");
  if (!a.dens || !b.dens) throw new Error("scalar keyframe pair missing dens");
  const d0 = cache.frameDeg[i0] ?? 0;
  const d1 = cache.frameDeg[i1] ?? 0;
  const canLerp =
    d0 === d1 &&
    d0 > 0 &&
    a.dens.length === b.dens.length &&
    t > 0 &&
    t < 1;
  const src = t < 0.5 ? a.dens : b.dens;
  ensureScratchVolume(cache, src.length);
  const out = cache.scratch;
  if (canLerp) {
    lerpFloat32(a.dens, b.dens, t, out.dens!);
  } else {
    out.dens!.fill(0);
    out.dens!.set(src);
  }
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  const M = gridMFromFrame({ dens: out.dens! });
  return {
    dens: out.dens!,
    cheb: ensured.cheb,
    fitRel: ensured.fitRel,
    M,
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
  const { i0, i1, t } = displayBlendForValue(cache, getParam(cache.paramName)?.value ?? cache.min);
  const a = cache.frames[i0];
  const b = cache.frames[i1];
  if (!a || !b) throw new Error("sync keyframe pair missing");
  if (!a.fx || !a.fy || !a.fz || !b.fx || !b.fy || !b.fz) {
    throw new Error("flow keyframe pair missing velocity grids");
  }
  const d0 = cache.frameDeg[i0] ?? 0;
  const d1 = cache.frameDeg[i1] ?? 0;
  const canLerp =
    d0 === d1 &&
    d0 > 0 &&
    a.fx.length === b.fx.length &&
    t > 0 &&
    t < 1;
  const pickA = t < 0.5;
  ensureScratchVolume(cache, (pickA ? a.fx : b.fx).length);
  const out = cache.scratch;
  if (canLerp) {
    lerpFloat32(a.fx, b.fx, t, out.fx!);
    lerpFloat32(a.fy, b.fy, t, out.fy!);
    lerpFloat32(a.fz, b.fz, t, out.fz!);
  } else {
    const sx = pickA ? a.fx : b.fx;
    const sy = pickA ? a.fy : b.fy;
    const sz = pickA ? a.fz : b.fz;
    out.fx!.fill(0);
    out.fy!.fill(0);
    out.fz!.fill(0);
    out.fx!.set(sx);
    out.fy!.set(sy);
    out.fz!.set(sz);
  }
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  const M = gridMFromFrame({ fx: out.fx! });
  return {
    fx: out.fx!.slice(),
    fy: out.fy!.slice(),
    fz: out.fz!.slice(),
    cheb: ensured.cheb,
    fitRel: ensured.fitRel,
    M,
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
  const blend = displayBlendForValue(cache, st.value);
  if (!cache.frames[blend.i0] || !cache.frames[blend.i1]) {
    tearLogOnce(`peek-null-frames-${layerId}`, "peek-null", {
      layerId,
      reason: "missing-frame",
      i0: blend.i0,
      i1: blend.i1,
      d0: cache.frameDeg[blend.i0] ?? 0,
      d1: cache.frameDeg[blend.i1] ?? 0,
    });
    return null;
  }
  return { id: layerId, i0: blend.i0, i1: blend.i1, t: blend.t };
}
