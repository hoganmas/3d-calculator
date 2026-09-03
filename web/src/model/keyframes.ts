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
import {
  bakeOrderND,
  hypercellBlend,
  paramValuesAtIndex,
  totalFrameCount,
  type HypercellCorner,
} from "../math/keyframeGrid.js";
import { getParam } from "./params.js";
import { gridMFromDens, tearLog, tearLogOnce } from "../app/tearDebug.js";
import type { CompiledExpr, CompiledVectorExpr, KeyframeFrame } from "../types/models.js";

export const DEFAULT_KEYFRAME_K = 8;
/** Max Lobatto work (sample + IDCT + grad) per render frame during async fill. */
export const KEYFRAME_LOBATTO_BUDGET_MS = 3;
/**
 * Wall-clock budget (ms) for the async keyframe pump per rendered frame — how
 * many maxWorks=1 units of KEYFRAME_LOBATTO_BUDGET_MS work tickKeyframePump
 * may chain in one rAF before yielding to render. Frame cadence (~16-21ms,
 * GPU-present-paced) is well above this, so it leaves headroom for render +
 * GPU submit while still processing several units instead of exactly one.
 */
export const KEYFRAME_PUMP_FRAME_BUDGET_MS = 7;

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
  paramNames: string[];
  mins: number[];
  maxs: number[];
  compiled?: { bind: CompiledExpr["bind"] };
  vectorCompiled?: CompiledVectorExpr;
  baseParams: Record<string, number>;
  half: number;
  deg: number;
  K?: number;
}

interface LayerKeyframeCache {
  paramNames: string[];
  mins: number[];
  maxs: number[];
  K: number;
  totalFrames: number;
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
  /** Fingerprint of free params held fixed during grid bake (not on keyframe axes). */
  fixedParamsFp: string;
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
  /** @deprecated use paramNames */
  paramName?: string;
  paramNames?: string[];
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
  paramNames: string[];
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
  sceneBake: {
    isosurfaceLayers: { id?: string; keyframes?: KeyframeFrame[]; blend?: { i0: number; i1: number; t: number } }[];
    cloudLayers?: { id?: string; dens?: Float32Array; keyframes?: KeyframeFrame[]; blend?: { i0: number; i1: number; t: number } }[];
    M: number;
  },
  uploadM?: number,
): number {
  const cache = caches.get(layerId);
  if (!cache || (cache.role !== "isosurface" && cache.role !== "cloud")) return sceneBake.M;
  const layer =
    cache.role === "cloud"
      ? sceneBake.cloudLayers?.find((x) => x.id === layerId)
      : sceneBake.isosurfaceLayers.find((x) => x.id === layerId);
  if (!layer) return sceneBake.M;
  const st = getParam(cache.paramNames[0]!);
  const value = st?.value ?? cache.mins[0]!;
  const blend = displayBlendForValue(cache, value);
  layer.blend = { i0: blend.i0, i1: blend.i1, t: blend.t };
  const blendM = isoBlendSceneM(cache, blend.i0, blend.i1, blend.t);
  if (blendM <= 0) return sceneBake.M;
  const M =
    uploadM && uploadM > 0 && cache.frames.some((fr) => frameAtGridM(fr, uploadM))
      ? uploadM
      : blendM;
  layer.keyframes = materializeKeyframeFramesAtM(cache.frames, M);
  if (cache.role === "cloud") {
    const dens = layer.keyframes[blend.i0]?.dens || layer.keyframes[0]?.dens;
    if (dens) (layer as { dens?: Float32Array }).dens = dens;
  }
  sceneBake.M = Math.max(sceneBake.M, M);
  return M;
}

/** Grid M for the current iso blend pair (matches displayBlendForValue snap semantics). */
export function getIsoBlendSceneM(layerId: string): number {
  const cache = caches.get(layerId);
  if (!cache || (cache.role !== "isosurface" && cache.role !== "cloud")) return 0;
  const st = getParam(cache.paramNames[0]!);
  const value = st?.value ?? cache.mins[0]!;
  const blend = displayBlendForValue(cache, value);
  return isoBlendSceneM(cache, blend.i0, blend.i1, blend.t);
}

/** Promote staged pair when both slots share a staging degree (display path is read-only). */
export function refreshIsoBlendDisplay(layerId: string): number[] {
  const cache = caches.get(layerId);
  if (!cache || (cache.role !== "isosurface" && cache.role !== "cloud")) return [];
  const st = getParam(cache.paramNames[0]!);
  if (!st) return [];
  const { i0, i1 } = segmentForValue(cache, st.value);
  return reconcileStaging(cache, [i0, i1]);
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
  pumpLayerCursor = 0;
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
    if (cache.readyCount < cache.totalFrames) return false;
  }
  return true;
}

/** Coarse hypercell ready for every active (enabled) cached layer (splash / first frame). */
export function keyframesSplashReady(): boolean {
  if (!hasActiveKeyframeCaches()) return true;
  for (const [id, cache] of caches) {
    if (isParkedKeyframeLayer(id)) continue;
    if (cacheNDims(cache) === 1) {
      if (peekKeyframeBlend(cache.bakeOpts.layerId) == null) return false;
      continue;
    }
    const corners = hypercellCornerIndices(cache);
    if (!corners.every((k) => !!cache.frames[k] && (cache.frameDeg[k] ?? 0) > 0)) return false;
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
    totalFrames: cache.totalFrames,
    paramNames: cache.paramNames,
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
    const st = getParam(cache.paramNames[0]!);
    const value = st?.value ?? cache.mins[0]!;
    const corners = hypercellCornerIndices(cache);
    const { i0, i1 } = segmentForValue(cache, value);
    const works = parked ? [] : peekPickNextWork(cache);
    const slots: KeyframeStallSlotDiag[] = [];
    for (let k = 0; k < cache.totalFrames; k++) {
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

/**
 * Relative cost of baking one ladder rung. Sampling and the separable IDCT
 * transform (chebLobatto.ts) both run over an (deg+1)^3 grid, so rungs near
 * the target (ladder doubles each step) dominate wall-clock time even though
 * they're a small fraction of the rung *count* — weight by grid size instead
 * of by rung index, or the bar sprints through cheap early rungs and crawls
 * the expensive final ones.
 */
function rungCost(deg: number): number {
  const m = deg + 1;
  return m * m * m;
}

function slotLoadFraction(cache: LayerKeyframeCache, k: number): number {
  const target = cache.targetDeg;
  if (target <= 0) return 1;
  const d = bakedDegAt(cache, k);
  if (d >= target) return 1;
  if (d <= 0) return 0;
  const ladder = lobattoLadderDegrees(target);
  let totalCost = 0;
  let doneCost = 0;
  for (const deg of ladder) {
    const cost = rungCost(deg);
    totalCost += cost;
    if (d >= deg) doneCost += cost;
  }
  return totalCost > 0 ? doneCost / totalCost : 0;
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
    slotsTotal += cache.totalFrames;
    slotsAtTarget += cache.readyCount;
    for (let k = 0; k < cache.totalFrames; k++) sum += slotLoadFraction(cache, k);
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
 * All dirty animating free-params eligible for keyframe caching.
 */
export function keyframeAnimParams(freeParams: string[], dirty: Set<string>): string[] | null {
  if (!dirty?.size || !freeParams?.length) return null;
  const hit: string[] = [];
  for (const p of freeParams) {
    if (!dirty.has(p)) continue;
    const st = getParam(p);
    if (!st || st.driven || !st.animating) continue;
    hit.push(p);
  }
  hit.sort();
  return hit.length ? hit : null;
}

/**
 * Eligible when dirty free-params collapse to exactly one animated slider
 * (not driven by another equation).
 */
export function keyframeAnimParam(freeParams: string[], dirty: Set<string>) {
  const hit = keyframeAnimParams(freeParams, dirty);
  return hit?.length === 1 ? hit[0]! : null;
}

/**
 * Keep existing N-D cache axes when a subset of params is still animating.
 * Paused axes stay in the grid at their current slider value (no rebake).
 */
export function resolveKeyframeParamNames(
  layerId: string,
  animatingParams: readonly string[],
): string[] {
  const requested = [...animatingParams].sort();
  const cache = caches.get(layerId);
  if (!cache || isParkedKeyframeLayer(layerId) || cache.paramNames.length <= requested.length) {
    return requested;
  }
  for (const p of requested) {
    if (!cache.paramNames.includes(p)) return requested;
  }
  for (const p of cache.paramNames) {
    if (requested.includes(p)) continue;
    const st = getParam(p);
    if (!st || st.driven || st.animating) return requested;
  }
  return [...cache.paramNames];
}

function normalizeParamNames(opts: Pick<EnsureKeyframesOpts, "paramName" | "paramNames">): string[] {
  if (opts.paramNames?.length) return [...opts.paramNames].sort();
  if (opts.paramName) return [opts.paramName];
  throw new Error("paramNames required for keyframes");
}

function resolveKeyframeAxes(opts: EnsureKeyframesOpts): {
  paramNames: string[];
  mins: number[];
  maxs: number[];
  K: number;
  totalFrames: number;
} {
  const paramNames = resolveKeyframeParamNames(opts.layerId, normalizeParamNames(opts));
  const nDims = paramNames.length;
  const K = Math.max(2, opts.K ?? DEFAULT_KEYFRAME_K);
  const mins: number[] = [];
  const maxs: number[] = [];
  for (const name of paramNames) {
    const st = getParam(name);
    if (!st) throw new Error(`Unknown param “${name}” for keyframes`);
    mins.push(st.min);
    maxs.push(st.max);
  }
  return { paramNames, mins, maxs, K, totalFrames: totalFrameCount(K, nDims) };
}

function keyframeFreeParams(opts: EnsureKeyframesOpts): string[] {
  if (opts.role === "flow") return opts.vectorCompiled?.freeParams ?? [];
  return opts.compiled?.freeParams ?? [];
}

/** Values of expression free params that are not keyframe axes (must match across cached frames). */
export function keyframeFixedParamsFingerprint(
  freeParams: readonly string[],
  paramNames: readonly string[],
  baseParams: Record<string, number>,
): string {
  const axis = new Set(paramNames);
  const parts: string[] = [];
  for (const p of [...freeParams].sort()) {
    if (axis.has(p)) continue;
    const v = baseParams[p];
    parts.push(`${p}=${Number.isFinite(v) ? String(v) : "nan"}`);
  }
  return parts.join("\0");
}

function cacheNDims(cache: LayerKeyframeCache): number {
  return cache.paramNames.length;
}

function currentParamValues(cache: LayerKeyframeCache): number[] {
  return cache.paramNames.map((name, d) => getParam(name)?.value ?? cache.mins[d]!);
}

function hypercellForCache(cache: LayerKeyframeCache) {
  return hypercellBlend(cache.mins, cache.maxs, cache.K, currentParamValues(cache));
}

function hypercellCornerIndices(cache: LayerKeyframeCache): number[] {
  return hypercellForCache(cache).corners.map((c) => c.index);
}

/**
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @param {number} t
 * @param {Float32Array} out
 */
function canMultilinearLerp(
  cache: LayerKeyframeCache,
  corners: HypercellCorner[],
  field: keyof KeyframeFrame,
): boolean {
  if (!corners.length) return false;
  const degs = corners.map((c) => cache.frameDeg[c.index] ?? 0);
  if (!degs.every((d) => d === degs[0] && d > 0)) return false;
  const first = cache.frames[corners[0]!.index]?.[field] as Float32Array | undefined;
  if (!first?.length) return false;
  for (const c of corners) {
    const arr = cache.frames[c.index]?.[field] as Float32Array | undefined;
    if (!arr || arr.length !== first.length) return false;
  }
  return corners.some((c) => c.weight > 0 && c.weight < 1);
}

function multilinearBlendField(
  cache: LayerKeyframeCache,
  corners: HypercellCorner[],
  field: keyof KeyframeFrame,
  out: Float32Array,
): void {
  out.fill(0);
  for (const { index, weight } of corners) {
    const src = cache.frames[index]?.[field] as Float32Array | undefined;
    if (!src) continue;
    for (let i = 0; i < out.length; i++) out[i] += weight * src[i]!;
  }
}

function heaviestCorner(corners: HypercellCorner[]): HypercellCorner {
  return corners.reduce((a, b) => (b.weight > a.weight ? b : a));
}

function slotHasDisplay(cache: LayerKeyframeCache, k: number): boolean {
  return !!cache.frames[k] && (cache.frameDeg[k] ?? 0) > 0;
}

function hasAnyDisplayFrame(cache: LayerKeyframeCache): boolean {
  for (let k = 0; k < cache.totalFrames; k++) {
    if (slotHasDisplay(cache, k)) return true;
  }
  return false;
}

/** Nearest display-ready slot to `prefer` (outward search). */
function nearestReadyDisplaySlot(cache: LayerKeyframeCache, prefer: number): number {
  const n = cache.totalFrames;
  if (n <= 0) return -1;
  const p = Math.min(n - 1, Math.max(0, prefer | 0));
  if (slotHasDisplay(cache, p)) return p;
  for (let d = 1; d < n; d++) {
    const lo = p - d;
    const hi = p + d;
    if (lo >= 0 && slotHasDisplay(cache, lo)) return lo;
    if (hi < n && slotHasDisplay(cache, hi)) return hi;
  }
  return -1;
}

function sampleVolumeFromCache(cache: LayerKeyframeCache) {
  const corners = hypercellForCache(cache).corners;
  let snapIdx = heaviestCorner(corners).index;
  let snap = cache.frames[snapIdx];
  if (!snap) {
    const readyCorners = corners.filter((c) => slotHasDisplay(cache, c.index));
    if (readyCorners.length) {
      snapIdx = heaviestCorner(readyCorners).index;
      snap = cache.frames[snapIdx];
    } else {
      snapIdx = nearestReadyDisplaySlot(cache, snapIdx);
      snap = snapIdx >= 0 ? cache.frames[snapIdx] : null;
    }
  }
  if (!snap) throw new Error("keyframe hypercell missing");
  ensureScratchVolume(cache, frameVolumeN(snap));
  const out = cache.scratch;

  if (out.dens) {
    if (canMultilinearLerp(cache, corners, "dens")) {
      multilinearBlendField(cache, corners, "dens", out.dens);
    } else if (snap.dens) {
      out.dens.fill(0);
      out.dens.set(snap.dens);
    }
  }
  for (const g of ["gx", "gy", "gz"] as const) {
    const dst = out[g];
    if (!dst) continue;
    if (canMultilinearLerp(cache, corners, g)) {
      multilinearBlendField(cache, corners, g, dst);
    } else if (snap[g]) {
      dst.fill(0);
      dst.set(snap[g]!);
    }
  }
  for (const f of ["fx", "fy", "fz"] as const) {
    const dst = out[f];
    if (!dst) continue;
    if (canMultilinearLerp(cache, corners, f)) {
      multilinearBlendField(cache, corners, f, dst);
    } else if (snap[f]) {
      dst.fill(0);
      dst.set(snap[f]!);
    }
  }
  return { out, snapIdx, corners };
}

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
  const min = cache.mins[0] ?? 0;
  const max = cache.maxs[0] ?? 1;
  const span = Math.max(1e-12, max - min);
  const u = Math.min(1, Math.max(0, (value - min) / span));
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
    | "paramNames"
    | "mins"
    | "maxs"
    | "K"
    | "totalFrames"
    | "deg"
    | "half"
    | "role"
    | "latex"
    | "isoLevel"
    | "fixedParamsFp"
  >,
) {
  if (
    cache.paramNames.length !== meta.paramNames.length ||
    cache.role !== meta.role ||
    cache.K !== meta.K ||
    cache.totalFrames !== meta.totalFrames ||
    cache.deg !== meta.deg ||
    Math.abs(cache.half - meta.half) >= 1e-12 ||
    Math.abs(cache.isoLevel - meta.isoLevel) >= 1e-12 ||
    cache.latex !== meta.latex ||
    cache.fixedParamsFp !== meta.fixedParamsFp ||
    cache.frames.length !== meta.totalFrames
  ) {
    return false;
  }
  for (let i = 0; i < cache.paramNames.length; i++) {
    if (cache.paramNames[i] !== meta.paramNames[i]) return false;
    if (Math.abs(cache.mins[i]! - meta.mins[i]!) >= 1e-12) return false;
    if (Math.abs(cache.maxs[i]! - meta.maxs[i]!) >= 1e-12) return false;
  }
  return true;
}

function bindParamsForSlot(opts: BakeFrameOpts, k: number): Record<string, number> {
  const gridParams = paramValuesAtIndex(opts.paramNames, opts.mins, opts.maxs, opts.K ?? DEFAULT_KEYFRAME_K, k);
  return { ...opts.baseParams, ...gridParams };
}

function bakeFlowFrameAtDeg(
  opts: BakeFrameOpts,
  k: number,
  deg: number,
): KeyframeFrame {
  if (!opts.vectorCompiled) throw new Error("flow keyframes require vectorCompiled");
  const params = bindParamsForSlot(opts, k);
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
  const params = bindParamsForSlot(opts, k);
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

/** Lowest display degree currently on screen (0 if nothing is displayed yet). */
function playingDisplayDeg(cache: LayerKeyframeCache): number {
  let min = Infinity;
  for (let k = 0; k < cache.totalFrames; k++) {
    const d = cache.frameDeg[k] ?? 0;
    if (d <= 0) continue;
    if (d < min) min = d;
  }
  return Number.isFinite(min) ? min : 0;
}

function slotHasRung(cache: LayerKeyframeCache, k: number, deg: number): boolean {
  if ((cache.frameDeg[k] ?? 0) >= deg) return true;
  return (cache.stagingDeg[k] ?? 0) >= deg && !!cache.stagingFrames[k];
}

/**
 * Promote every slot to the highest ladder rung that is staged (or already
 * displayed) on all slots. Playback stays on the previous complete rung until
 * then, so the playhead can keep interpolating at lower quality.
 */
function tryPromoteGlobalRung(cache: LayerKeyframeCache): number[] {
  const ladder = lobattoLadderDegrees(cache.targetDeg);
  let promoteDeg = 0;
  for (const d of ladder) {
    let ready = true;
    for (let k = 0; k < cache.totalFrames; k++) {
      if (!slotHasRung(cache, k, d)) {
        ready = false;
        break;
      }
    }
    if (!ready) break;
    promoteDeg = d;
  }
  if (promoteDeg <= 0) return [];
  const promoted: number[] = [];
  for (let k = 0; k < cache.totalFrames; k++) {
    if ((cache.frameDeg[k] ?? 0) >= promoteDeg) continue;
    const sd = cache.stagingDeg[k] ?? 0;
    const fr = cache.stagingFrames[k];
    if (!fr || sd < promoteDeg) continue;
    applyDisplayFrame(cache, k, fr, sd);
    promoted.push(k);
  }
  if (promoted.length) {
    tearLog("promote-global-rung", {
      layerId: cache.bakeOpts.layerId,
      deg: promoteDeg,
      promoted,
    });
  }
  return promoted;
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
  // Mismatched degrees: displayBlendForValue always snaps t to exactly 0 or
  // 1 here (shows one side alone, no interpolation), so the resolution
  // should follow whichever side is actually on screen — not the lower of
  // the two. Picking the lower one unconditionally downgraded the shown
  // frame the moment its *invisible* neighbor got any bake at all, which
  // read as a visible quality drop ("rung hop") for a frame that was never
  // actually being blended with anything.
  if (d0 > 0 && d1 > 0 && d0 !== d1) {
    if (t <= 0) return M0 || M1 || 0;
    if (t >= 1) return M1 || M0 || 0;
    return d0 < d1 ? M0 || M1 || 0 : M1 || M0 || 0;
  }
  if (d0 > d1 || (d0 > 0 && d1 <= 0)) return M0 || M1 || 0;
  if (d1 > d0 || (d1 > 0 && d0 <= 0)) return M1 || M0 || 0;
  if (M0 > 0 && M0 === M1) return M0;
  return t < 0.5 ? M0 || M1 || 0 : M1 || M0 || 0;
}

/**
 * Flush staging onto display when it does not jump ahead of the playing rung,
 * then promote a complete ladder rung across all slots.
 */
function reconcileStaging(cache: LayerKeyframeCache, _corners: number[]): number[] {
  const playDeg = playingDisplayDeg(cache);
  const promoted: number[] = [];
  for (let k = 0; k < cache.totalFrames; k++) {
    const sd = cache.stagingDeg[k] ?? 0;
    const fr = cache.stagingFrames[k];
    if (!fr || sd <= (cache.frameDeg[k] ?? 0)) continue;
    if (playDeg > 0 && sd > playDeg) continue;
    applyDisplayFrame(cache, k, fr, sd);
    promoted.push(k);
  }
  promoted.push(...tryPromoteGlobalRung(cache));
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

/**
 * Commit a baked frame: display it as soon as it's ready.
 *
 * Used to hold higher rungs back in staging until every slot in the K^N
 * grid reached the same rung (playingDisplayDeg/tryPromoteGlobalRung) —
 * intended to avoid visible mismatch, but the only mismatch that's ever
 * actually rendered is between the *currently blending* pair (i0, i1), and
 * isoBlendSceneM() already handles that case on its own (falls back to the
 * lower-degree side's M until both catch up). Gating on the whole grid
 * instead meant any slot's completed bake sat invisible until distant,
 * rarely-visited parts of the timeline finished too — jumping to a fresh
 * point showed its first (coarse) rung, then stayed stuck there.
 */
function commitFrame(
  cache: LayerKeyframeCache,
  k: number,
  frame: KeyframeFrame,
  deg: number,
): number[] {
  if (deg <= (cache.frameDeg[k] ?? 0)) {
    tearLog("commit-stale-skip", {
      layerId: cache.bakeOpts.layerId,
      k,
      deg,
      displayDeg: cache.frameDeg[k] ?? 0,
    });
    return [];
  }
  clearCoarseStaging(cache, k);
  applyDisplayFrame(cache, k, frame, deg);
  tearLog("commit-join-play", {
    layerId: cache.bakeOpts.layerId,
    k,
    deg,
    M: gridMFromDens(frame.dens),
  });
  return [k];
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
    const ready = nearestReadyDisplaySlot(cache, t < 0.5 ? i0 : i1);
    const snap =
      d0 > 0 && a && d1 <= 0 ? { i0, i1, t: 0 } :
      d1 > 0 && b && d0 <= 0 ? { i0, i1, t: 1 } :
      d0 > d1 && a ? { i0, i1, t: 0 } :
      d1 > d0 && b ? { i0, i1, t: 1 } :
      ready >= 0 ? { i0: ready, i1: ready, t: 0 } :
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

function bakeOrderForCache(cache: LayerKeyframeCache): number[] {
  const corners = hypercellCornerIndices(cache);
  reconcileStaging(cache, corners);
  if (cacheNDims(cache) === 1) {
    const { i0, i1 } = segmentForValue(cache, currentParamValues(cache)[0]!);
    return bakeOrder(cache.K, i0, i1);
  }
  return bakeOrderND(cache.K, cacheNDims(cache), corners);
}

function keyframesFullyReady(cache: LayerKeyframeCache): boolean {
  return cache.readyCount >= cache.totalFrames;
}

function schedDegAt(cache: LayerKeyframeCache, k: number): number {
  return Math.max(cache.frameDeg[k] ?? 0, cache.stagingDeg[k] ?? 0);
}

/** In-flight Lobatto work targeting exactly phaseDeg, if any. */
function inFlightWorkAtPhase(cache: LayerKeyframeCache, phaseDeg: number): KeyframeWork | null {
  const order = bakeOrderForCache(cache);
  for (const k of order) {
    if (cache.lobattoFinalizeByK.has(k)) {
      const fin = cache.lobattoFinalizeByK.get(k)!;
      if (fin.lob.deg === phaseDeg) return { kind: "refine", k, nextDeg: fin.lob.deg };
    }
    if (cache.lobattoJobByK.has(k)) {
      const job = cache.lobattoJobByK.get(k)!;
      if (job.targetDeg === phaseDeg) return { kind: "refine", k, nextDeg: job.targetDeg };
    }
  }
  return null;
}

function newWorkAtPhase(cache: LayerKeyframeCache, phaseDeg: number): KeyframeWork | null {
  const target = cache.targetDeg;
  const ladder = lobattoLadderDegrees(target);
  const start = ladder[0] ?? startLadderDeg(target);
  if (!ladder.includes(phaseDeg) && phaseDeg !== start) return null;
  const order = bakeOrderForCache(cache);
  for (const k of order) {
    // Don't start a new slot while another is mid-chunk on this layer.
    if (cache.lobattoFinalizeByK.has(k) || cache.lobattoJobByK.has(k)) continue;
    const d = schedDegAt(cache, k);
    if (d === 0) {
      if (phaseDeg !== start) continue;
      return { kind: "coarse", k };
    }
    if (d < phaseDeg && phaseDeg <= target) return { kind: "refine", k, nextDeg: phaseDeg };
  }
  return null;
}

/** Work for one layer at a single ladder rung (in-flight first, then new slots). */
function pickWorkAtPhase(cache: LayerKeyframeCache, phaseDeg: number): KeyframeWork | null {
  return inFlightWorkAtPhase(cache, phaseDeg) ?? newWorkAtPhase(cache, phaseDeg);
}

/**
 * Continue ANY slot that's already mid-chunk, anywhere in the grid —
 * checked before starting new work for the active pair. t moves every
 * frame during playback, so the active pair drifts continuously; without
 * this, a job started for slot k when it was the pair got abandoned the
 * instant t ticked past it, since only the (now different) current pair
 * was ever checked for in-flight work — scattering partial progress across
 * many slots and never reliably finishing any of them (visible as the
 * quality jumping around / freezing instead of progressing smoothly).
 */
function anyInFlightWork(cache: LayerKeyframeCache): KeyframeWork | null {
  const order = bakeOrderForCache(cache);
  for (const k of order) {
    if (cache.lobattoFinalizeByK.has(k)) {
      const fin = cache.lobattoFinalizeByK.get(k)!;
      return { kind: "refine", k, nextDeg: fin.lob.deg };
    }
    if (cache.lobattoJobByK.has(k)) {
      const job = cache.lobattoJobByK.get(k)!;
      return { kind: "refine", k, nextDeg: job.targetDeg };
    }
  }
  return null;
}

/**
 * Start new work for whichever of the actively-blended pair is behind,
 * through its own ladder, independent of the rest of the K^N grid — keeps
 * the two slots roughly in lockstep with each other (matches
 * isoBlendSceneM's assumption that a mismatch is at most one rung), but
 * doesn't wait on unrelated, rarely-visited parts of the timeline the way
 * the old phase-first/whole-grid scan did. That scan tried every slot's
 * rung N before any slot's rung N+1, so jumping to a fresh point meant the
 * pair sat at whatever coarse rung the *slowest* corner of the grid had
 * reached, even once its own bake finished. Assumes anyInFlightWork() has
 * already been checked — it doesn't itself look at in-flight jobs.
 *
 * Once the pair itself has nothing left to do (both caught up, or both
 * mid-chunk elsewhere), gives the immediately neighboring segments — the
 * ones the playhead is about to enter — a head start as a secondary
 * priority, below the pair but still ahead of the rest of the grid. Without
 * this, crossing into a fresh segment starts it from zero baked data right
 * as you reach it, which isoBlendSceneM has no choice but to show as a
 * visible quality drop at every segment boundary (a "rung hop").
 */
function pickNewWorkForActivePair(
  cache: LayerKeyframeCache,
  i0: number,
  i1: number,
): KeyframeWork | null {
  const target = cache.targetDeg;
  const ladder = lobattoLadderDegrees(target);
  const start = ladder[0] ?? startLadderDeg(target);
  const tryAdvance = (k: number): KeyframeWork | null => {
    if (cache.lobattoFinalizeByK.has(k) || cache.lobattoJobByK.has(k)) return null;
    const d = schedDegAt(cache, k);
    if (d >= target) return null;
    if (d === 0) return { kind: "coarse", k };
    const next = ladder[ladder.indexOf(d) + 1] ?? (d < start ? start : undefined);
    return next != null ? { kind: "refine", k, nextDeg: next } : null;
  };

  const core = i0 === i1 ? [i0] : [i0, i1];
  core.sort((a, b) => schedDegAt(cache, a) - schedDegAt(cache, b));
  for (const k of core) {
    const w = tryAdvance(k);
    if (w) return w;
  }

  const K = cache.K;
  const lookahead = [i0 - 1, i1 + 1].filter((k) => k >= 0 && k < K && k !== i0 && k !== i1);
  lookahead.sort((a, b) => schedDegAt(cache, a) - schedDegAt(cache, b));
  for (const k of lookahead) {
    const w = tryAdvance(k);
    if (w) return w;
  }
  return null;
}

function pickNextKeyframeWork(cache: LayerKeyframeCache): KeyframeWork[] {
  // Finish anything already mid-chunk before starting something new, no
  // matter where the active pair has drifted to since that job began.
  const inFlight = anyInFlightWork(cache);
  if (inFlight) return [inFlight];
  if (cacheNDims(cache) === 1) {
    const { i0, i1 } = segmentForValue(cache, currentParamValues(cache)[0]!);
    const paired = pickNewWorkForActivePair(cache, i0, i1);
    if (paired) return [paired];
  }
  const ladder = lobattoLadderDegrees(cache.targetDeg);
  for (const phaseDeg of ladder) {
    const w = pickWorkAtPhase(cache, phaseDeg);
    if (w) return [w];
  }
  return [];
}

/** Round-robin cursor across pending layers at the current global ladder rung. */
let pumpLayerCursor = 0;

function pendingPumpLayerIds(): string[] {
  const ids: string[] = [];
  for (const layerId of pendingPumpLayers) {
    if (isParkedKeyframeLayer(layerId)) continue;
    const cache = caches.get(layerId);
    const job = asyncJobs.get(layerId);
    if (!cache || !job || job.cancelled || job.gen !== cache.gen) continue;
    ids.push(layerId);
  }
  ids.sort(); // stable, deterministic across Set iteration order
  return ids;
}

/**
 * Round-robin across expressions; each climbs its own degree ladder independently.
 * One dirty/regenerating expression must not block or reset another's rung.
 */
function pickNextGlobalKeyframeWork(): {
  layerId: string;
  cache: LayerKeyframeCache;
  work: KeyframeWork;
} | null {
  const layerIds = pendingPumpLayerIds();
  if (!layerIds.length) return null;

  const n = layerIds.length;
  const start = ((pumpLayerCursor % n) + n) % n;
  for (let i = 0; i < n; i++) {
    const idx = (start + i) % n;
    const layerId = layerIds[idx]!;
    const cache = caches.get(layerId)!;
    const works = pickNextKeyframeWork(cache);
    if (works[0]) {
      pumpLayerCursor = idx + 1;
      return { layerId, cache, work: works[0] };
    }
  }
  return null;
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
  if (cacheNDims(cache) > 1) {
    const corners = hypercellCornerIndices(cache);
    const proto = corners.map((k) => cache.frames[k]).find(Boolean);
    if (!proto) {
      throw new Error(`keyframe hypercell not ready · ${opts.layerId}`);
    }
    const displayBlend = segmentForValue(cache, value);
    const frames = materializeKeyframeFrames(cache.frames);
    return {
      frames,
      rawFrames: cache.frames.slice(),
      blend: displayBlend,
      cheb: proto.cheb,
      fitRel: proto.fitRel,
      M: gridMFromFrame(proto),
      baked,
      gpuUploadNeeded: false,
      readyCount: cache.readyCount,
      complete: keyframesFullyReady(cache),
    };
  }

  const displayBlend = displayBlendForValue(cache, value);
  const a0 = cache.frames[displayBlend.i0];
  const a1 = cache.frames[displayBlend.i1];
  let blendM = isoBlendSceneM(cache, displayBlend.i0, displayBlend.i1, displayBlend.t);
  if (blendM <= 0 || (!a0 && !a1)) {
    const ready = nearestReadyDisplaySlot(cache, displayBlend.i0);
    const fallback = ready >= 0 ? cache.frames[ready] : null;
    if (!fallback) {
      throw new Error(
        `keyframe blend pair not ready · ${opts.layerId} · ` +
          `(${displayBlend.i0},${displayBlend.i1}) M=${blendM}`,
      );
    }
    blendM = gridMFromFrame(fallback);
    const frames =
      cache.role === "isosurface"
        ? materializeKeyframeFramesAtM(cache.frames, blendM)
        : materializeKeyframeFrames(cache.frames);
    return {
      frames,
      rawFrames: cache.frames.slice(),
      blend: { i0: ready, i1: ready, t: 0 },
      cheb: fallback.cheb,
      fitRel: fallback.fitRel,
      M: blendM,
      baked,
      gpuUploadNeeded: false,
      readyCount: cache.readyCount,
      complete: keyframesFullyReady(cache),
    };
  }
  const frames =
    cache.role === "isosurface"
      ? materializeKeyframeFramesAtM(cache.frames, blendM)
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
    `[keyframes] blend pair ready · ${layerId} · ${cache.paramNames.join("+")} · ` +
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
  const sceneM = gridMFromFrame(last);
  if (sceneM <= 0) throw new Error("no keyframes ready");
  return materializeKeyframeFramesAtM(frames, sceneM);
}

function frameAtGridM(fr: KeyframeFrame | null | undefined, sceneM: number): boolean {
  if (!fr || sceneM <= 0) return false;
  return gridMFromFrame(fr) === sceneM;
}

/** GPU upload: every slot gets a buffer sized for sceneM (nearest same-M neighbor fill). */
function materializeKeyframeFramesAtM(
  frames: (KeyframeFrame | null)[],
  sceneM: number,
): KeyframeFrame[] {
  if (sceneM <= 0) throw new Error("no keyframes ready");
  let fallback: KeyframeFrame | null = null;
  for (const fr of frames) {
    if (frameAtGridM(fr, sceneM)) {
      fallback = fr!;
      break;
    }
  }
  if (!fallback) {
    // Wrong target M (e.g. stale scene M=2) — use this layer's own available grid.
    let anyM = 0;
    for (const fr of frames) {
      const m = gridMFromFrame(fr ?? {});
      if (m > 0) {
        anyM = m;
        break;
      }
    }
    if (anyM > 0 && anyM !== sceneM) {
      tearLog("materialize-m-fallback", { requestedM: sceneM, usedM: anyM });
      return materializeKeyframeFramesAtM(frames, anyM);
    }
    throw new Error(`no keyframes at grid M=${sceneM}`);
  }
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
  // Drop stale / parked entries so the global ladder only sees live layers.
  for (const layerId of [...pendingPumpLayers]) {
    if (isParkedKeyframeLayer(layerId)) {
      pendingPumpLayers.delete(layerId);
      continue;
    }
    const cache = caches.get(layerId);
    const job = asyncJobs.get(layerId);
    if (!cache || !job || job.cancelled || job.gen !== cache.gen) {
      pendingPumpLayers.delete(layerId);
      asyncJobs.delete(layerId);
    }
  }

  const picked = pickNextGlobalKeyframeWork();
  if (!picked) {
    for (const layerId of [...pendingPumpLayers]) {
      const cache = caches.get(layerId);
      if (!cache) {
        pendingPumpLayers.delete(layerId);
        asyncJobs.delete(layerId);
        continue;
      }
      if (keyframesFullyReady(cache)) {
        pendingPumpLayers.delete(layerId);
        asyncJobs.delete(layerId);
        const frame = cache.frames.find(Boolean) ?? cache.frames[0];
        if (frame) {
          onKeyframeProgress?.({
            layerId,
            index: -1,
            frame,
            readyCount: cache.readyCount,
            K: cache.totalFrames,
            done: true,
          });
        }
        continue;
      }
      if (!pickNextKeyframeWork(cache).length) {
        tearLog("keyframe-pump-stall", {
          layerId,
          diag: diagnoseKeyframeCaches().find((d) => d.layerId === layerId),
        });
      }
    }
    return false;
  }

  const { layerId, cache, work } = picked;
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
        paramNames: cache.paramNames,
        bakeMs: bakeMs / promoted.length,
        async: true,
        index: idx,
      });
    }
    const blend = segmentForValue(cache, getParam(cache.paramNames[0]!)?.value ?? cache.mins[0]!);
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
  const axes = resolveKeyframeAxes(opts);
  const { paramNames, mins, maxs, K, totalFrames } = axes;
  const freeParams = keyframeFreeParams(opts);
  const fixedParamsFp = keyframeFixedParamsFingerprint(freeParams, paramNames, opts.baseParams);
  const value = getParam(paramNames[0]!)?.value ?? mins[0]!;
  const key = opts.layerId;

  let cache = caches.get(key);
  const meta = {
    paramNames,
    mins,
    maxs,
    K,
    totalFrames,
    deg: opts.deg,
    half: opts.half,
    role: opts.role,
    latex: opts.latex,
    isoLevel: opts.isoLevel ?? 0,
    fixedParamsFp,
  };

  parkedKeyframeLayers.delete(key);

  let baked = false;
  if (!cache || !cacheMatches(cache, meta)) {
    cancelAsyncJob(key);
    cacheGen++;
    cache = {
      ...meta,
      targetDeg: opts.deg,
      frames: new Array(totalFrames).fill(null),
      frameDeg: new Array(totalFrames).fill(0),
      stagingFrames: new Array(totalFrames).fill(null),
      stagingDeg: new Array(totalFrames).fill(0),
      lobattoByK: new Map(),
      lobattoJobByK: new Map(),
      lobattoFinalizeByK: new Map(),
      scratch: { dens: new Float32Array(0) },
      gen: cacheGen,
      readyCount: 0,
      fixedParamsFp,
      bakeOpts: {
        layerId: opts.layerId,
        latex: opts.latex,
        role: opts.role,
        isoLevel: opts.isoLevel ?? 0,
        paramNames,
        mins,
        maxs,
        compiled: opts.compiled,
        vectorCompiled: opts.vectorCompiled,
        baseParams: { ...opts.baseParams },
        half: opts.half,
        deg: opts.deg,
        K,
      },
    };
    caches.set(key, cache);
    baked = true;
  } else {
    cache.bakeOpts = {
      ...cache.bakeOpts,
      baseParams: { ...opts.baseParams },
      compiled: opts.compiled,
      vectorCompiled: opts.vectorCompiled,
    };
    if (keyframesFullyReady(cache)) {
      return buildLayerKeyframeResult(cache, value, opts, false);
    }
  }

  const cornerIndices = hypercellCornerIndices(cache);
  const blend = segmentForValue(cache, value);
  const t0 = performance.now();
  const stages: BakeStages = { sampleMs: 0, chebMs: 0, idctMs: 0, gradMs: 0 };
  const startDeg = startLadderDeg(cache.targetDeg);
  let syncCount = 0;
  let lastSyncK = cornerIndices[0] ?? 0;
  let syncPromoted: number[] = [];
  let syncDeg = startDeg;
  const canDeferSync = !!opts.deferSyncBake && hasAnyDisplayFrame(cache);
  if (!canDeferSync) {
    syncDeg = Math.max(startDeg, ...cornerIndices.map((c) => cache.frameDeg[c] ?? 0));
    for (const k of cornerIndices) {
      if (slotHasDisplay(cache, k)) continue;
      const { frame, complete } = bakeFrameAtDeg(cache, k, syncDeg, stages, null);
      if (!complete || !frame) throw new Error("sync keyframe bake incomplete");
      syncPromoted = commitFrame(cache, k, frame, syncDeg);
      syncCount++;
      lastSyncK = k;
      baked = true;
    }
  }

  const proto =
    cornerIndices.map((k) => cache.frames[k]).find(Boolean) ??
    cache.frames[blend.i0] ??
    cache.frames[blend.i1] ??
    cache.frames.find(Boolean);
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
      deg: syncDeg,
      M: gridMFromFrame(proto ?? { dens: new Float32Array(0) }),
      role: opts.role,
      layerId: opts.layerId,
      paramNames,
      bakeMs,
      async: false,
      syncPair: [blend.i0, blend.i1],
    });
    if (syncPromoted.length === cornerIndices.length && cornerIndices.length >= 2) {
      maybeLogBlendPairReady(
        cache,
        opts.layerId,
        blend.i0,
        blend.i1,
        syncDeg,
        bakeMs,
        "sync",
        lastSyncK,
      );
    }
  }

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
  const tLerp = performance.now();
  const { out } = sampleVolumeFromCache(cache);
  if (!out.dens) throw new Error("scalar keyframe missing dens");
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  const M = gridMFromFrame({ dens: out.dens });
  return {
    dens: out.dens,
    cheb: ensured.cheb,
    fitRel: ensured.fitRel,
    M,
    baked: ensured.baked,
    readyCount: ensured.readyCount,
    complete: ensured.complete,
    frames: ensured.frames,
  };
}

/** CPU multilinear sample for isosurface layers with 2+ animated params. */
export function sampleIsoLayerKeyframes(opts: EnsureKeyframesOpts) {
  if (opts.role !== "isosurface") throw new Error("sampleIsoLayerKeyframes requires role isosurface");
  const ensured = ensureLayerKeyframes(opts);
  const cache = caches.get(opts.layerId);
  if (!cache) throw new Error("keyframe cache missing after ensure");
  const tLerp = performance.now();
  const { out } = sampleVolumeFromCache(cache);
  if (!out.dens || !out.gx || !out.gy || !out.gz) {
    throw new Error("isosurface keyframe missing dens/grad");
  }
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;
  const M = gridMFromFrame({ dens: out.dens });
  return {
    dens: out.dens.slice(),
    gx: out.gx.slice(),
    gy: out.gy.slice(),
    gz: out.gz.slice(),
    cheb: ensured.cheb,
    fitRel: ensured.fitRel,
    M,
    baked: ensured.baked,
    readyCount: ensured.readyCount,
    complete: ensured.complete,
  };
}

/** CPU-lerp fx/fy/fz velocity grids at the current param value. */
export function sampleFlowLayerKeyframes(opts: EnsureKeyframesOpts) {
  if (opts.role !== "flow") throw new Error("sampleFlowLayerKeyframes requires role flow");
  const ensured = ensureLayerKeyframes(opts);
  const cache = caches.get(opts.layerId);
  if (!cache) throw new Error("keyframe cache missing after ensure");
  const tLerp = performance.now();
  const { out } = sampleVolumeFromCache(cache);
  if (!out.fx || !out.fy || !out.fz) throw new Error("flow keyframe missing velocity grids");
  lerpMsAcc += performance.now() - tLerp;
  lastLerpMs = lerpMsAcc;

  const M = gridMFromFrame({ fx: out.fx });
  return {
    fx: out.fx.slice(),
    fy: out.fy.slice(),
    fz: out.fz.slice(),
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
  if (!cache || cacheNDims(cache) !== 1) return null;
  const st = getParam(cache.paramNames[0]!);
  if (!st) return null;
  const blend = displayBlendForValue(cache, st.value);
  const a = cache.frames[blend.i0];
  const b = cache.frames[blend.i1];
  if (a && b) return { id: layerId, i0: blend.i0, i1: blend.i1, t: blend.t };
  if (a) return { id: layerId, i0: blend.i0, i1: blend.i0, t: 0 };
  if (b) return { id: layerId, i0: blend.i1, i1: blend.i1, t: 0 };
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
