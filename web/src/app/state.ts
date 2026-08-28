/** Shared mutable app state — no imports from app/. */

import type { ExprListApi, ExprMeta, FitTiming, PresetParamSeed, SceneBake } from "../types/models.js";

export const state = {
  /** Preset param defaults applied on next successful compile/sync. */
  pendingParamSeed: {} as Record<string, PresetParamSeed>,

  exprListApi: null as ExprListApi | null,

  lastExprMeta: {
    kind: "bare",
    shade: "volume",
    isoLevel: 0,
    label: "expression → volume",
  } as ExprMeta,

  lastSceneBake: null as SceneBake | null,

  worldCheb: null as Float32Array | null,

  fitDeg: 20,
  clipDirty: true,
  bakeMsSmooth: 0,
  lastDensSubmitMs: 0,
  densSubmittedThisFrame: false,
  frameDtSmooth: 16,
  lastRafAt: 0,
  lastVolumeM: 0,
  lastMetricsText: "",
  copyMetricsResetTimer: 0,
  lastFitTiming: null as FitTiming | null,
  lastFitRel: NaN,
  lastNCoeff: 0,

  fitTimer: 0,
  lastAnimFitAt: 0,

  pendingFitOpts: {} as { fromAnim?: boolean },

  loopFps: 0,
  loopFpsFrames: 0,
  loopFpsLast: performance.now(),
  cpuMsSmooth: 0,

  /** Spatial stripe density (scales k·x). */
  flowStripeScale: 5.0,
  /** Temporal frequency in rad/s (scales t only). */
  flowTimeScale: 4.0,
  /** Fixed Beer density for flow layers (before global scale). */
  flowOpacity: 0.15,
};

export const FIT_DEBOUNCE_MS = 320;
/** Min ms between full anim refits (DCT / dens CPU lerp). GPU blends update every frame. */
export const ANIM_FIT_MIN_MS = 50;

export const MARCH_DOWNSCALE_MIN = 1;
export const MARCH_DOWNSCALE_MAX = 16;
/** Label only these notches (every integer still snaps). */
export const MARCH_DOWNSCALE_LABELS = new Set([1, 2, 4, 8, 16]);
