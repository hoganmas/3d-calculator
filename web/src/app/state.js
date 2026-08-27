/** Shared mutable app state — no imports from app/. */

export const state = {
  /** Preset param defaults applied on next successful compile/sync. */
  pendingParamSeed: {},

  /** @type {{ render: () => void, syncAllParamSliders?: () => void, syncParamChrome?: () => boolean } | null} */
  exprListApi: null,

  /** Last successful classify/compile summary. */
  lastExprMeta: {
    kind: "bare",
    shade: "volume",
    isoLevel: 0,
    label: "expression → volume",
  },

  /** @type {{ densLayers: any[], constraints: any[], M: number, dens: Float32Array | null } | null} */
  lastSceneBake: null,

  /** @type {Float32Array | null} */
  worldCheb: null,

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
  /** Last CPU Chebyshev→monomial fit breakdown (ms). */
  lastFitTiming: null,
  lastFitRel: NaN,
  lastNCoeff: 0,

  fitTimer: 0,
  lastAnimFitAt: 0,

  /** @type {{ fromAnim?: boolean }} */
  pendingFitOpts: {},

  loopFps: 0,
  loopFpsFrames: 0,
  loopFpsLast: performance.now(),
  cpuMsSmooth: 0,
};

export const FIT_DEBOUNCE_MS = 320;
/** Min ms between full anim refits (DCT / dens CPU lerp). GPU blends update every frame. */
export const ANIM_FIT_MIN_MS = 50;

export const MARCH_DOWNSCALE_MIN = 1;
export const MARCH_DOWNSCALE_MAX = 16;
/** Label only these notches (every integer still snaps). */
export const MARCH_DOWNSCALE_LABELS = new Set([1, 2, 4, 8, 16]);
