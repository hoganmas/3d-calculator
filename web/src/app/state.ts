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

  fitDeg: 32,
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

  /** Advection timestep Δt. */
  flowDt: 0.05,
  /** Animation speed multiplier (applied to Δt). */
  flowSpeed: 0.1,
  /** Velocity clamp vMax (0 = auto). */
  flowVMax: 0,
  /** Fixed Beer density for flow layers (before global scale). */
  flowOpacity: 0.5,
  /** Particle lifetime (seconds) / trail age cap. */
  flowAgeMax: 30.0,
  /** World-unit grid planes and RGB axis guides (+x right, +y forward, +z up). */
  showGridAxes: true,
  /** Particle count for flow advection (GPU instanced billboards). */
  flowParticleCount: 1000,
  /** Trail history length (segments = steps − 1). */
  flowTrailSteps: 32,
  /** Trail stroke width in pixels. */
  flowTrailWidth: 10,

  /** Prod UI quality sliders (0–100). */
  scalarQuality: 50,
  surfaceQuality: 50,
  vectorQuality: 50,
  precisionQuality: 50,

  /** Boot-detected device tier for perf defaults. */
  deviceTier: "desktop" as "mobile" | "tablet" | "desktop",
  webGpuFailed: false,
  /** Timestamp of last manual quality slider move (blocks auto step-down). */
  qualityUserOverrideAt: 0,
  perfAdaptStepDownCount: 0,
};

export const FIT_DEBOUNCE_MS = 320;
/** Min ms between full anim refits (DCT / dens CPU lerp). GPU blends update every frame. */
export const ANIM_FIT_MIN_MS = 50;

export const MARCH_DOWNSCALE_MIN = 1;
export const MARCH_DOWNSCALE_MAX = 16;
/** Label only these notches (every integer still snaps). */
export const MARCH_DOWNSCALE_LABELS = new Set([1, 2, 4, 8, 16]);

export const BOUNDS_SIZE_MIN = 1;
export const BOUNDS_SIZE_MAX = 10;
