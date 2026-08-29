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

  /** IBFV grid injection rate α (0 = pure advection; >0 refreshes lines at upstream footpoint). */
  flowAlpha: 0.1,
  /** Spatial grid spacing for IBFV injection (world units). */
  flowNoiseScale: 0.3,
  /** When true, inject at grid points; otherwise axis-aligned grid lines. */
  flowGridPoints: false,
  /** Advection timestep Δt. */
  flowDt: 0.05,
  /** Animation speed multiplier (applied to Δt). */
  flowSpeed: 0.1,
  /** Velocity clamp vMax (0 = auto: noiseScale/dt). */
  flowVMax: 0,
  /** Fixed Beer density for flow layers (before global scale). */
  flowOpacity: 0.5,
  /** Age (seconds) at which advected dye reaches gradient color 2 / particle respawn. */
  flowAgeMax: 30.0,
  /** World-unit grid planes and RGB axis guides (+ x/y/z labels). */
  showGridAxes: true,

  /** Flow visualization: advected particles (depth-sorted) or IBFV dye grid. */
  flowVizMode: "particles" as "particles" | "ibfv",
  /** Particle count for flow advection (GPU instanced billboards). */
  flowParticleCount: 1000,
  /** Trail history length (segments = steps − 1). */
  flowTrailSteps: 32,
  /** Trail stroke width in pixels. */
  flowTrailWidth: 10,
};

export const FIT_DEBOUNCE_MS = 320;
/** Min ms between full anim refits (DCT / dens CPU lerp). GPU blends update every frame. */
export const ANIM_FIT_MIN_MS = 50;

export const MARCH_DOWNSCALE_MIN = 1;
export const MARCH_DOWNSCALE_MAX = 16;
/** Label only these notches (every integer still snaps). */
export const MARCH_DOWNSCALE_LABELS = new Set([1, 2, 4, 8, 16]);
