export const PERF_ADAPT_FPS_THRESHOLD = 24;
export const PERF_ADAPT_FRAME_MS_THRESHOLD = 42;
export const PERF_ADAPT_STREAK_WINDOWS = 3;
export const PERF_ADAPT_STEP = 10;
export const PERF_ADAPT_MIN_QUALITY = 10;
export const PERF_ADAPT_MAX_STEPDOWNS = 3;
export const QUALITY_OVERRIDE_COOLDOWN_MS = 60_000;

export function nextPerfAdaptStreak(
  streak: number,
  loopFps: number,
  frameDtSmooth: number,
): number {
  const lowFps = loopFps > 0 && loopFps < PERF_ADAPT_FPS_THRESHOLD;
  const highFrameMs = frameDtSmooth > PERF_ADAPT_FRAME_MS_THRESHOLD;
  if (lowFps || highFrameMs) return streak + 1;
  return 0;
}

export function shouldTriggerPerfStepDown(streak: number): boolean {
  return streak >= PERF_ADAPT_STREAK_WINDOWS;
}

export function clampAdaptedQuality(q: number): number {
  return Math.min(100, Math.max(PERF_ADAPT_MIN_QUALITY, Math.round(q)));
}

export function stepDownQualityValues(values: {
  precisionQuality: number;
  scalarQuality: number;
  surfaceQuality: number;
  vectorQuality: number;
}) {
  return {
    precisionQuality: clampAdaptedQuality(values.precisionQuality - PERF_ADAPT_STEP),
    scalarQuality: clampAdaptedQuality(values.scalarQuality - PERF_ADAPT_STEP),
    surfaceQuality: clampAdaptedQuality(values.surfaceQuality - PERF_ADAPT_STEP),
    vectorQuality: clampAdaptedQuality(values.vectorQuality - PERF_ADAPT_STEP),
  };
}

export function perfAdaptBlockedByUserOverride(
  now: number,
  qualityUserOverrideAt: number,
): boolean {
  return now - qualityUserOverrideAt < QUALITY_OVERRIDE_COOLDOWN_MS;
}
