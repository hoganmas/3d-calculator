import { PANEL_LAYOUT_MQ } from "./panelLayout.js";

export type DeviceTier = "mobile" | "tablet" | "desktop";

export const COARSE_POINTER_MQ = "(pointer: coarse)";

export interface BootQualityPreset {
  precisionQuality: number;
  scalarQuality: number;
  surfaceQuality: number;
  vectorQuality: number;
}

export function isCoarsePointer(): boolean {
  return typeof window !== "undefined" && window.matchMedia(COARSE_POINTER_MQ).matches;
}

export function isNarrowViewport(): boolean {
  return typeof window !== "undefined" && window.matchMedia(PANEL_LAYOUT_MQ).matches;
}

export function detectDeviceTier(opts: { webGpuFailed?: boolean } = {}): DeviceTier {
  if (typeof window === "undefined") return "desktop";

  const coarse = isCoarsePointer();
  const narrow = isNarrowViewport();
  const cores = typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 8 : 8;
  const lowCpu = cores <= 4;

  if (coarse && narrow) return "mobile";
  if (coarse) return "tablet";
  if ((opts.webGpuFailed || lowCpu) && narrow) return "mobile";
  return "desktop";
}

export function bootQualityForTier(tier: DeviceTier): BootQualityPreset {
  switch (tier) {
    case "mobile":
      return { precisionQuality: 25, scalarQuality: 25, surfaceQuality: 25, vectorQuality: 20 };
    case "tablet":
      return { precisionQuality: 40, scalarQuality: 40, surfaceQuality: 40, vectorQuality: 35 };
    default:
      return { precisionQuality: 50, scalarQuality: 50, surfaceQuality: 50, vectorQuality: 50 };
  }
}

export function webGpuPowerPreference(tier: DeviceTier): GPUPowerPreference {
  return tier === "mobile" ? "low-power" : "high-performance";
}

/** Finest iso compose divisor floor. Mobile Hermite refine at 2×/1× misses 30fps. */
export function isoComposeDownscaleFloor(tier: DeviceTier): number {
  switch (tier) {
    case "mobile":
      return 4;
    case "tablet":
      return 2;
    default:
      return 1;
  }
}

export function effectiveIsoComposeDownscale(slider: number, tier: DeviceTier): number {
  const n = Math.min(16, Math.max(1, Math.round(slider) || 1));
  return Math.max(n, isoComposeDownscaleFloor(tier));
}

/** Occupancy pass can be coarser than the Hermite refine. Shader clamps to ≥16. */
export function coarseIsoSteps(isoSteps: number, tier: DeviceTier): number {
  const steps = Math.min(192, Math.max(16, isoSteps | 0));
  return tier === "mobile" ? Math.min(steps, 16) : steps;
}

/** Cap refine iso-step count so boot quality 25 (44 steps) does not outrun HTML's 32. */
export function clampIsoStepsForTier(isoSteps: number, tier: DeviceTier): number {
  const max = tier === "mobile" ? 32 : 192;
  return Math.min(max, Math.max(16, isoSteps | 0));
}
