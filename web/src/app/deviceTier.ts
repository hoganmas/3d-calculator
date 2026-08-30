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
