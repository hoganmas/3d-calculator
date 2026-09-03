import type { DeviceTier } from "./deviceTier.js";

/**
 * Lava skybox on the GPU iso path — phones share one GPU with WebGPU.
 * Confirmed the throttle itself (background lagging the every-rAF WebGPU
 * foreground) was the source of "camera jitters as rotation slows" on
 * mobile. Kept throttled for the GPU-sharing win; loop.ts now reprojects the
 * frozen frame via CSS transform between real renders to close the gap.
 */
export const GPU_PATH_LAVA_INTERVAL_MOBILE_MS = 50;
export const GPU_PATH_LAVA_INTERVAL_TABLET_MS = 33;

/**
 * Min ms between Three.js presents. 0 = every rAF.
 * WebGL fallback is the viewport, so it always presents.
 * GPU iso path: lava is a slow skybox under the overlay.
 */
export function threeJsPresentIntervalMs(tier: DeviceTier, gpuPath: boolean): number {
  if (!gpuPath) return 0;
  if (tier === "mobile") return GPU_PATH_LAVA_INTERVAL_MOBILE_MS;
  if (tier === "tablet") return GPU_PATH_LAVA_INTERVAL_TABLET_MS;
  return 0;
}

export function shouldPresentThreeJs(
  now: number,
  lastPresentAt: number,
  intervalMs: number,
): boolean {
  if (intervalMs <= 0) return true;
  if (lastPresentAt <= 0) return true;
  return now - lastPresentAt >= intervalMs;
}
