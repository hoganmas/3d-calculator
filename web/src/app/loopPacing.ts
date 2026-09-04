import type { DeviceTier } from "./deviceTier.js";

/**
 * Lava skybox on the GPU iso path — phones share one GPU with WebGPU.
 * Was throttled on mobile (50ms) and tablet (33ms), but the frozen
 * background visibly lagged the every-rAF WebGPU foreground — most
 * noticeable as rotation decelerates, read as the camera itself jittering.
 * A CSS-transform reprojection between throttled renders was tried to keep
 * the GPU-sharing win, but it's only a linear approximation of a nonlinear
 * perspective rotation: during an active drag the per-interval angular delta
 * gets large enough that the pan visibly diverges from the next real render,
 * producing a smear-then-snap every throttle interval — a worse artifact
 * than the lag it replaced. Disabled outright on both tiers instead.
 */
export const GPU_PATH_LAVA_INTERVAL_MOBILE_MS = 0;
export const GPU_PATH_LAVA_INTERVAL_TABLET_MS = 0;

/**
 * Min ms between Three.js presents. 0 = every rAF.
 * WebGL fallback is the viewport, so it always presents.
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
