import { gpu } from "./gpuState.js";
import type { ClipGpuProfile } from "./marchTypes.js";
import { STAMP_COUNT } from "./gpuSubmit.js";

export function noteGpuPresent(submitWallAt: number): void {
  const now = performance.now();
  if (gpu.lastPresentAt > 0) {
    gpu.profilePresentIntervalMs = gpu.profilePresentIntervalMs * 0.85 + (now - gpu.lastPresentAt) * 0.15;
  } else {
    gpu.profilePresentIntervalMs = now - submitWallAt;
  }
  gpu.lastPresentAt = now;
}

export function getClipGpuProfile(): ClipGpuProfile {
  return {
    sceneUploadMs: gpu.profileBakeMs,
    sceneUploadAt: gpu.profileBakeAt,
    marchMs: gpu.profileMarchMs,
    beerMs: gpu.profileBeerMs,
    flowMs: gpu.profileFlowMs,
    fxaaMs: gpu.profileFxaaMs,
    gridMs: gpu.profileGridMs,
    marchFbW: gpu.profileMarchFbW,
    marchFbH: gpu.profileMarchFbH,
    presentWallMs: gpu.profilePresentWallMs,
    presentIntervalMs: gpu.profilePresentIntervalMs,
    lastPresentAt: gpu.lastPresentAt,
    method: gpu.profileMethod,
    gridM: gpu.profileGridM,
    timestamps: gpu.timestampsSupported,
  };
}

export function resetClipGpuProfile(): void {
  gpu.profileBakeMs = 0;
  gpu.profileBakeAt = 0;
  gpu.profileMarchMs = 0;
  gpu.profileBeerMs = 0;
  gpu.profileFlowMs = 0;
  gpu.profileFxaaMs = 0;
  gpu.profileGridMs = 0;
  gpu.profileMarchFbW = 0;
  gpu.profileMarchFbH = 0;
}

/** EMA-smooth one stage's duration (ms), matching the existing 0.7/0.3 blend. */
function smoothMs(prev: number, ms: number): number {
  return prev * 0.7 + ms * 0.3;
}

export function scheduleStampReadback(): void {
  if (!gpu.timestampsSupported || !gpu.stampReadBuf || gpu.stampReadPending) return;
  gpu.stampReadPending = true;
  const readBuf = gpu.stampReadBuf;
  readBuf.mapAsync(GPUMapMode.READ).then(() => {
    const stamps = new BigInt64Array(readBuf.getMappedRange().slice(0, STAMP_COUNT * 8));
    readBuf.unmap();
    gpu.stampReadPending = false;
    // stamps: 0=begin, 1=end-march, 2=end-beer, 3=end-flow, 4=end-fxaa, 5=end-grid.
    const ok = stamps.every((s, i) => i === 0 || s >= stamps[i - 1]!);
    if (ok && stamps[STAMP_COUNT - 1]! > stamps[0]!) {
      const toMs = (a: bigint, b: bigint) => Number(b - a) / 1e6;
      gpu.profileMarchMs = smoothMs(gpu.profileMarchMs, toMs(stamps[0]!, stamps[1]!));
      gpu.profileBeerMs = smoothMs(gpu.profileBeerMs, toMs(stamps[1]!, stamps[2]!));
      gpu.profileFlowMs = smoothMs(gpu.profileFlowMs, toMs(stamps[2]!, stamps[3]!));
      gpu.profileFxaaMs = smoothMs(gpu.profileFxaaMs, toMs(stamps[3]!, stamps[4]!));
      gpu.profileGridMs = smoothMs(gpu.profileGridMs, toMs(stamps[4]!, stamps[5]!));
      gpu.profilePresentWallMs = smoothMs(gpu.profilePresentWallMs, toMs(stamps[0]!, stamps[STAMP_COUNT - 1]!));
    }
  }).catch(() => { gpu.stampReadPending = false; });
}
