import { gpu } from "./gpuState.js";
import type { ClipGpuProfile } from "./marchTypes.js";

export function noteGpuPresent(submitWallAt: number): void {
  const now = performance.now();
  gpu.profilePresentWallMs = gpu.profilePresentWallMs * 0.85 + (now - submitWallAt) * 0.15;
  if (gpu.lastPresentAt > 0) {
    gpu.profilePresentIntervalMs = gpu.profilePresentIntervalMs * 0.85 + (now - gpu.lastPresentAt) * 0.15;
  } else {
    gpu.profilePresentIntervalMs = now - submitWallAt;
  }
  gpu.lastPresentAt = now;
}

export function getClipGpuProfile(): ClipGpuProfile {
  return {
    idctMs: gpu.profileBakeMs,
    marchMs: gpu.profileMarchMs,
    marchFbW: gpu.profileMarchFbW,
    marchFbH: gpu.profileMarchFbH,
    presentWallMs: gpu.profilePresentWallMs,
    presentIntervalMs: gpu.profilePresentIntervalMs,
    lastPresentAt: gpu.lastPresentAt,
    method: gpu.profileMethod,
    gridM: gpu.profileGridM,
    timestamps: gpu.timestampsSupported,
    isoInterp: gpu.isoInterpHermite ? "hermite" : "trilinear",
  };
}

export function resetClipGpuProfile(): void {
  gpu.profileBakeMs = 0;
  gpu.profileMarchMs = 0;
  gpu.profileMarchFbW = 0;
  gpu.profileMarchFbH = 0;
}

export function scheduleStampReadback(): void {
  if (!gpu.timestampsSupported || !gpu.stampReadBuf || gpu.stampReadPending) return;
  gpu.stampReadPending = true;
  const readBuf = gpu.stampReadBuf;
  readBuf.mapAsync(GPUMapMode.READ).then(() => {
    const stamps = new BigInt64Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    gpu.stampReadPending = false;
    if (stamps[1] > stamps[0]) {
      gpu.profileMarchMs = gpu.profileMarchMs * 0.7 + Number(stamps[1] - stamps[0]) / 1e6 * 0.3;
    }
  }).catch(() => { gpu.stampReadPending = false; });
}
