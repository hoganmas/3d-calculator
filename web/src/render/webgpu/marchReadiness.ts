import { gpu } from "./gpuState.js";

export function getIsoInterpHermite(): boolean {
  return gpu.isoInterpHermite;
}

/** @returns true if the mode changed (iso pipeline must rebuild) */
export function setIsoInterpHermite(on: boolean): boolean {
  const next = !!on;
  if (next === gpu.isoInterpHermite) return false;
  gpu.isoInterpHermite = next;
  gpu.isoPipeline = null;
  return true;
}

export function isClipBakeGpuReady(): boolean {
  return Boolean(
    gpu.device && gpu.isoPipeline && gpu.beerPipeline && gpu.fxaaPipeline && gpu.ssaoPipeline &&
    gpu.gridPipeline && gpu.labelPipeline,
  );
}

export function isClipMarchReady(): boolean {
  return Boolean(
    isClipBakeGpuReady() && gpu.ctx && gpu.sceneM > 1 &&
    (gpu.densLayerCount > 0 || gpu.sceneConstraints.length > 0),
  );
}
