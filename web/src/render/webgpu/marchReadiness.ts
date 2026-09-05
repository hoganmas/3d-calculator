import { gpu } from "./gpuState.js";
import { hasFlowGpuLayers } from "./flowGpu.js";

/** Device + buffers sufficient to pack scene volumes (no render pipelines required). */
export function isClipGpuUploadReady(): boolean {
  return Boolean(gpu.device && gpu.colorBuf);
}

export function isClipBakeGpuReady(): boolean {
  return Boolean(
    gpu.device && gpu.isoPipeline && gpu.isoRefinePipeline && gpu.isoUpsamplePipeline &&
    gpu.beerPipeline && gpu.beerRefinePipeline && gpu.fxaaPipeline &&
    gpu.gridPipeline && gpu.labelPipeline,
  );
}

export function isClipMarchReady(): boolean {
  return Boolean(
    isClipBakeGpuReady() && gpu.ctx && gpu.sceneM > 1 &&
    (gpu.densLayerCount > 0 || gpu.sceneConstraints.length > 0 || hasFlowGpuLayers()),
  );
}

/**
 * WebGPU init is in flight (adapter/device requested, or pipelines still
 * compiling) but hasn't yet resolved either way. `gpu.initPromise` is set the
 * instant `initClipBakeGpu` is called and never cleared on success, so this
 * only reads true during the actual gap — false before init is ever attempted
 * (nothing to wait for) and false once it succeeds or fails for good.
 */
export function isClipGpuInitPending(): boolean {
  return Boolean(gpu.initPromise) && !gpu.initFailed && !isClipBakeGpuReady();
}
