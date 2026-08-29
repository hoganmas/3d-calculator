import { gpu } from "./gpuState.js";
import { hasFlowGpuLayers } from "./flowGpu.js";

/** Device + buffers sufficient to pack scene volumes (no render pipelines required). */
export function isClipGpuUploadReady(): boolean {
  return Boolean(gpu.device && gpu.colorBuf);
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
    (gpu.densLayerCount > 0 || gpu.sceneConstraints.length > 0 || hasFlowGpuLayers()),
  );
}
