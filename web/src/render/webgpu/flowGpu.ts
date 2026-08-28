import { gpu } from "./gpuState.js";

export function hasFlowGpuLayers(): boolean {
  return gpu.flowLayerStart >= 0;
}
