import { MAX_GRAD_STOPS } from "../../model/expressions.js";
import {
  gpu,
  MAX_DENS_LAYERS,
  resetPipelinesOnDeviceLost,
  DEFAULT_DENS_RGB,
  DEFAULT_DENS_RGB2,
} from "./gpuState.js";
import { writeLayerColors } from "./uniforms.js";
import { ensureVolumeBuf } from "./sceneUpload.js";
import { ensurePipelinesForDegree as buildPipelines } from "./pipelines.js";
import { syncClipGpuWorldGrid } from "./gridOverlay.js";
import { attachMarchCanvas, bindMarchCanvasContext } from "./marchCanvas.js";
import { isClipBakeGpuReady } from "./marchReadiness.js";

export async function ensurePipelinesForDegree(deg: number): Promise<boolean> {
  const result = await buildPipelines(deg);
  if (result && result.gridRebuildHalf != null) {
    syncClipGpuWorldGrid(result.gridRebuildHalf);
  }
  return result !== false;
}

export async function initClipBakeGpu(viewportEl: HTMLElement | null | undefined): Promise<boolean> {
  if (isClipBakeGpuReady()) return true;
  if (gpu.initFailed) return false;
  if (gpu.initPromise) return gpu.initPromise;
  gpu.initPromise = (async () => {
    try {
      if (!navigator.gpu) { gpu.initFailed = true; return false; }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) { gpu.initFailed = true; return false; }
      gpu.timestampsSupported = adapter.features.has("timestamp-query");
      const requiredFeatures: GPUFeatureName[] = gpu.timestampsSupported ? ["timestamp-query"] : [];
      gpu.device = await adapter.requestDevice({ requiredFeatures });
      gpu.device.lost.then(() => {
        gpu.device = null;
        resetPipelinesOnDeviceLost();
        gpu.initFailed = true;
      });
      if (gpu.timestampsSupported) {
        gpu.stampQuerySet = gpu.device.createQuerySet({ type: "timestamp", count: 2 });
        gpu.stampResolveBuf = gpu.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        gpu.stampReadBuf = gpu.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }
      gpu.drawParamBuf = gpu.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.drawParamBufBeer = gpu.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.fxaaParamBuf = gpu.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.ssaoParamBuf = gpu.device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.gridParamBuf = gpu.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.fxaaSampler = gpu.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      });
      gpu.colorBuf = gpu.device.createBuffer({
        size: MAX_DENS_LAYERS * MAX_GRAD_STOPS * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      writeLayerColors(gpu.device, gpu.colorBuf, [[DEFAULT_DENS_RGB, DEFAULT_DENS_RGB2]]);
      ensureVolumeBuf(8 * 8 * 8);
      await ensurePipelinesForDegree(4);
      if (viewportEl) attachMarchCanvas(viewportEl);
      bindMarchCanvasContext();
      return isClipBakeGpuReady();
    } catch (e) {
      console.warn("[clipBakeGpu] init failed", e);
      gpu.initFailed = true;
      gpu.device = null;
      resetPipelinesOnDeviceLost();
      return false;
    }
  })();
  return gpu.initPromise;
}
