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
import { ensureFlowParticlesPipeline } from "./flowParticles.js";
import { syncClipGpuWorldGrid } from "./gridOverlay.js";
import { attachMarchCanvas, bindMarchCanvasContext } from "./marchCanvas.js";
import { isClipBakeGpuReady, isClipGpuUploadReady } from "./marchReadiness.js";
import { startupBegin, startupEnd, startupMark } from "../../app/startupProfile.js";
import { state } from "../../app/state.js";
import { detectDeviceTier, webGpuPowerPreference } from "../../app/deviceTier.js";

let marchPipelinesPromise: Promise<boolean> | null = null;

type MarchPipelinesReadyHandler = (source: string) => void;
let marchPipelinesReadyHandler: MarchPipelinesReadyHandler | null = null;

/** Called once when background pipeline build finishes (avoids circular imports with app/). */
export function setMarchPipelinesReadyHandler(fn: MarchPipelinesReadyHandler | null): void {
  marchPipelinesReadyHandler = fn;
}

function notifyMarchPipelinesReady(source: string): void {
  if (!isClipBakeGpuReady()) return;
  marchPipelinesReadyHandler?.(source);
}

export async function ensurePipelinesForDegree(deg: number): Promise<boolean> {
  const result = await buildPipelines(deg);
  if (result && result.gridRebuildHalf != null) {
    syncClipGpuWorldGrid(result.gridRebuildHalf);
  }
  return result !== false;
}

/** Build march render pipelines in the background (does not block volume upload). */
export function scheduleMarchPipelines(deg = 4): Promise<boolean> {
  if (isClipBakeGpuReady()) return Promise.resolve(true);
  if (!gpu.device) return Promise.resolve(false);
  if (!marchPipelinesPromise) {
    marchPipelinesPromise = (async () => {
      startupBegin("gpu.pipelines.background");
      try {
        await ensurePipelinesForDegree(deg);
        await ensureFlowParticlesPipeline();
        const ok = isClipBakeGpuReady();
        if (ok) notifyMarchPipelinesReady("pipelines.background");
        return ok;
      } finally {
        startupEnd("gpu.pipelines.background");
      }
    })().catch(() => false);
  }
  return marchPipelinesPromise;
}

export async function initClipBakeGpu(
  viewportEl: HTMLElement | null | undefined,
  source = "unknown",
): Promise<boolean> {
  if (isClipGpuUploadReady()) return true;
  if (gpu.initFailed) return false;
  if (gpu.initPromise) return gpu.initPromise;
  startupMark("gpu.init.queued", { source });
  gpu.initPromise = (async () => {
    startupBegin("gpu.init");
    try {
      if (!navigator.gpu) {
        gpu.initFailed = true;
        state.webGpuFailed = true;
        return false;
      }
      startupBegin("gpu.init.adapter");
      const tier = detectDeviceTier();
      state.deviceTier = tier;
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: webGpuPowerPreference(tier),
      });
      startupEnd("gpu.init.adapter", { ok: !!adapter });
      if (!adapter) {
        gpu.initFailed = true;
        state.webGpuFailed = true;
        return false;
      }
      gpu.timestampsSupported = adapter.features.has("timestamp-query");
      const requiredFeatures: GPUFeatureName[] = gpu.timestampsSupported ? ["timestamp-query"] : [];
      startupBegin("gpu.init.device");
      gpu.device = await adapter.requestDevice({ requiredFeatures });
      startupEnd("gpu.init.device");
      gpu.device.lost.then(() => {
        gpu.device = null;
        resetPipelinesOnDeviceLost();
        gpu.initFailed = true;
        gpu.initPromise = null;
        marchPipelinesPromise = null;
        void import("../../app/webglFallback.js").then((m) => m.resetGpuPresentSync());
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
        size: 512,
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
      gpu.isoUpsampleParamBuf = gpu.device.createBuffer({
        size: 16,
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
      if (viewportEl) attachMarchCanvas(viewportEl);
      bindMarchCanvasContext();
      void scheduleMarchPipelines(4);
      startupEnd("gpu.init", { source, ok: isClipGpuUploadReady() });
      return isClipGpuUploadReady();
    } catch (e) {
      console.warn("[clipBakeGpu] init failed", e);
      gpu.initFailed = true;
      state.webGpuFailed = true;
      gpu.device = null;
      resetPipelinesOnDeviceLost();
      startupEnd("gpu.init", { source, ok: false, error: String(e) });
      return false;
    }
  })();
  return gpu.initPromise;
}
