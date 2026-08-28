import flowIbfvWgsl from "./shaders/flowIbfv.wgsl?raw";
import { gpu, PIPELINE_EPOCH } from "./gpuState.js";
import { state } from "../../app/state.js";
import { hasFlowGpuLayers } from "./flowGpu.js";
import { seedFlowDyeGridlines, FLOW_DYE_CHANNELS } from "../../math/fitVector.js";

export const DEFAULT_FLOW_GRID_M = 32;
export const MAX_FLOW_GRID_M = 64;
const MAX_FLOW_LAYERS = 4;
const PARAM_FLOATS = 12;

let flowFrameIdx = 0;
let flowIbfvBuiltEpoch = -1;

function effectiveVMax(): number {
  if (state.flowVMax > 1e-8) return state.flowVMax;
  const dt = Math.max(state.flowDt, 1e-6);
  return state.flowNoiseScale / dt;
}

/** Speed reference for Beer |V|-modulated opacity (half-box when vMax auto). */
export function effectiveFlowVRef(half: number): number {
  if (state.flowVMax > 1e-8) return state.flowVMax;
  return Math.max(half, 1e-6);
}

function destroyBuffer(buf: GPUBuffer | null): void {
  if (!buf || !gpu.device) return;
  const device = gpu.device;
  void device.queue.onSubmittedWorkDone().then(() => {
    try { buf.destroy(); } catch { /* device lost */ }
  });
}

export function destroyFlowDyeBuffers(): void {
  destroyBuffer(gpu.flowDyeBufA);
  destroyBuffer(gpu.flowDyeBufB);
  gpu.flowDyeBufA = null;
  gpu.flowDyeBufB = null;
  gpu.flowDyeReadIsA = true;
  gpu.flowLayerCount = 0;
  gpu.flowGridM = 0;
  flowFrameIdx = 0;
}

export function ensureFlowDyeBuffers(layerCount: number, gridM: number, half: number): void {
  const { device } = gpu;
  if (!device || layerCount <= 0) {
    destroyFlowDyeBuffers();
    return;
  }
  const M = Math.max(8, Math.min(MAX_FLOW_GRID_M, gridM | 0));
  const layers = Math.min(MAX_FLOW_LAYERS, layerCount | 0);
  const volN = M * M * M;
  const floatCount = volN * layers * FLOW_DYE_CHANNELS;
  const byteSize = Math.max(256, Math.ceil((floatCount * 4) / 256) * 256);

  const resize =
    !gpu.flowDyeBufA ||
    !gpu.flowDyeBufB ||
    gpu.flowGridM !== M ||
    gpu.flowLayerCount !== layers;

  if (resize) {
    destroyFlowDyeBuffers();
    gpu.flowDyeBufA = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gpu.flowDyeBufB = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gpu.flowDyeReadIsA = true;
    const seed = seedFlowDyeGridlines(M, half, state.flowNoiseScale, layers, state.flowGridPoints);
    device.queue.writeBuffer(gpu.flowDyeBufA, 0, seed);
    device.queue.writeBuffer(gpu.flowDyeBufB, 0, seed);
  }

  gpu.flowGridM = M;
  gpu.flowLayerCount = layers;
  gpu.flowHalf = half;
  gpu.flowEpoch++;
  flowFrameIdx = 0;
}

export function getFlowDyeReadBuffer(): GPUBuffer | null {
  if (!gpu.flowDyeBufA || !gpu.flowDyeBufB) return null;
  return gpu.flowDyeReadIsA ? gpu.flowDyeBufA : gpu.flowDyeBufB;
}

async function compileFlowIbfvModule(): Promise<GPUShaderModule | null> {
  if (!gpu.device) return null;
  const mod = gpu.device.createShaderModule({ code: flowIbfvWgsl });
  const info = await mod.getCompilationInfo();
  for (const m of info.messages) {
    if (m.type === "error") {
      console.error("[flowIbfv]", m.message);
      return null;
    }
  }
  return mod;
}

export async function ensureFlowIbfvPipeline(): Promise<boolean> {
  if (!gpu.device) return false;
  if (gpu.flowIbfvPipeline && flowIbfvBuiltEpoch === PIPELINE_EPOCH) return true;

  const mod = await compileFlowIbfvModule();
  if (!mod) return false;

  gpu.flowIbfvPipeline = gpu.device.createComputePipeline({
    layout: "auto",
    compute: { module: mod, entryPoint: "main" },
  });

  if (!gpu.flowIbfvParamBuf) {
    gpu.flowIbfvParamBuf = gpu.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  flowIbfvBuiltEpoch = PIPELINE_EPOCH;
  return true;
}

function packIbfvParams(
  flowGridM: number,
  velGridM: number,
  half: number,
  velBase: number,
  dyeLayerOff: number,
  frameIdx: number,
): Float32Array {
  const f = new Float32Array(PARAM_FLOATS);
  const u = new Uint32Array(f.buffer);
  u[0] = flowGridM;
  u[1] = velGridM;
  f[2] = half;
  f[3] = state.flowAlpha;
  f[4] = state.flowNoiseScale;
  f[5] = state.flowDt;
  f[6] = effectiveVMax();
  u[7] = frameIdx >>> 0;
  u[8] = velBase >>> 0;
  u[9] = dyeLayerOff >>> 0;
  f[10] = state.flowGridPoints ? 1 : 0;
  return f;
}

/** Re-seed dye buffers after grid mode/spacing change (buffers must exist). */
export function reseedFlowDyeBuffers(): void {
  const { device } = gpu;
  if (!device || !gpu.flowDyeBufA || !gpu.flowDyeBufB || gpu.flowLayerCount <= 0) return;
  const seed = seedFlowDyeGridlines(
    gpu.flowGridM,
    gpu.flowHalf,
    state.flowNoiseScale,
    gpu.flowLayerCount,
    state.flowGridPoints,
  );
  device.queue.writeBuffer(gpu.flowDyeBufA, 0, seed);
  device.queue.writeBuffer(gpu.flowDyeBufB, 0, seed);
  flowFrameIdx = 0;
  gpu.flowEpoch++;
}

export function tickFlowIbfv(): void {
  if (!hasFlowGpuLayers() || !gpu.device || !gpu.volumeBuf) return;
  if (!gpu.flowIbfvPipeline || !gpu.flowIbfvParamBuf) return;
  if (!gpu.flowDyeBufA || !gpu.flowDyeBufB || gpu.flowLayerCount <= 0) return;

  const { device } = gpu;
  const Mf = gpu.flowGridM;
  const Mv = gpu.sceneM;
  const volN = Mv * Mv * Mv;
  const layerVolN = Mf * Mf * Mf;
  const half = gpu.flowHalf;
  const readBuf = gpu.flowDyeReadIsA ? gpu.flowDyeBufA! : gpu.flowDyeBufB!;
  const writeBuf = gpu.flowDyeReadIsA ? gpu.flowDyeBufB! : gpu.flowDyeBufA!;

  const enc = device.createCommandEncoder();
  const wg = Math.ceil(Mf / 4);

  for (let layer = 0; layer < gpu.flowLayerCount; layer++) {
    const velBase = (gpu.flowVelBase + layer * volN * 3) >>> 0;
    const dyeLayerOff = layer * layerVolN * FLOW_DYE_CHANNELS;
    device.queue.writeBuffer(
      gpu.flowIbfvParamBuf,
      0,
      packIbfvParams(Mf, Mv, half, velBase, dyeLayerOff, flowFrameIdx),
    );
    const bg = device.createBindGroup({
      layout: gpu.flowIbfvPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpu.flowIbfvParamBuf } },
        { binding: 1, resource: { buffer: gpu.volumeBuf } },
        { binding: 2, resource: { buffer: readBuf } },
        { binding: 3, resource: { buffer: writeBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(gpu.flowIbfvPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wg, wg, wg);
    pass.end();
  }

  device.queue.submit([enc.finish()]);
  gpu.flowDyeReadIsA = !gpu.flowDyeReadIsA;
  flowFrameIdx++;
}

export function resetFlowIbfvPipeline(): void {
  gpu.flowIbfvPipeline = null;
  flowIbfvBuiltEpoch = -1;
}
