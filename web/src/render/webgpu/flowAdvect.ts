import flowAdvectWgsl from "./shaders/flowAdvect.wgsl?raw";
import type { FlowLayer, SceneBake } from "../../types/models.js";
import { gpu } from "./gpuState.js";
import { state } from "../../app/state.js";

const MAX_FLOW_LAYERS = 4;

interface GpuFlowLayer {
  id: string | null;
  velBuf: GPUBuffer;
  dyeA: GPUBuffer;
  dyeB: GPUBuffer;
  readDyeA: boolean;
  densByteOffset: number;
}

let pipeline: GPUComputePipeline | null = null;
let paramBuf: GPUBuffer | null = null;
let flowGpu: GpuFlowLayer[] = [];

function ensurePipeline() {
  if (pipeline || !gpu.device) return;
  const mod = gpu.device.createShaderModule({ code: flowAdvectWgsl });
  pipeline = gpu.device.createComputePipeline({
    layout: "auto",
    compute: { module: mod, entryPoint: "csMain" },
  });
  paramBuf = gpu.device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function resetFlowGpuLayers() {
  for (const f of flowGpu) {
    try {
      f.velBuf.destroy();
      f.dyeA.destroy();
      f.dyeB.destroy();
    } catch {
      /* ignore */
    }
  }
  flowGpu = [];
}

export function uploadFlowLayers(
  flowLayers: FlowLayer[],
  M: number,
  half: number,
  flowDensOffsets: number[],
): void {
  if (!gpu.device) return;
  ensurePipeline();
  resetFlowGpuLayers();
  const volN = M * M * M;
  const velFloats = volN * 3;
  for (let i = 0; i < Math.min(flowLayers.length, MAX_FLOW_LAYERS); i++) {
    const f = flowLayers[i]!;
    const packed = new Float32Array(velFloats);
    packed.set(f.fx.subarray(0, volN), 0);
    packed.set(f.fy.subarray(0, volN), volN);
    packed.set(f.fz.subarray(0, volN), volN * 2);
    const velBuf = gpu.device.createBuffer({
      size: Math.max(256, velFloats * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gpu.device.queue.writeBuffer(velBuf, 0, packed);
    const dyeA = gpu.device.createBuffer({
      size: Math.max(256, volN * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const dyeB = gpu.device.createBuffer({
      size: Math.max(256, volN * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    gpu.device.queue.writeBuffer(dyeA, 0, f.dye.subarray(0, volN));
    flowGpu.push({
      id: f.id ?? null,
      velBuf,
      dyeA,
      dyeB,
      readDyeA: true,
      densByteOffset: (flowDensOffsets[i] ?? 0) * 4,
    });
  }
  gpu.flowHalf = half;
  gpu.flowGridM = M;
}

export function tickFlowAdvectionGpu(dtMs: number, bake: SceneBake): boolean {
  if (!gpu.device || !pipeline || !paramBuf || !flowGpu.length) return false;
  const half = bake.half ?? gpu.flowHalf;
  const M = bake.M;
  const speed = state.flowSpeed;
  const dissipation = state.flowDissipation;
  const dt = (dtMs / 1000) * speed;
  const params = new Float32Array([M, half, dt, dissipation]);
  gpu.device.queue.writeBuffer(paramBuf, 0, params);

  const enc = gpu.device.createCommandEncoder();
  for (let li = 0; li < flowGpu.length; li++) {
    const layer = flowGpu[li]!;
    const readBuf = layer.readDyeA ? layer.dyeA : layer.dyeB;
    const writeBuf = layer.readDyeA ? layer.dyeB : layer.dyeA;
    const bind = gpu.device.createBindGroup({
      layout: pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf! } },
        { binding: 1, resource: { buffer: layer.velBuf } },
        { binding: 2, resource: { buffer: readBuf } },
        { binding: 3, resource: { buffer: writeBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline!);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(M / 4), Math.ceil(M / 4), Math.ceil(M / 4));
    pass.end();
    layer.readDyeA = !layer.readDyeA;
    if (gpu.volumeBuf && layer.densByteOffset >= 0) {
      enc.copyBufferToBuffer(writeBuf, 0, gpu.volumeBuf, layer.densByteOffset, M * M * M * 4);
    }
    const flowLayer = bake.flowLayers[li];
    if (flowLayer) {
      void gpu.device.queue
        .onSubmittedWorkDone()
        .then(() => {
          /* CPU mirror optional — skip for perf */
        });
    }
  }
  gpu.device.queue.submit([enc.finish()]);
  state.clipDirty = true;
  return true;
}
