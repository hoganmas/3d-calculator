import type { PerspectiveCamera } from "three";
import { gpu, PIPELINE_EPOCH } from "./gpuState.js";
import { state } from "../../app/state.js";
import { hasFlowGpuLayers } from "./flowGpu.js";
import {
  advectFlowParticles,
  DEFAULT_FLOW_TRAIL_STEPS,
  FLOW_PARTICLE_STRIDE,
  FLOW_TRAIL_SLOT_STRIDE,
  flowTrailBaseIndex,
  buildFlowParticleDensityGrids,
  FLOW_PARTICLE_DENSITY_GRID,
  MAX_FLOW_TRAIL_STEPS,
  pushFlowTrailHist,
  redistributeOvercrowdedFlowParticles,
  flowSpeedPercentileMinMax,
  resolveFlowParticleColorRange,
  sampleVelGridAt,
  seedFlowParticles,
  seedFlowTrailHist,
  sortFlowParticlesByDepth,
  updateFlowTrailHead,
  type FlowParticleLayerVel,
} from "../../math/fitVector.js";
import { effectiveFlowDt } from "./flowIbfv.js";
import { getFlowParticlesShader } from "./shaders/compose.js";

export const DEFAULT_FLOW_PARTICLE_COUNT = 1000;
export const MAX_FLOW_PARTICLE_COUNT = 32000;
/** Shift trail ring buffer every N frames to stretch visible history in time. */
const TRAIL_PUSH_INTERVAL = 3;

let posAge: Float32Array | null = null;
let layerIds: Uint32Array | null = null;
let trailHist: Float32Array | null = null;
let sortScratch: Uint32Array | null = null;
let flowParticleFrameIdx = 0;
let trailPushCounter = 0;
let flowParticlesBuiltEpoch = -1;

function trailSteps(): number {
  return Math.max(2, Math.min(MAX_FLOW_TRAIL_STEPS, state.flowTrailSteps | 0 || DEFAULT_FLOW_TRAIL_STEPS));
}

function effectiveVMax(): number {
  if (state.flowVMax > 1e-8) return state.flowVMax;
  return state.flowNoiseScale / effectiveFlowDt();
}

function destroyBuffer(buf: GPUBuffer | null): void {
  if (!buf || !gpu.device) return;
  const device = gpu.device;
  void device.queue.onSubmittedWorkDone().then(() => {
    try { buf.destroy(); } catch { /* device lost */ }
  });
}

/** UI setting: particles seeded per flow layer (total = perLayer × layerCount, capped). */
export function flowParticlesPerLayerSetting(): number {
  return Math.max(
    100,
    Math.min(
      MAX_FLOW_PARTICLE_COUNT,
      state.flowParticleCount | 0 || DEFAULT_FLOW_PARTICLE_COUNT,
    ),
  );
}

export function flowParticleBudget(layerCount: number): { perLayer: number; total: number } {
  const layers = Math.max(1, layerCount | 0);
  const perLayer = flowParticlesPerLayerSetting();
  let total = perLayer * layers;
  if (total > MAX_FLOW_PARTICLE_COUNT) {
    const cappedPer = Math.max(100, Math.floor(MAX_FLOW_PARTICLE_COUNT / layers));
    return { perLayer: cappedPer, total: cappedPer * layers };
  }
  return { perLayer, total };
}

export function destroyFlowParticleBuffers(): void {
  destroyBuffer(gpu.flowParticleBuf);
  destroyBuffer(gpu.flowParticleLayerBuf);
  destroyBuffer(gpu.flowTrailBuf);
  gpu.flowParticleBuf = null;
  gpu.flowParticleLayerBuf = null;
  gpu.flowTrailBuf = null;
  gpu.flowParticleCount = 0;
  gpu.flowParticlesPerLayer = 0;
  posAge = null;
  layerIds = null;
  trailHist = null;
  sortScratch = null;
  flowParticleFrameIdx = 0;
  trailPushCounter = 0;
}

export function ensureFlowParticleBuffers(layerCount: number, half: number): void {
  const { device } = gpu;
  if (!device || layerCount <= 0) {
    destroyFlowParticleBuffers();
    return;
  }

  const { perLayer, total: count } = flowParticleBudget(layerCount);
  const steps = trailSteps();
  const posBytes = Math.max(256, Math.ceil((count * FLOW_PARTICLE_STRIDE * 4) / 256) * 256);
  const trailFloats = count * MAX_FLOW_TRAIL_STEPS * FLOW_TRAIL_SLOT_STRIDE;
  const trailBytes = Math.max(256, Math.ceil((trailFloats * 4) / 256) * 256);
  gpu.flowLayerCount = layerCount;
  gpu.flowHalf = half;
  const resize =
    !gpu.flowParticleBuf ||
    !gpu.flowParticleLayerBuf ||
    !gpu.flowTrailBuf ||
    gpu.flowParticleCount !== count ||
    gpu.flowParticlesPerLayer !== perLayer ||
    gpu.flowLayerCount !== layerCount ||
    gpu.flowParticleBuf.size < posBytes ||
    gpu.flowTrailBuf.size < trailBytes;

  if (resize) {
    destroyFlowParticleBuffers();
    const layerBytes = Math.max(256, Math.ceil((count * 4) / 256) * 256);
    gpu.flowParticleBuf = device.createBuffer({
      size: posBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gpu.flowParticleLayerBuf = device.createBuffer({
      size: layerBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gpu.flowTrailBuf = device.createBuffer({
      size: trailBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gpu.flowParticleCount = count;
    gpu.flowParticlesPerLayer = perLayer;
    const seeded = seedFlowParticles(
      perLayer,
      layerCount,
      half,
      state.flowNoiseScale,
      state.flowGridPoints,
    );
    posAge = seeded.posAge;
    layerIds = seeded.layerIds;
    trailHist = new Float32Array(count * MAX_FLOW_TRAIL_STEPS * FLOW_TRAIL_SLOT_STRIDE);
    seedFlowTrailHist(posAge, trailHist, steps, count);
    sortScratch = new Uint32Array(count);
    const layers = extractFlowLayers();
    if (layers.length) refreshTrailSpeeds(layers, gpu.sceneM, half, count, steps);
    device.queue.writeBuffer(gpu.flowParticleBuf, 0, posAge);
    device.queue.writeBuffer(gpu.flowParticleLayerBuf, 0, layerIds);
    device.queue.writeBuffer(gpu.flowTrailBuf, 0, trailHist);
  }

  flowParticleFrameIdx = 0;
  trailPushCounter = 0;
}

export function reseedFlowParticles(): void {
  const { device } = gpu;
  if (!device || gpu.flowLayerCount <= 0) return;
  const { perLayer, total: count } = flowParticleBudget(gpu.flowLayerCount);
  if (
    !gpu.flowParticleBuf ||
    !gpu.flowParticleLayerBuf ||
    !gpu.flowTrailBuf ||
    gpu.flowParticleCount !== count ||
    gpu.flowParticlesPerLayer !== perLayer
  ) {
    ensureFlowParticleBuffers(gpu.flowLayerCount, gpu.flowHalf);
    return;
  }
  const steps = trailSteps();
  const seeded = seedFlowParticles(
    perLayer,
    gpu.flowLayerCount,
    gpu.flowHalf,
    state.flowNoiseScale,
    state.flowGridPoints,
  );
  posAge = seeded.posAge;
  layerIds = seeded.layerIds;
  if (!trailHist || trailHist.length < count * MAX_FLOW_TRAIL_STEPS * FLOW_TRAIL_SLOT_STRIDE) {
    trailHist = new Float32Array(count * MAX_FLOW_TRAIL_STEPS * FLOW_TRAIL_SLOT_STRIDE);
  }
  seedFlowTrailHist(posAge, trailHist, steps, count);
  const layers = extractFlowLayers();
  if (layers.length) refreshTrailSpeeds(layers, gpu.sceneM, gpu.flowHalf, count, steps);
  device.queue.writeBuffer(gpu.flowParticleBuf, 0, posAge);
  device.queue.writeBuffer(gpu.flowParticleLayerBuf, 0, layerIds);
  if (gpu.flowTrailBuf) device.queue.writeBuffer(gpu.flowTrailBuf, 0, trailHist);
  flowParticleFrameIdx = 0;
  trailPushCounter = 0;
}

function extractFlowLayers(): FlowParticleLayerVel[] {
  const packed = gpu.scenePacked;
  const M = gpu.sceneM;
  if (!packed || M <= 0 || gpu.flowLayerCount <= 0) return [];
  const volN = M * M * M;
  const layers: FlowParticleLayerVel[] = [];
  let off = gpu.flowVelBase;
  for (let i = 0; i < gpu.flowLayerCount; i++) {
    layers.push({
      fx: packed.subarray(off, off + volN),
      fy: packed.subarray(off + volN, off + volN * 2),
      fz: packed.subarray(off + volN * 2, off + volN * 3),
    });
    off += volN * 3;
  }
  return layers;
}

function reorderBySortOrder(): void {
  if (!posAge || !layerIds || !sortScratch) return;
  const n = sortScratch.length;
  const sortedPos = new Float32Array(posAge.length);
  const sortedLayers = new Uint32Array(layerIds.length);
  const sortedTrail = trailHist ? new Float32Array(n * MAX_FLOW_TRAIL_STEPS * FLOW_TRAIL_SLOT_STRIDE) : null;
  for (let i = 0; i < n; i++) {
    const src = sortScratch[i]!;
    const so = src * FLOW_PARTICLE_STRIDE;
    const do_ = i * FLOW_PARTICLE_STRIDE;
    sortedPos[do_] = posAge[so]!;
    sortedPos[do_ + 1] = posAge[so + 1]!;
    sortedPos[do_ + 2] = posAge[so + 2]!;
    sortedPos[do_ + 3] = posAge[so + 3]!;
    sortedPos[do_ + 4] = posAge[so + 4]!;
    sortedLayers[i] = layerIds[src]!;
    if (sortedTrail && trailHist) {
      const ss = flowTrailBaseIndex(src);
      const ds = flowTrailBaseIndex(i);
      sortedTrail.set(
        trailHist.subarray(ss, ss + MAX_FLOW_TRAIL_STEPS * FLOW_TRAIL_SLOT_STRIDE),
        ds,
      );
    }
  }
  posAge.set(sortedPos);
  layerIds.set(sortedLayers);
  if (sortedTrail && trailHist) trailHist.set(sortedTrail);
}

export function tickFlowParticles(
  ro: [number, number, number],
  viewDir: [number, number, number],
): void {
  if (!hasFlowGpuLayers() || state.flowVizMode !== "particles") return;
  if (!posAge || !layerIds || !sortScratch || !gpu.device) return;
  if (!gpu.flowParticleBuf || !gpu.flowParticleLayerBuf) return;

  const layers = extractFlowLayers();
  if (!layers.length) return;

  const steps = trailSteps();
  const densityGrids = buildFlowParticleDensityGrids(
    posAge,
    layerIds,
    gpu.flowParticleCount,
    gpu.flowHalf,
    gpu.flowLayerCount,
  );
  advectFlowParticles(
    posAge,
    layerIds,
    layers,
    gpu.sceneM,
    {
      dt: effectiveFlowDt(),
      vMax: effectiveVMax(),
      half: gpu.flowHalf,
      alpha: 0,
      gridSpacing: state.flowNoiseScale,
      gridPoints: state.flowGridPoints,
      ageMax: state.flowAgeMax,
      frameIdx: flowParticleFrameIdx,
    },
    trailHist,
    steps,
    densityGrids,
    FLOW_PARTICLE_DENSITY_GRID,
  );
  redistributeOvercrowdedFlowParticles(
    posAge,
    layerIds,
    gpu.flowParticleCount,
    gpu.flowHalf,
    state.flowNoiseScale,
    state.flowGridPoints,
    flowParticleFrameIdx,
    trailHist,
    steps,
    FLOW_PARTICLE_DENSITY_GRID,
    layers,
    gpu.sceneM,
    densityGrids,
  );
  refreshTrailSpeeds(layers, gpu.sceneM, gpu.flowHalf, gpu.flowParticleCount, steps);
  if (trailHist && gpu.flowTrailBuf) {
    updateFlowTrailHead(posAge, trailHist, gpu.flowParticleCount);
    trailPushCounter++;
    if (trailPushCounter >= TRAIL_PUSH_INTERVAL) {
      pushFlowTrailHist(posAge, trailHist, steps, gpu.flowParticleCount);
      trailPushCounter = 0;
    }
  }
  sortFlowParticlesByDepth(posAge, sortScratch, ro, viewDir);
  reorderBySortOrder();

  const { device } = gpu;
  device.queue.writeBuffer(gpu.flowParticleBuf, 0, posAge);
  device.queue.writeBuffer(gpu.flowParticleLayerBuf, 0, layerIds);
  if (trailHist && gpu.flowTrailBuf) {
    device.queue.writeBuffer(gpu.flowTrailBuf, 0, trailHist);
  }
  flowParticleFrameIdx++;
}

async function compileFlowParticlesModule(): Promise<GPUShaderModule | null> {
  if (!gpu.device) return null;
  const mod = gpu.device.createShaderModule({ code: getFlowParticlesShader() });
  const info = await mod.getCompilationInfo();
  for (const m of info.messages) {
    if (m.type === "error") {
      console.error("[flowParticles]", m.message);
      return null;
    }
  }
  return mod;
}

export async function ensureFlowParticlesPipeline(): Promise<boolean> {
  if (!gpu.device) return false;
  if (gpu.flowParticlesPipeline && flowParticlesBuiltEpoch === PIPELINE_EPOCH) return true;

  const mod = await compileFlowParticlesModule();
  if (!mod) return false;

  const blendPremul: GPUBlendState = {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };

  gpu.flowParticlesPipeline = gpu.device.createRenderPipeline({
    layout: "auto",
    vertex: { module: mod, entryPoint: "vsMain" },
    fragment: {
      module: mod,
      entryPoint: "fsMain",
      targets: [{ format: gpu.canvasFormat, blend: blendPremul }],
    },
    primitive: { topology: "triangle-list" },
  });

  if (!gpu.flowParticlesParamBuf) {
    gpu.flowParticlesParamBuf = gpu.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  flowParticlesBuiltEpoch = PIPELINE_EPOCH;
  return true;
}

function flowFieldSpeedRange(): [number, number] | null {
  const layers = extractFlowLayers();
  if (!layers.length) return null;
  let lo = Infinity;
  let hi = 0;
  for (const L of layers) {
    const [l, h] = flowSpeedPercentileMinMax(L.fx, L.fy, L.fz, 0.08, 0.92);
    if (l < lo) lo = l;
    if (h > hi) hi = h;
  }
  if (!Number.isFinite(lo) || !(hi > lo + 1e-8)) return null;
  return [lo, hi];
}

function resolveFlowSpeedRange(): [number, number] {
  return resolveFlowParticleColorRange(
    trailHist,
    gpu.flowParticleCount,
    trailSteps(),
    effectiveVMax(),
    flowFieldSpeedRange(),
  );
}

function refreshTrailSpeeds(
  layers: FlowParticleLayerVel[],
  M: number,
  half: number,
  count: number,
  trailSteps: number,
): void {
  if (!posAge || !layerIds) return;
  for (let i = 0; i < count; i++) {
    const layer = layerIds[i]!;
    const vel = layers[layer];
    if (!vel) continue;
    const o = i * FLOW_PARTICLE_STRIDE;
    const [vx, vy, vz] = sampleVelGridAt(
      vel.fx, vel.fy, vel.fz, M, half,
      posAge[o]!, posAge[o + 1]!, posAge[o + 2]!,
    );
    posAge[o + 4] = Math.hypot(vx, vy, vz);
    if (!trailHist || trailSteps < 2) continue;
    const ho = flowTrailBaseIndex(i);
    for (let j = 0; j < trailSteps; j++) {
      const to = ho + j * FLOW_TRAIL_SLOT_STRIDE;
      const [tvx, tvy, tvz] = sampleVelGridAt(
        vel.fx, vel.fy, vel.fz, M, half,
        trailHist[to]!, trailHist[to + 1]!, trailHist[to + 2]!,
      );
      trailHist[to + 4] = Math.hypot(tvx, tvy, tvz);
    }
  }
}

function packFlowParticleParams(
  viewProj: Float32Array,
  ro: [number, number, number],
  half: number,
  dirMatrix: Float64Array | Float32Array | number[],
  fbW: number,
  fbH: number,
  cameraFovDeg: number,
  speedRange: [number, number],
): Float32Array {
  const steps = trailSteps();
  const segCount = Math.max(1, steps - 1);
  const f = new Float32Array(64);
  const u = new Uint32Array(f.buffer);
  f.set(viewProj, 0);
  f[16] = ro[0]; f[17] = ro[1]; f[18] = ro[2]; f[19] = half;
  f[20] = dirMatrix[0]; f[21] = dirMatrix[1]; f[22] = dirMatrix[2];
  f[24] = dirMatrix[3]; f[25] = dirMatrix[4]; f[26] = dirMatrix[5];
  f[28] = dirMatrix[6]; f[29] = dirMatrix[7]; f[30] = dirMatrix[8];
  f[32] = fbW; f[33] = fbH;
  f[34] = gpu.flowLayerStart;
  f[35] = state.flowAgeMax;
  f[36] = state.flowOpacity;
  f[37] = 0;
  u[38] = gpu.flowParticleCount >>> 0;
  u[39] = steps >>> 0;
  u[40] = segCount >>> 0;
  u[41] = 0;
  f[42] = state.flowTrailWidth;
  f[43] = speedRange[0];
  f[44] = speedRange[1];
  const dt = effectiveFlowDt();
  const vMax = state.flowVMax > 1e-8 ? state.flowVMax : state.flowNoiseScale / dt;
  f[45] = Math.max(half * 0.2, dt * vMax * 2.5 * TRAIL_PUSH_INTERVAL);
  const fovRad = (cameraFovDeg * Math.PI) / 180;
  f[46] = (2 * Math.tan(fovRad / 2)) / Math.max(fbH, 1);
  u[47] = gpu.sceneM >>> 0;
  f[48] = gpu.flowVelBase;
  return f;
}

function recordRibbonDraw(
  device: GPUDevice,
  sceneView: GPUTextureView,
  occlIsoView: GPUTextureView,
  viewProj: Float32Array,
  ro: [number, number, number],
  dirMatrix: Float64Array | Float32Array | number[],
  half: number,
  fbW: number,
  fbH: number,
  cameraFovDeg: number,
  speedRange: [number, number],
  vertexCount: number,
  instanceCount: number,
): void {
  if (vertexCount <= 0 || instanceCount <= 0) return;
  device.queue.writeBuffer(
    gpu.flowParticlesParamBuf!,
    0,
    packFlowParticleParams(viewProj, ro, half, dirMatrix, fbW, fbH, cameraFovDeg, speedRange),
  );
  const bg = device.createBindGroup({
    layout: gpu.flowParticlesPipeline!.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpu.flowParticlesParamBuf! } },
      { binding: 2, resource: { buffer: gpu.flowParticleLayerBuf! } },
      { binding: 3, resource: { buffer: gpu.colorBuf! } },
      { binding: 4, resource: occlIsoView },
      { binding: 5, resource: { buffer: gpu.flowTrailBuf! } },
    ],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: sceneView, loadOp: "load", storeOp: "store" }],
  });
  pass.setPipeline(gpu.flowParticlesPipeline!);
  pass.setBindGroup(0, bg);
  pass.draw(vertexCount, instanceCount);
  pass.end();
  device.queue.submit([enc.finish()]);
}

export function drawFlowParticlesPass(
  camera: PerspectiveCamera,
  sceneView: GPUTextureView,
  occlIsoView: GPUTextureView,
  ro: [number, number, number],
  dirMatrix: Float64Array | Float32Array | number[],
  half: number,
  fbW: number,
  fbH: number,
): void {
  if (state.flowVizMode !== "particles") return;
  if (!gpu.flowParticlesPipeline || !gpu.flowParticlesParamBuf) return;
  if (!gpu.flowParticleBuf || !gpu.flowParticleLayerBuf || !gpu.flowTrailBuf || !gpu.colorBuf) return;
  if (gpu.flowParticleCount <= 0) return;

  const { device } = gpu;
  if (!device) return;
  camera.updateMatrixWorld(true);
  const viewProj = new Float32Array(16);
  const e = camera.projectionMatrix.elements;
  const v = camera.matrixWorldInverse.elements;
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      viewProj[c * 4 + r] =
        e[0 * 4 + r] * v[c * 4 + 0] +
        e[1 * 4 + r] * v[c * 4 + 1] +
        e[2 * 4 + r] * v[c * 4 + 2] +
        e[3 * 4 + r] * v[c * 4 + 3];
    }
  }

  const steps = trailSteps();
  const segCount = Math.max(1, steps - 1);
  const speedRange = resolveFlowSpeedRange();
  recordRibbonDraw(
    device,
    sceneView,
    occlIsoView,
    viewProj,
    ro,
    dirMatrix,
    half,
    fbW,
    fbH,
    camera.fov,
    speedRange,
    segCount * 6,
    gpu.flowParticleCount,
  );
}

export function resetFlowParticlesPipeline(): void {
  gpu.flowParticlesPipeline = null;
  flowParticlesBuiltEpoch = -1;
}
