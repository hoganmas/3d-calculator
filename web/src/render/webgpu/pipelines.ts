import { MAX_GRAD_STOPS } from "../../model/expressions.js";
import { gpu, PIPELINE_EPOCH } from "./gpuState.js";
import { MAX_DENS_LAYERS } from "./gpuState.js";
import { startupBegin, startupEnd } from "../../app/startupProfile.js";
import {
  getIsoShader,
  getBeerShader,
  getGridShader,
  getAxisLabelShader,
  getFxaaShader,
  getSsaoShader,
} from "./shaders/compose.js";

export interface PipelineBuildResult {
  gridRebuildHalf?: number;
}

async function compileChecked(label: string, code: string): Promise<GPUShaderModule> {
  if (!gpu.device) throw new Error("GPU device not initialized");
  const mod = gpu.device.createShaderModule({ code });
  const info = await mod.getCompilationInfo();
  for (const m of info.messages) {
    if (m.type === "error") throw new Error(`${label}: ${m.message}`);
  }
  return mod;
}

export async function ensurePipelinesForDegree(_deg: number): Promise<PipelineBuildResult | false> {
  if (!gpu.device) return false;
  if (
    gpu.isoPipeline && gpu.beerPipeline && gpu.fxaaPipeline && gpu.ssaoPipeline &&
    gpu.gridPipeline && gpu.labelPipeline &&
    gpu.builtEpoch === PIPELINE_EPOCH
  ) {
    return {};
  }

  startupBegin("gpu.pipelines.compile-shaders");
  gpu.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const [isoMod, beerMod, gridMod, labelMod, fxaaMod, ssaoMod] = await Promise.all([
    compileChecked("iso", getIsoShader(MAX_GRAD_STOPS)),
    compileChecked("beer", getBeerShader(MAX_GRAD_STOPS, MAX_DENS_LAYERS)),
    compileChecked("grid", getGridShader()),
    compileChecked("axisLabel", getAxisLabelShader()),
    compileChecked("fxaa", getFxaaShader()),
    compileChecked("ssao", getSsaoShader()),
  ]);
  startupEnd("gpu.pipelines.compile-shaders");

  startupBegin("gpu.pipelines.create");
  const blendPremul: GPUBlendState = {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };
  const blendMin: GPUBlendState = {
    color: { srcFactor: "one", dstFactor: "one", operation: "min" },
    alpha: { srcFactor: "one", dstFactor: "one", operation: "min" },
  };

  const { device } = gpu;

  device.pushErrorScope("validation");
  const nextIso = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: isoMod, entryPoint: "vsMain" },
    fragment: {
      module: isoMod,
      entryPoint: "fsMain",
      targets: [
        { format: gpu.canvasFormat, blend: blendPremul },
        { format: "rgba16float", blend: blendMin },
        { format: "rgba8unorm" },
      ],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`iso: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextBeer = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: beerMod, entryPoint: "vsMain" },
    fragment: {
      module: beerMod,
      entryPoint: "fsMain",
      targets: [
        { format: gpu.canvasFormat, blend: blendPremul },
        { format: "rgba16float" },
      ],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`beer: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextGrid = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: gridMod,
      entryPoint: "vsMain",
      buffers: [{
        arrayStride: 32,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 16, format: "float32x4" },
        ],
      }],
    },
    fragment: {
      module: gridMod,
      entryPoint: "fsMain",
      targets: [{ format: gpu.canvasFormat, blend: blendPremul }],
    },
    primitive: { topology: "line-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`grid: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextLabel = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: labelMod,
      entryPoint: "vsMain",
      buffers: [{
        arrayStride: 24,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 16, format: "float32x2" },
        ],
      }],
    },
    fragment: {
      module: labelMod,
      entryPoint: "fsMain",
      targets: [{ format: gpu.canvasFormat, blend: blendPremul }],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`axisLabel: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextSsao = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: ssaoMod, entryPoint: "vsMain" },
    fragment: {
      module: ssaoMod,
      entryPoint: "fsMain",
      targets: [{ format: gpu.canvasFormat }],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`ssao: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextFxaa = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: fxaaMod, entryPoint: "vsMain" },
    fragment: {
      module: fxaaMod,
      entryPoint: "fsMain",
      targets: [{ format: gpu.canvasFormat }],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`fxaa: ${err.message}`);
  }

  gpu.isoPipeline = nextIso;
  gpu.beerPipeline = nextBeer;
  gpu.gridPipeline = nextGrid;
  gpu.labelPipeline = nextLabel;
  gpu.ssaoPipeline = nextSsao;
  gpu.fxaaPipeline = nextFxaa;
  gpu.builtEpoch = PIPELINE_EPOCH;
  gpu.labelAtlasDirty = true;
  startupEnd("gpu.pipelines.create");

  if (Number.isFinite(gpu.gridHalf)) {
    const h = gpu.gridHalf;
    gpu.gridHalf = NaN;
    return { gridRebuildHalf: h };
  }
  return {};
}
