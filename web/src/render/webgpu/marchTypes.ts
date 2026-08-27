import type { PerspectiveCamera } from "three";
import type { DirMatrix } from "../camera.js";
import { gpu } from "./gpuState.js";
import { isClipBakeGpuReady } from "./marchReadiness.js";

export interface ClipGpuTheme {
  gridMajor?: number;
  gridMinor?: number;
  boxEdgeRgb?: number[];
  axisXRgb?: number[];
  axisYRgb?: number[];
  axisZRgb?: number[];
  labelStroke?: string;
}

export interface ClipGpuProfile {
  idctMs: number;
  marchMs: number;
  marchFbW: number;
  marchFbH: number;
  presentWallMs: number;
  presentIntervalMs: number;
  lastPresentAt: number;
  method: string;
  gridM: number;
  timestamps: boolean;
  isoInterp: "hermite" | "trilinear";
}

export interface RenderClipFrameGpuParams {
  camera: PerspectiveCamera;
  half: number;
  fbW: number;
  fbH: number;
  scale: number;
  steps: number;
  ndcOffsetX?: number;
  displayW?: number;
  displayH?: number;
}

export interface CanvasSize {
  w: number;
  h: number;
}

/** Non-null GPU handles required to record a march frame. */
export interface MarchGpuHandles {
  device: GPUDevice;
  ctx: GPUCanvasContext;
  volumeBuf: GPUBuffer;
  colorBuf: GPUBuffer;
  fxaaParamBuf: GPUBuffer;
  ssaoParamBuf: GPUBuffer;
  fxaaSampler: GPUSampler;
  gridParamBuf: GPUBuffer;
  isoPipeline: GPURenderPipeline;
  beerPipeline: GPURenderPipeline;
  ssaoPipeline: GPURenderPipeline;
  fxaaPipeline: GPURenderPipeline;
  gridPipeline: GPURenderPipeline;
  drawParamBuf: GPUBuffer;
  drawParamBufBeer: GPUBuffer;
}

/** Offscreen march textures (iso / SSAO / Beer pass). */
export interface MarchTargets {
  sceneColorTex: GPUTexture;
  sceneColorAoTex: GPUTexture;
  occlTex: GPUTexture;
  depthTex: GPUTexture;
  normalTex: GPUTexture;
}

export interface MarchRaySetup {
  ro: [number, number, number];
  dirMatrix: DirMatrix;
  half: number;
  marchW: number;
  marchH: number;
  outW: number;
  outH: number;
}

export function acquireMarchGpuHandles(): MarchGpuHandles | null {
  if (
    !isClipBakeGpuReady() || !gpu.ctx || !gpu.volumeBuf || !gpu.colorBuf ||
    !gpu.fxaaParamBuf || !gpu.ssaoParamBuf || !gpu.fxaaSampler || !gpu.gridParamBuf ||
    !gpu.device || !gpu.isoPipeline || !gpu.beerPipeline || !gpu.ssaoPipeline ||
    !gpu.fxaaPipeline || !gpu.gridPipeline || !gpu.drawParamBuf || !gpu.drawParamBufBeer
  ) {
    return null;
  }
  return {
    device: gpu.device,
    ctx: gpu.ctx,
    volumeBuf: gpu.volumeBuf,
    colorBuf: gpu.colorBuf,
    fxaaParamBuf: gpu.fxaaParamBuf,
    ssaoParamBuf: gpu.ssaoParamBuf,
    fxaaSampler: gpu.fxaaSampler,
    gridParamBuf: gpu.gridParamBuf,
    isoPipeline: gpu.isoPipeline,
    beerPipeline: gpu.beerPipeline,
    ssaoPipeline: gpu.ssaoPipeline,
    fxaaPipeline: gpu.fxaaPipeline,
    gridPipeline: gpu.gridPipeline,
    drawParamBuf: gpu.drawParamBuf,
    drawParamBufBeer: gpu.drawParamBufBeer,
  };
}

export function acquireMarchTargets(): MarchTargets | null {
  const sceneColorTex = gpu.sceneColorTex;
  const sceneColorAoTex = gpu.sceneColorAoTex;
  const occlTex = gpu.occlTex;
  const depthTex = gpu.depthTex;
  const normalTex = gpu.normalTex;
  if (!sceneColorTex || !sceneColorAoTex || !occlTex || !depthTex || !normalTex) return null;
  return { sceneColorTex, sceneColorAoTex, occlTex, depthTex, normalTex };
}
