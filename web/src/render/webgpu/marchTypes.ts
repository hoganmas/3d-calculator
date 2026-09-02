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
}

export interface RenderClipFrameGpuParams {
  camera: PerspectiveCamera;
  half: number;
  /** Iso / SSAO / compose framebuffer size. */
  fbW: number;
  fbH: number;
  /** Beer / volume framebuffer size (defaults to fbW/fbH). */
  volFbW?: number;
  volFbH?: number;
  scale: number;
  /** Beer / scalar-volume ray-march step count. */
  steps: number;
  /** Iso-surface (constraint) ray-march step count. */
  isoSteps?: number;
  ndcOffsetX?: number;
  ndcOffsetY?: number;
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
  isoRefinePipeline: GPURenderPipeline;
  isoUpsamplePipeline: GPURenderPipeline;
  beerPipeline: GPURenderPipeline;
  ssaoPipeline: GPURenderPipeline;
  fxaaPipeline: GPURenderPipeline;
  gridPipeline: GPURenderPipeline;
  drawParamBuf: GPUBuffer;
  drawParamBufBeer: GPUBuffer;
  isoUpsampleParamBuf: GPUBuffer;
}

/** Offscreen march textures (iso / SSAO / Beer pass). */
export interface MarchTargets {
  sceneColorTex: GPUTexture;
  sceneColorAoTex: GPUTexture;
  occlIsoTex: GPUTexture;
  occlSurfTex: GPUTexture;
  depthTex: GPUTexture;
  normalTex: GPUTexture;
  volColorTex: GPUTexture;
}

export interface MarchRaySetup {
  ro: [number, number, number];
  dirMatrix: DirMatrix;
  half: number;
  marchW: number;
  marchH: number;
  volW: number;
  volH: number;
  outW: number;
  outH: number;
}

export function acquireMarchGpuHandles(): MarchGpuHandles | null {
  if (
    !isClipBakeGpuReady() || !gpu.ctx || !gpu.volumeBuf || !gpu.colorBuf ||
    !gpu.fxaaParamBuf || !gpu.ssaoParamBuf || !gpu.fxaaSampler || !gpu.gridParamBuf ||
    !gpu.device || !gpu.isoPipeline || !gpu.isoRefinePipeline || !gpu.isoUpsamplePipeline ||
    !gpu.beerPipeline || !gpu.ssaoPipeline ||
    !gpu.fxaaPipeline || !gpu.blitPipeline || !gpu.blitSampler || !gpu.gridPipeline ||
    !gpu.drawParamBuf || !gpu.drawParamBufBeer || !gpu.isoUpsampleParamBuf
  ) {
    return null;
  }
  if (gpu.drawParamBufBeer.size < 512) {
    gpu.drawParamBufBeer = gpu.device.createBuffer({
      size: 512,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
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
    isoRefinePipeline: gpu.isoRefinePipeline,
    isoUpsamplePipeline: gpu.isoUpsamplePipeline,
    beerPipeline: gpu.beerPipeline,
    ssaoPipeline: gpu.ssaoPipeline,
    fxaaPipeline: gpu.fxaaPipeline,
    gridPipeline: gpu.gridPipeline,
    drawParamBuf: gpu.drawParamBuf,
    drawParamBufBeer: gpu.drawParamBufBeer,
    isoUpsampleParamBuf: gpu.isoUpsampleParamBuf,
  };
}

export function acquireMarchTargets(): MarchTargets | null {
  const sceneColorTex = gpu.sceneColorTex;
  const sceneColorAoTex = gpu.sceneColorAoTex;
  const occlIsoTex = gpu.occlIsoTex;
  const occlSurfTex = gpu.occlSurfTex;
  const depthTex = gpu.depthTex;
  const normalTex = gpu.normalTex;
  const volColorTex = gpu.volColorTex;
  if (
    !sceneColorTex || !sceneColorAoTex || !occlIsoTex || !occlSurfTex ||
    !depthTex || !normalTex || !volColorTex
  ) {
    return null;
  }
  return {
    sceneColorTex,
    sceneColorAoTex,
    occlIsoTex,
    occlSurfTex,
    depthTex,
    normalTex,
    volColorTex,
  };
}
