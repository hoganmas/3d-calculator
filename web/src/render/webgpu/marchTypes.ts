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
  /** Scene-volume repack + GPU upload time (uploadSceneVolumes) — NOT a per-frame IDCT cost. Only updates when a real upload runs (M-change/ladder-promote/rebind), so check sceneUploadAt before reading this as "current". */
  sceneUploadMs: number;
  /** performance.now() when sceneUploadMs was last updated. */
  sceneUploadAt: number;
  /** iso/march stage only. */
  marchMs: number;
  /** Beer/volume compositing stage. */
  beerMs: number;
  /** Flow particles stage. */
  flowMs: number;
  /** FXAA stage. */
  fxaaMs: number;
  /** Grid overlay stage — previously untimestamped entirely. */
  gridMs: number;
  marchFbW: number;
  marchFbH: number;
  /** Total measured GPU work this frame (begin → end of grid). Compare against gpu_present_iv in the HUD — the gap is present/vsync/compositor overhead outside these timestamps. */
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
  /** Iso coarse occupancy framebuffer size. */
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
  /** Finest iso compose divisor (1 = display). Coarse occupancy is always 16×. */
  isoFineDownscale?: number;
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
  fxaaSampler: GPUSampler;
  gridParamBuf: GPUBuffer;
  isoPipeline: GPURenderPipeline;
  isoRefinePipeline: GPURenderPipeline;
  isoUpsamplePipeline: GPURenderPipeline;
  beerPipeline: GPURenderPipeline;
  beerRefinePipeline: GPURenderPipeline;
  fxaaPipeline: GPURenderPipeline;
  gridPipeline: GPURenderPipeline;
  drawParamBuf: GPUBuffer;
  drawParamBufBeer: GPUBuffer;
  drawParamBufRefine: GPUBuffer;
  isoUpsampleParamBuf: GPUBuffer;
}

/** Offscreen march textures (iso / Beer pass). */
export interface MarchTargets {
  sceneColorTex: GPUTexture;
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
    !gpu.fxaaParamBuf || !gpu.fxaaSampler || !gpu.gridParamBuf ||
    !gpu.device || !gpu.isoPipeline || !gpu.isoRefinePipeline || !gpu.isoUpsamplePipeline ||
    !gpu.beerPipeline || !gpu.beerRefinePipeline ||
    !gpu.fxaaPipeline || !gpu.blitPipeline || !gpu.blitMidPipeline ||
    !gpu.blitSampler || !gpu.blitSamplerNearest || !gpu.gridPipeline ||
    !gpu.drawParamBuf || !gpu.drawParamBufBeer || !gpu.drawParamBufRefine || !gpu.isoUpsampleParamBuf
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
    fxaaSampler: gpu.fxaaSampler,
    gridParamBuf: gpu.gridParamBuf,
    isoPipeline: gpu.isoPipeline,
    isoRefinePipeline: gpu.isoRefinePipeline,
    isoUpsamplePipeline: gpu.isoUpsamplePipeline,
    beerPipeline: gpu.beerPipeline,
    beerRefinePipeline: gpu.beerRefinePipeline,
    fxaaPipeline: gpu.fxaaPipeline,
    gridPipeline: gpu.gridPipeline,
    drawParamBuf: gpu.drawParamBuf,
    drawParamBufBeer: gpu.drawParamBufBeer,
    drawParamBufRefine: gpu.drawParamBufRefine,
    isoUpsampleParamBuf: gpu.isoUpsampleParamBuf,
  };
}

export function acquireMarchTargets(): MarchTargets | null {
  const sceneColorTex = gpu.sceneColorTex;
  const occlIsoTex = gpu.occlIsoTex;
  const occlSurfTex = gpu.occlSurfTex;
  const depthTex = gpu.depthTex;
  const normalTex = gpu.normalTex;
  const volColorTex = gpu.volColorTex;
  if (
    !sceneColorTex || !occlIsoTex || !occlSurfTex ||
    !depthTex || !normalTex || !volColorTex
  ) {
    return null;
  }
  return {
    sceneColorTex,
    occlIsoTex,
    occlSurfTex,
    depthTex,
    normalTex,
    volColorTex,
  };
}
