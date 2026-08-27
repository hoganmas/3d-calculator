import { hexToRgb01, EXPR_GRADIENTS } from "../../model/expressions.js";

export const MAX_DENS_LAYERS = 8;

export const DEFAULT_DENS_RGB = hexToRgb01(EXPR_GRADIENTS[0].color);
export const DEFAULT_DENS_RGB2 = hexToRgb01(EXPR_GRADIENTS[0].color2);
export const DEFAULT_ISO_RGB = hexToRgb01(EXPR_GRADIENTS[1].color);
export const DEFAULT_ISO_RGB2 = hexToRgb01(EXPR_GRADIENTS[1].color2);

/** RGB in 0..1 (length 3). */
export type RgbTriplet = number[];

export interface GpuSceneConstraint {
  id: string | null;
  color: RgbTriplet;
  color2: RgbTriplet;
  colors: RgbTriplet[];
  isoLevel: number;
  base: number;
  frameStride: number;
  K: number;
  i0: number;
  i1: number;
  t: number;
}

/** Shared mutable WebGPU march state (single device lifetime). */
export interface GpuState {
  device: GPUDevice | null;
  ctx: GPUCanvasContext | null;
  canvas: (HTMLCanvasElement & { _clipConfigured?: boolean }) | null;
  canvasFormat: GPUTextureFormat;
  isoPipeline: GPURenderPipeline | null;
  beerPipeline: GPURenderPipeline | null;
  fxaaPipeline: GPURenderPipeline | null;
  ssaoPipeline: GPURenderPipeline | null;
  gridPipeline: GPURenderPipeline | null;
  gridParamBuf: GPUBuffer | null;
  gridVertexBuf: GPUBuffer | null;
  gridVertexCapacity: number;
  gridVertexCount: number;
  gridHalf: number;
  labelPipeline: GPURenderPipeline | null;
  labelVertexBuf: GPUBuffer | null;
  labelAtlasTex: GPUTexture | null;
  labelAtlasSamp: GPUSampler | null;
  labelAtlasDirty: boolean;
  drawParamBuf: GPUBuffer | null;
  drawParamBufBeer: GPUBuffer | null;
  fxaaParamBuf: GPUBuffer | null;
  ssaoParamBuf: GPUBuffer | null;
  volumeBuf: GPUBuffer | null;
  volumeCapacity: number;
  colorBuf: GPUBuffer | null;
  occlTex: GPUTexture | null;
  occlW: number;
  occlH: number;
  depthTex: GPUTexture | null;
  depthW: number;
  depthH: number;
  normalTex: GPUTexture | null;
  normalW: number;
  normalH: number;
  sceneColorTex: GPUTexture | null;
  sceneColorW: number;
  sceneColorH: number;
  sceneColorAoTex: GPUTexture | null;
  sceneColorAoW: number;
  sceneColorAoH: number;
  fxaaSampler: GPUSampler | null;
  sceneConstraints: GpuSceneConstraint[];
  densPacked: boolean;
  densGradStops: RgbTriplet[][];
  densLayerCount: number;
  densBase: number;
  sceneM: number;
  sceneEpoch: number;
  scenePacked: Float32Array | null;
  initFailed: boolean;
  initPromise: Promise<boolean> | null;
  timestampsSupported: boolean;
  stampQuerySet: GPUQuerySet | null;
  stampResolveBuf: GPUBuffer | null;
  stampReadBuf: GPUBuffer | null;
  stampReadPending: boolean;
  profileBakeMs: number;
  profileMarchMs: number;
  profileMarchFbW: number;
  profileMarchFbH: number;
  profilePresentWallMs: number;
  profilePresentIntervalMs: number;
  lastPresentAt: number;
  profileMethod: string;
  profileGridM: number;
  builtEpoch: number;
  isoInterpHermite: boolean;
}

export const gpu: GpuState = {
  device: null,
  ctx: null,
  canvas: null,
  canvasFormat: "bgra8unorm",
  isoPipeline: null,
  beerPipeline: null,
  fxaaPipeline: null,
  ssaoPipeline: null,
  gridPipeline: null,
  gridParamBuf: null,
  gridVertexBuf: null,
  gridVertexCapacity: 0,
  gridVertexCount: 0,
  gridHalf: NaN,
  labelPipeline: null,
  labelVertexBuf: null,
  labelAtlasTex: null,
  labelAtlasSamp: null,
  labelAtlasDirty: true,
  drawParamBuf: null,
  drawParamBufBeer: null,
  fxaaParamBuf: null,
  ssaoParamBuf: null,
  volumeBuf: null,
  volumeCapacity: 0,
  colorBuf: null,
  occlTex: null,
  occlW: 0,
  occlH: 0,
  depthTex: null,
  depthW: 0,
  depthH: 0,
  normalTex: null,
  normalW: 0,
  normalH: 0,
  sceneColorTex: null,
  sceneColorW: 0,
  sceneColorH: 0,
  sceneColorAoTex: null,
  sceneColorAoW: 0,
  sceneColorAoH: 0,
  fxaaSampler: null,
  sceneConstraints: [],
  densPacked: false,
  densGradStops: [],
  densLayerCount: 0,
  densBase: 0,
  sceneM: 0,
  sceneEpoch: 0,
  scenePacked: null,
  initFailed: false,
  initPromise: null,
  timestampsSupported: false,
  stampQuerySet: null,
  stampResolveBuf: null,
  stampReadBuf: null,
  stampReadPending: false,
  profileBakeMs: 0,
  profileMarchMs: 0,
  profileMarchFbW: 0,
  profileMarchFbH: 0,
  profilePresentWallMs: 0,
  profilePresentIntervalMs: 0,
  lastPresentAt: 0,
  profileMethod: "",
  profileGridM: 0,
  builtEpoch: -1,
  isoInterpHermite: true,
};

export const PIPELINE_EPOCH = 25;
export const labelVertScratch = new Float32Array(18 * 6);

export function resetPipelinesOnDeviceLost(): void {
  gpu.isoPipeline = gpu.beerPipeline = gpu.fxaaPipeline = gpu.ssaoPipeline = null;
  gpu.gridPipeline = gpu.labelPipeline = null;
  gpu.labelAtlasTex = gpu.labelAtlasSamp = null;
  gpu.labelVertexBuf = null;
  gpu.labelAtlasDirty = true;
}
