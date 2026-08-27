import { hexToRgb01, EXPR_GRADIENTS } from "../../model/expressions.js";

export const MAX_DENS_LAYERS = 8;

export const DEFAULT_DENS_RGB = hexToRgb01(EXPR_GRADIENTS[0].color);
export const DEFAULT_DENS_RGB2 = hexToRgb01(EXPR_GRADIENTS[0].color2);
export const DEFAULT_ISO_RGB = hexToRgb01(EXPR_GRADIENTS[1].color);
export const DEFAULT_ISO_RGB2 = hexToRgb01(EXPR_GRADIENTS[1].color2);

/** Shared mutable WebGPU march state (single device lifetime). */
export const gpu = {
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

export function resetPipelinesOnDeviceLost() {
  gpu.isoPipeline = gpu.beerPipeline = gpu.fxaaPipeline = gpu.ssaoPipeline = null;
  gpu.gridPipeline = gpu.labelPipeline = null;
  gpu.labelAtlasTex = gpu.labelAtlasSamp = null;
  gpu.labelVertexBuf = null;
  gpu.labelAtlasDirty = true;
}
