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

export interface GpuDensLayer {
  id: string | null;
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
  isoRefinePipeline: GPURenderPipeline | null;
  isoUpsamplePipeline: GPURenderPipeline | null;
  beerPipeline: GPURenderPipeline | null;
  beerRefinePipeline: GPURenderPipeline | null;
  blitPipeline: GPURenderPipeline | null;
  /** TEMP DIAGNOSTIC: blit.wgsl's fsMainSwap entry point (mid-cascade corner-swap test). */
  blitMidPipeline: GPURenderPipeline | null;
  fxaaPipeline: GPURenderPipeline | null;
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
  drawParamBufRefine: GPUBuffer | null;
  /** One uniform buffer per iso layer so a pass can draw all constraints without flushing. */
  isoDrawParamBufs: GPUBuffer[];
  fxaaParamBuf: GPUBuffer | null;
  isoUpsampleParamBuf: GPUBuffer | null;
  volumeBuf: GPUBuffer | null;
  volumeCapacity: number;
  colorBuf: GPUBuffer | null;
  /** Iso manifold depths (iso write, Beer read). */
  occlIsoTex: GPUTexture | null;
  occlIsoW: number;
  occlIsoH: number;
  /** Combined iso+density depths for grid/axis occlusion (Beer write). */
  occlSurfTex: GPUTexture | null;
  occlSurfW: number;
  occlSurfH: number;
  /** Coarse iso G-buffer (occupancy-guided refine source). */
  isoCoarseColorTex: GPUTexture | null;
  isoCoarseOcclTex: GPUTexture | null;
  isoCoarseNormalTex: GPUTexture | null;
  isoCoarseDepthTex: GPUTexture | null;
  isoCoarseW: number;
  isoCoarseH: number;
  /** Mid iso G-buffer (4× occupancy between coarse and compose). */
  isoMidColorTex: GPUTexture | null;
  isoMidOcclTex: GPUTexture | null;
  isoMidNormalTex: GPUTexture | null;
  isoMidDepthTex: GPUTexture | null;
  isoMidW: number;
  isoMidH: number;
  depthTex: GPUTexture | null;
  depthW: number;
  depthH: number;
  normalTex: GPUTexture | null;
  normalW: number;
  normalH: number;
  sceneColorTex: GPUTexture | null;
  sceneColorW: number;
  sceneColorH: number;
  /** Beer / volume march color (may differ in resolution from iso scene). */
  volColorTex: GPUTexture | null;
  volColorW: number;
  volColorH: number;
  /** Beer remarch at 4× mid iso size (coarse-mixed tiles). */
  volMidColorTex: GPUTexture | null;
  volMidW: number;
  volMidH: number;
  fxaaSampler: GPUSampler | null;
  blitSampler: GPUSampler | null;
  /** Nearest-filter blit sampler — for compositing textures with a hard-cleared
   *  (transparent) exterior, where a linear filter would bleed real color into
   *  the clear value right at the shaded/unshaded boundary. */
  blitSamplerNearest: GPUSampler | null;
  sceneConstraints: GpuSceneConstraint[];
  densPacked: boolean;
  densGradStops: RgbTriplet[][];
  densLayerCount: number;
  densBase: number;
  densLayers: GpuDensLayer[];
  sceneM: number;
  sceneEpoch: number;
  /** Last `sceneEpoch` written to `volumeBuf` (skip redundant per-frame uploads). */
  volumeUploadEpoch: number;
  scenePacked: Float32Array | null;
  initFailed: boolean;
  initPromise: Promise<boolean> | null;
  timestampsSupported: boolean;
  stampQuerySet: GPUQuerySet | null;
  stampResolveBuf: GPUBuffer | null;
  stampReadBuf: GPUBuffer | null;
  stampReadPending: boolean;
  /** At most one in-flight `onSubmittedWorkDone` sample (avoid per-frame GPU idle waits). */
  presentWorkSamplePending: boolean;
  profileBakeMs: number;
  /** performance.now() when profileBakeMs was last updated — tells a live reading from a frozen one. */
  profileBakeAt: number;
  /** iso/march stage only (begin → end of the iso-refine-ladder / iso-constraints / clear branch). */
  profileMarchMs: number;
  /** Beer/volume compositing stage (end of march → end of beer). */
  profileBeerMs: number;
  /** Flow particles stage (end of beer → end of flow). */
  profileFlowMs: number;
  /** FXAA stage (end of flow → end of fxaa). */
  profileFxaaMs: number;
  /** Grid overlay stage (end of fxaa → end of grid) — previously untimestamped entirely. */
  profileGridMs: number;
  profileMarchFbW: number;
  profileMarchFbH: number;
  /** Total measured GPU work this frame (begin → end of grid) — compare against gpu_present_iv; the gap is present/vsync/compositor overhead outside these timestamps. */
  profilePresentWallMs: number;
  profilePresentIntervalMs: number;
  lastPresentAt: number;
  profileMethod: string;
  profileGridM: number;
  builtEpoch: number;
  flowHalf: number;
  /** First Beer layer index that is a flow layer. */
  flowLayerStart: number;
  /** Float offset in volume for first flow layer fx slice. */
  flowVelBase: number;
  flowLayerCount: number;
  flowParticlesPipeline: GPURenderPipeline | null;
  flowParticlesParamBuf: GPUBuffer | null;
  flowParticleBuf: GPUBuffer | null;
  flowParticleLayerBuf: GPUBuffer | null;
  flowParticleSortBuf: GPUBuffer | null;
  flowTrailBuf: GPUBuffer | null;
  /** Total instanced particles in GPU buffers. */
  flowParticleCount: number;
  /** Particles allocated per flow expression (count ≈ perLayer × flowLayerCount). */
  flowParticlesPerLayer: number;
}

export const gpu: GpuState = {
  device: null,
  ctx: null,
  canvas: null,
  canvasFormat: "bgra8unorm",
  isoPipeline: null,
  isoRefinePipeline: null,
  isoUpsamplePipeline: null,
  beerPipeline: null,
  beerRefinePipeline: null,
  blitPipeline: null,
  blitMidPipeline: null,
  fxaaPipeline: null,
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
  drawParamBufRefine: null,
  isoDrawParamBufs: [],
  fxaaParamBuf: null,
  isoUpsampleParamBuf: null,
  volumeBuf: null,
  volumeCapacity: 0,
  colorBuf: null,
  occlIsoTex: null,
  occlIsoW: 0,
  occlIsoH: 0,
  occlSurfTex: null,
  occlSurfW: 0,
  occlSurfH: 0,
  isoCoarseColorTex: null,
  isoCoarseOcclTex: null,
  isoCoarseNormalTex: null,
  isoCoarseDepthTex: null,
  isoCoarseW: 0,
  isoCoarseH: 0,
  isoMidColorTex: null,
  isoMidOcclTex: null,
  isoMidNormalTex: null,
  isoMidDepthTex: null,
  isoMidW: 0,
  isoMidH: 0,
  depthTex: null,
  depthW: 0,
  depthH: 0,
  normalTex: null,
  normalW: 0,
  normalH: 0,
  sceneColorTex: null,
  sceneColorW: 0,
  sceneColorH: 0,
  volColorTex: null,
  volColorW: 0,
  volColorH: 0,
  volMidColorTex: null,
  volMidW: 0,
  volMidH: 0,
  fxaaSampler: null,
  blitSampler: null,
  blitSamplerNearest: null,
  sceneConstraints: [],
  densPacked: false,
  densGradStops: [],
  densLayerCount: 0,
  densBase: 0,
  densLayers: [],
  sceneM: 0,
  sceneEpoch: 0,
  volumeUploadEpoch: -1,
  scenePacked: null,
  initFailed: false,
  initPromise: null,
  timestampsSupported: false,
  stampQuerySet: null,
  stampResolveBuf: null,
  stampReadBuf: null,
  stampReadPending: false,
  presentWorkSamplePending: false,
  profileBakeMs: 0,
  profileBakeAt: 0,
  profileMarchMs: 0,
  profileBeerMs: 0,
  profileFlowMs: 0,
  profileFxaaMs: 0,
  profileGridMs: 0,
  profileMarchFbW: 0,
  profileMarchFbH: 0,
  profilePresentWallMs: 0,
  profilePresentIntervalMs: 0,
  lastPresentAt: 0,
  profileMethod: "",
  profileGridM: 0,
  builtEpoch: -1,
  flowHalf: 2.5,
  flowLayerStart: -1,
  flowVelBase: 0,
  flowLayerCount: 0,
  flowParticlesPipeline: null,
  flowParticlesParamBuf: null,
  flowParticleBuf: null,
  flowParticleLayerBuf: null,
  flowParticleSortBuf: null,
  flowTrailBuf: null,
  flowParticleCount: 0,
  flowParticlesPerLayer: 0,
};

export const PIPELINE_EPOCH = 92;
export const labelVertScratch = new Float32Array(18 * 6);

export function resetPipelinesOnDeviceLost(): void {
  gpu.isoPipeline = gpu.isoRefinePipeline = gpu.isoUpsamplePipeline = gpu.beerPipeline = gpu.fxaaPipeline = null;
  gpu.beerRefinePipeline = null;
  gpu.blitPipeline = null;
  gpu.blitMidPipeline = null;
  gpu.gridPipeline = gpu.labelPipeline = null;
  gpu.flowParticlesPipeline = null;
  gpu.flowParticlesParamBuf = null;
  gpu.flowParticleBuf = gpu.flowParticleLayerBuf = gpu.flowParticleSortBuf = gpu.flowTrailBuf = null;
  gpu.flowParticleCount = 0;
  gpu.flowParticlesPerLayer = 0;
  gpu.labelAtlasTex = gpu.labelAtlasSamp = null;
  gpu.labelVertexBuf = null;
  gpu.labelAtlasDirty = true;
  gpu.volumeUploadEpoch = -1;
  gpu.presentWorkSamplePending = false;
  gpu.isoDrawParamBufs = [];
}
