/**
 * WebGPU volume march — public API.
 * Implementation split across march*.ts modules; import from here only.
 */
export { MAX_DENS_LAYERS } from "./gpuState.js";
export {
  uploadSceneColors,
  uploadSceneVolumes,
  setConstraintKeyframeBlends,
  setDensKeyframeBlends,
  patchConstraintKeyframeFrame,
  hasUploadedVolume,
} from "./sceneUpload.js";

export type {
  ClipGpuTheme,
  ClipGpuProfile,
  RenderClipFrameGpuParams,
  CanvasSize,
} from "./marchTypes.js";

export {
  isClipGpuUploadReady,
  isClipBakeGpuReady,
  isClipMarchReady,
  isClipGpuInitPending,
} from "./marchReadiness.js";

export {
  getClipGpuProfile,
  resetClipGpuProfile,
} from "./marchProfile.js";

export {
  applyClipGpuTheme,
  syncClipGpuWorldGrid,
} from "./gridOverlay.js";

export {
  resizeClipGpuCanvas,
  setClipGpuCanvasVisible,
  clearClipGpuFrame,
} from "./marchCanvas.js";

export {
  ensurePipelinesForDegree,
  initClipBakeGpu,
  scheduleMarchPipelines,
  setMarchPipelinesReadyHandler,
} from "./marchInit.js";

export { renderClipFrameGpu } from "./renderFrame.js";
