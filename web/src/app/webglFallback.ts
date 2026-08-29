import * as THREE from "three";
import { clipGridVertex, clipGridFragment } from "../render/webgl/marchShaders.js";
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix } from "../render/camera.js";
import {
  initClipBakeGpu,
  isClipBakeGpuReady,
  isClipMarchReady,
  renderClipFrameGpu,
  setClipGpuCanvasVisible,
  ensurePipelinesForDegree,
  uploadSceneVolumes,
  hasUploadedVolume,
  clearClipGpuFrame,
} from "../render/webgpu/march.js";
import { flowPresenceSlice } from "../math/fitVector.js";
import { els, viewportSize } from "./dom.js";
import { state } from "./state.js";
import {
  scene,
  camera,
  renderer,
  themeColors,
} from "./scene.js";
import {
  compositionNdcOffsetX,
  compositionNdcOffsetY,
  marchFramebufferSize,
  syncClipPresentation,
} from "./presentation.js";

const volPlaceholder = new Float32Array(8);
const volumeTex = new THREE.DataTexture(volPlaceholder, 2, 4, THREE.RedFormat, THREE.FloatType);
volumeTex.minFilter = THREE.LinearFilter;
volumeTex.magFilter = THREE.LinearFilter;
volumeTex.generateMipmaps = false;
volumeTex.flipY = false;
volumeTex.colorSpace = THREE.NoColorSpace;
volumeTex.needsUpdate = true;

/** @type {THREE.DataTexture | null} */
let clipVolumeTex: THREE.DataTexture | null = null;
let clipVolumeM = 0;

export const clipUniforms = {
  uVolumeTex: { value: volumeTex },
  uGridM: { value: 2 },
  uFbW: { value: 1 },
  uFbH: { value: 1 },
  uHalf: { value: 2.5 },
  uScale: { value: 2.5 },
  uSteps: { value: 32 },
  uCameraPos: { value: new THREE.Vector3() },
  uDirM: { value: new THREE.Matrix3() },
  uAbsorbColor: { value: new THREE.Color(...themeColors.beerAbsorb) },
  uEmitColor: { value: new THREE.Color(...themeColors.beerEmit) },
};

const clipMat = new THREE.ShaderMaterial({
  vertexShader: clipGridVertex,
  fragmentShader: clipGridFragment,
  uniforms: clipUniforms,
  transparent: true,
  depthWrite: false,
  depthTest: false,
  side: THREE.DoubleSide,
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor,
  blendSrcAlpha: THREE.OneFactor,
  blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
});

export const clipQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), clipMat);
clipQuad.frustumCulled = false;
clipQuad.visible = false;
scene.add(clipQuad);

export function applyVolumeTexture(dens: Float32Array, M: number) {
  const h = M * M;
  if (!clipVolumeTex || clipVolumeM !== M) {
    if (clipVolumeTex) clipVolumeTex.dispose();
    clipVolumeTex = new THREE.DataTexture(dens, M, h, THREE.RedFormat, THREE.FloatType);
    clipVolumeTex.minFilter = THREE.LinearFilter;
    clipVolumeTex.magFilter = THREE.LinearFilter;
    clipVolumeTex.generateMipmaps = false;
    clipVolumeTex.flipY = false;
    clipVolumeTex.colorSpace = THREE.NoColorSpace;
    clipVolumeTex.needsUpdate = true;
    clipVolumeM = M;
    clipUniforms.uVolumeTex.value = clipVolumeTex;
  } else if (clipVolumeTex.image?.data) {
    clipVolumeTex.image.data.set(dens);
    clipVolumeTex.needsUpdate = true;
  }
  clipUniforms.uGridM.value = M;
}

export function syncClipFiberUniforms() {
  // CPU/WebGL path draws into the full Three.js canvas — NDC must use that
  // buffer size, not the march-downscale size (that is GPU-canvas only).
  camera.updateMatrixWorld(true);
  const { vw, vh } = viewportSize();
  const { sx, sy } = perspectiveDirScale(camera);
  const M = offsetDirMatrix(
    ndcToDirMatrix(camera, sx, sy),
    compositionNdcOffsetX(vw),
    compositionNdcOffsetY(vh),
  );
  clipUniforms.uFbW.value = Math.max(1, renderer.domElement.width);
  clipUniforms.uFbH.value = Math.max(1, renderer.domElement.height);
  clipUniforms.uDirM.value.set(M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]);
  clipUniforms.uCameraPos.value.copy(camera.position);
}

export function useGpuClipPath() {
  return isClipBakeGpuReady() && isClipMarchReady();
}

/** True when the first baked volume is visible (WebGPU march or WebGL fallback). */
export function isVolumePresented() {
  const bake = state.lastSceneBake;
  if (!bake) return false;
  const hasLayers =
    (bake.cloudLayers?.length ?? 0) > 0 ||
    (bake.isosurfaceLayers?.length ?? 0) > 0 ||
    (bake.flowLayers?.length ?? 0) > 0;
  if (!hasLayers) return true;
  if (useGpuClipPath()) return hasUploadedVolume() && state.densSubmittedThisFrame;
  return clipQuad.visible;
}

/** Lazy dens sum for the WebGL Beer texture (skipped on the GPU path). */
export function ensureDensSumForWebGl() {
  if (!state.lastSceneBake) return null;
  if (state.lastSceneBake.dens) return state.lastSceneBake.dens;
  const { cloudLayers, flowLayers, M } = state.lastSceneBake;
  if (!cloudLayers?.length && !flowLayers?.length) return null;
  const densSum = new Float32Array(M * M * M);
  for (const d of cloudLayers ?? []) {
    for (let i = 0; i < densSum.length; i++) densSum[i] += d.dens[i] || 0;
  }
  for (const f of flowLayers ?? []) {
    // WebGL path: static presence × opacity only (no IBFV dye animation).
    const presence = flowPresenceSlice(f.fx, f.fy, f.fz, M);
    for (let i = 0; i < densSum.length; i++) densSum[i] += presence[i]! * state.flowOpacity;
  }
  state.lastSceneBake.dens = densSum;
  return densSum;
}

/** Fit-time: IDCT each expression → GPU scene (manifolds + densities). */
export function bakeChebVolume() {
  if (!state.lastSceneBake) return null;
  const { cloudLayers, isosurfaceLayers, flowLayers, M, half } = state.lastSceneBake;
  if (isClipBakeGpuReady()) {
    const up = uploadSceneVolumes({ cloudLayers, isosurfaceLayers, flowLayers, M, half });
    if (up) state.bakeMsSmooth = state.bakeMsSmooth * 0.5 + up.bakeMs * 0.5;
  }
  state.lastVolumeM = M;
  // WebGL fallback only — GPU path uses per-layer dens via uploadSceneVolumes.
  if (!useGpuClipPath()) {
    const dens = ensureDensSumForWebGl();
    if (dens) applyVolumeTexture(dens, M);
  }
  return { dens: state.lastSceneBake.dens, M };
}

/** Per-frame GPU volume march (IDCT bake is fit-time only). */
export function drawClipGpuFrame() {
  state.densSubmittedThisFrame = false;
  const { vw, vh } = viewportSize();
  const { mw, mh } = marchFramebufferSize();
  if (!state.lastSceneBake || !isClipBakeGpuReady()) return false;
  if (!hasUploadedVolume()) {
    clearClipGpuFrame(vw, vh);
    state.clipDirty = false;
    return true;
  }
  if (!useGpuClipPath()) return false;

  camera.updateMatrixWorld(true);
  const t0 = performance.now();
  const ok = renderClipFrameGpu({
    camera,
    half: clipUniforms.uHalf.value,
    fbW: mw,
    fbH: mh,
    displayW: vw,
    displayH: vh,
    scale: clipUniforms.uScale.value,
    steps: clipUniforms.uSteps.value | 0,
    ndcOffsetX: compositionNdcOffsetX(vw),
    ndcOffsetY: compositionNdcOffsetY(vh),
  });
  const submitMs = performance.now() - t0;
  if (ok) {
    state.densSubmittedThisFrame = true;
    state.lastDensSubmitMs = submitMs;
    state.bakeMsSmooth = state.bakeMsSmooth * 0.85 + submitMs * 0.15;
    state.clipDirty = false;
  }
  return ok;
}

export function syncClipCpuVolume() {
  if (useGpuClipPath()) return;
  if (!state.worldCheb || !hasUploadedVolume()) {
    clipQuad.visible = false;
    return;
  }
  if (state.clipDirty || !clipVolumeTex) bakeChebVolume();
  syncClipFiberUniforms();
  setClipGpuCanvasVisible(false);
  clipQuad.visible = true;
  state.clipDirty = false;
}

export async function prepareClipGpuForDegree(deg: number) {
  try {
    const ok = await initClipBakeGpu(els.viewport);
    if (!ok) return false;
    await ensurePipelinesForDegree(deg);
    if (state.lastSceneBake) bakeChebVolume();
    syncClipPresentation();
    return true;
  } catch (e) {
    console.warn("[clip-grid] pipeline specialize failed", e);
    return false;
  }
}

export function initWebglFallback() {
  // clipQuad already added to scene at module load
}
