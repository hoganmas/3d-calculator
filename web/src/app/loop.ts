import {
  initClipBakeGpu,
  isClipBakeGpuReady,
  hasUploadedVolume,
  clearClipGpuFrame,
} from "../render/webgpu/march.js";
import {
  anyParamAnimating,
  tickParamAnimation,
  evalParamEquations,
  listParamNames,
  getParam,
} from "../model/params.js";
import { updateExprSilent } from "../model/expressions.js";
import { els, viewportSize } from "./dom.js";
import { state, ANIM_FIT_MIN_MS } from "./state.js";
import {
  renderer,
  labelRenderer,
  labelScene,
  scene,
  camera,
  controls,
  lavaBg,
} from "./scene.js";
import {
  clipUniforms,
  useGpuClipPath,
  syncClipCpuVolume,
  drawClipGpuFrame,
  isVolumePresented,
  presentSceneAfterGpuReady,
} from "./webglFallback.js";
import { scheduleMarchPipelines } from "../render/webgpu/march.js";
import { uploadFit, tickGpuKeyframeBlends, shouldRunAnimUploadFit } from "./pipeline.js";
import { tickKeyframePump } from "../model/keyframes.js";
import { hudText, refreshMetricsDump } from "./hud.js";
import { isSplashContentReady, markSplashFrameReady } from "./splash.js";
import { startupMark } from "./startupProfile.js";
import { syncKeyframeLoadBar } from "./keyframeLoadBar.js";
import { tickPerfAdapt } from "./perfAdapt.js";
import { shouldPresentThreeJs, threeJsPresentIntervalMs } from "./loopPacing.js";

let splashFrameReported = false;
let lastThreeJsPresentAt = 0;
let lastThreeJsFbW = 0;
let lastThreeJsFbH = 0;

function reportSplashFrameReady() {
  if (splashFrameReported) return;
  if (!isSplashContentReady()) return;
  if (!isVolumePresented()) return;
  splashFrameReported = true;
  markSplashFrameReady();
}

function tickSimulation(t0: number) {
  // Named param animation → Chebyshev refit (throttled).
  if (anyParamAnimating()) {
    const tSec = t0 / 1000;
    const animChanged = tickParamAnimation(tSec);
    const eqChanged = evalParamEquations();
    if (animChanged || eqChanged) {
      if (animChanged) {
        for (const name of listParamNames()) {
          const p = getParam(name);
          if (p?.exprId && !p.driven) {
            updateExprSilent(p.exprId, {
              latex: p.latex,
              sliderAnimating: p.animating,
              sliderPhase: p.phase,
              sliderSpeed: p.speed,
              sliderAnimMode: p.animMode,
            });
          }
        }
      }
      state.exprListApi?.syncAllParamSliders?.();
      // GPU iso keyframes: blend every rAF so the thumb + field stay continuous.
      tickGpuKeyframeBlends();
      if (t0 - state.lastAnimFitAt >= ANIM_FIT_MIN_MS) {
        state.lastAnimFitAt = t0;
        // Don't cancel a pending structural (latex) refit — anim would starve it.
        const pendingStructural =
          !!state.fitTimer && state.pendingFitOpts?.fromAnim !== true;
        if (state.uploadFitBusy || pendingStructural) {
          // Keep the timer; GPU keyframe blends above still run this frame.
        } else if (shouldRunAnimUploadFit()) {
          if (state.fitTimer) {
            clearTimeout(state.fitTimer);
            state.fitTimer = 0;
          }
          uploadFit({ fromAnim: true });
        } else {
          state.lastAnimFitAt = t0;
        }
      }
    }
  }

  if (t0 - state.loopFpsLast >= 500) {
    const winMs = t0 - state.loopFpsLast;
    state.loopFps = (state.loopFpsFrames * 1000) / winMs;
    state.loopFpsFrames = 0;
    state.loopFpsLast = t0;
    tickPerfAdapt(t0);
    if (els.hud) els.hud.textContent = hudText();
    refreshMetricsDump();
  }
}

function presentThreeJs(now: number, gpuPath: boolean) {
  const w = renderer.domElement.width;
  const h = renderer.domElement.height;
  const resized = w !== lastThreeJsFbW || h !== lastThreeJsFbH;
  const interval = threeJsPresentIntervalMs(state.deviceTier, gpuPath);
  if (!resized && !shouldPresentThreeJs(now, lastThreeJsPresentAt, interval)) return;

  lavaBg.setTime(now / 1000);
  lavaBg.syncCamera(camera);
  renderer.autoClear = true;
  renderer.render(scene, camera);
  lastThreeJsPresentAt = now;
  lastThreeJsFbW = w;
  lastThreeJsFbH = h;
}

function presentGpuClip(gpuPath: boolean) {
  if (!isClipBakeGpuReady()) return;
  if (hasUploadedVolume() && gpuPath) {
    drawClipGpuFrame();
    return;
  }
  if (!hasUploadedVolume()) {
    const { vw, vh } = viewportSize();
    clearClipGpuFrame(vw, vh);
    state.densSubmittedThisFrame = false;
  }
}

function frame(rafNow: number) {
  const t0 = performance.now();
  if (state.lastRafAt > 0) {
    const rafDt = rafNow > 0 ? rafNow - state.lastRafAt : t0 - state.lastRafAt;
    if (rafDt > 0 && rafDt < 500) {
      state.frameDtSmooth = state.frameDtSmooth * 0.85 + rafDt * 0.15;
    }
  }
  state.lastRafAt = rafNow > 0 ? rafNow : t0;
  state.loopFpsFrames++;

  tickSimulation(t0);

  controls.update();
  clipUniforms.uCameraPos.value.copy(camera.position);

  const gpuPath = useGpuClipPath();
  if (!gpuPath) {
    syncClipCpuVolume();
  }

  presentThreeJs(t0, gpuPath);
  presentGpuClip(gpuPath);

  if (!gpuPath) {
    labelRenderer.render(labelScene, camera);
  }

  reportSplashFrameReady();

  const dt = performance.now() - t0;
  state.cpuMsSmooth = state.cpuMsSmooth * 0.85 + dt * 0.15;

  // Keyframe progressive fill runs after draw so animation stays smooth.
  if (anyParamAnimating()) tickKeyframePump();
  syncKeyframeLoadBar();

  requestAnimationFrame(frame);
}

export function startRenderLoop() {
  controls.addEventListener("start", () => {
    state.clipDirty = true;
  });

  controls.addEventListener("change", () => {
    state.clipDirty = true;
  });

  controls.addEventListener("end", () => {
    state.clipDirty = true;
  });

  startupMark("boot.render-loop-started");
  requestAnimationFrame(frame);

  void initClipBakeGpu(els.viewport, "render-loop").then((ok) => {
    if (ok) void scheduleMarchPipelines(state.fitDeg);
    presentSceneAfterGpuReady("render-loop-init");
    if (!useGpuClipPath()) {
      syncClipCpuVolume();
    }
  });
}
