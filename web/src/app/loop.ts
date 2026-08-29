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
import { uploadFit, tickGpuKeyframeBlends } from "./pipeline.js";
import { tickKeyframePump } from "../model/keyframes.js";
import { hudText, refreshMetricsDump } from "./hud.js";
import { isSplashContentReady, markSplashFrameReady } from "./splash.js";
import { startupMark } from "./startupProfile.js";
import { syncKeyframeLoadBar } from "./keyframeLoadBar.js";

let splashFrameReported = false;

function reportSplashFrameReady() {
  if (splashFrameReported) return;
  if (!isSplashContentReady()) return;
  if (!isVolumePresented()) return;
  splashFrameReported = true;
  markSplashFrameReady();
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
      // GPU iso keyframes: blend every frame so the thumb + field stay continuous.
      tickGpuKeyframeBlends();
      if (t0 - state.lastAnimFitAt >= ANIM_FIT_MIN_MS) {
        state.lastAnimFitAt = t0;
        if (state.fitTimer) {
          clearTimeout(state.fitTimer);
          state.fitTimer = 0;
        }
        uploadFit({ fromAnim: true });
      }
    }
  }

  if (t0 - state.loopFpsLast >= 500) {
    const winMs = t0 - state.loopFpsLast;
    state.loopFps = (state.loopFpsFrames * 1000) / winMs;
    state.loopFpsFrames = 0;
    state.loopFpsLast = t0;
    if (els.hud) els.hud.textContent = hudText();
    refreshMetricsDump();
  }

  controls.update();
  clipUniforms.uCameraPos.value.copy(camera.position);

  if (!useGpuClipPath()) {
    syncClipCpuVolume();
  }

  lavaBg.setTime(t0 / 1000);
  lavaBg.syncCamera(camera);
  renderer.autoClear = true;
  renderer.render(scene, camera);

  if (isClipBakeGpuReady()) {
    if (hasUploadedVolume() && useGpuClipPath()) {
      drawClipGpuFrame();
    } else if (!hasUploadedVolume()) {
      const { vw, vh } = viewportSize();
      clearClipGpuFrame(vw, vh);
      state.densSubmittedThisFrame = false;
    }
  }

  if (!useGpuClipPath()) {
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
