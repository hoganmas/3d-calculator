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
import { syncClipPresentation } from "./presentation.js";
import {
  renderer,
  labelRenderer,
  labelScene,
  scene,
  lavaScene,
  camera,
  controls,
  lavaBg,
  DEFAULT_FOV,
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
import {
  tickKeyframePump,
  KEYFRAME_PUMP_FRAME_BUDGET_MS,
  hasActiveKeyframeCaches,
  allKeyframesComplete,
} from "../model/keyframes.js";
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
  const animating = anyParamAnimating();
  let animChanged = false;
  let eqChanged = false;
  if (animating) {
    const tSec = t0 / 1000;
    animChanged = tickParamAnimation(tSec);
    eqChanged = evalParamEquations();
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
      state.exprListApi?.syncAllParamSliders?.();
    }
  }

  // GPU iso keyframes: blend + pick up newly-ready (promoted) frames every
  // rAF whenever there's any active keyframe cache — not just while playing.
  // Background generation (the pump below) keeps refining ladder rungs
  // regardless of play state; gating this on `animating` too meant a paused
  // isosurface never got shown the higher-degree frame once it finished
  // baking, so it stayed stuck at whatever coarse rung was on-screen at the
  // moment of pausing.
  if (hasActiveKeyframeCaches()) {
    tickGpuKeyframeBlends();
  }

  if (animating && (animChanged || eqChanged)) {
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

  // Lava's blob pattern is tuned for DEFAULT_FOV and has no notion of FOV
  // itself — render it with that fixed FOV regardless of the isometric
  // toggle's tiny camera.fov, or isometric mode zooms into a sliver of the
  // pattern instead of showing it naturally. Restored before returning so
  // the WebGPU raymarch pass right after this still sees the real FOV.
  const realFov = camera.fov;
  const isometric = Math.abs(realFov - DEFAULT_FOV) > 1e-6;
  if (isometric) {
    camera.fov = DEFAULT_FOV;
    camera.updateProjectionMatrix();
  }
  renderer.autoClear = true;
  renderer.render(lavaScene, camera);

  if (isometric) {
    camera.fov = realFov;
    camera.updateProjectionMatrix();
  }
  renderer.autoClear = false;
  renderer.render(scene, camera);
  renderer.autoClear = true;

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
  // Re-derive the WebGL fallback's grid/label/box visibility from this exact
  // per-frame value every frame, not just on scene-bake events — otherwise
  // the cached flags can lag the live check `drawClipGpuFrame` uses below,
  // and both paths draw the axes/grid at once (worse on mobile, where the
  // pipeline-ready → resync events land less predictably).
  syncClipPresentation(gpuPath);
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
  // Chain several maxWorks=1 units per frame (time-budgeted, not a fixed
  // count) instead of exactly one — units are cheap relative to the frame's
  // real cadence, so this cuts total generation wall time without touching
  // render pacing: it still yields well within the frame once out of budget
  // or once there's no more pending work.
  // Runs whenever there's active, incomplete keyframe work — not gated on
  // playback — so pausing mid-generation doesn't freeze it: without this,
  // an isosurface paused before its ladder finished stayed stuck at whatever
  // coarse rung was on-screen, since nothing ever pumped it further.
  if (hasActiveKeyframeCaches() && !allKeyframesComplete()) {
    const tPump0 = performance.now();
    while (performance.now() - tPump0 < KEYFRAME_PUMP_FRAME_BUDGET_MS) {
      if (tickKeyframePump(1) === 0) break;
    }
  }
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
