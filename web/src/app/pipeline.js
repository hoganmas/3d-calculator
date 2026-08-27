import { MAX_DEG } from "../math/limits.js";
import { fitChebyshev3D } from "../math/fit.js";
import { idctCheb3D, idctChebGrad3D } from "../math/idct.js";
import {
  beginKeyframePass,
  clearKeyframeCaches,
  getKeyframeMetrics,
  logKeyframeBake,
  setKeyframeProgressHandler,
  keyframeAnimParam,
  noteKeyframeLayer,
  ensureLayerKeyframes,
  sampleLayerKeyframes,
  peekKeyframeBlend,
} from "../model/keyframes.js";
import {
  getParamValues,
  collectAnimDirtyParams,
} from "../model/params.js";
import {
  isClipBakeGpuReady,
  uploadSceneVolumes,
  uploadSceneColors,
  setConstraintKeyframeBlends,
  patchConstraintKeyframeFrame,
  resetClipGpuProfile,
  setIsoInterpHermite,
} from "../render/webgpu/march.js";
import { compileExpr, classifyExpr } from "../math/fit.js";
import { listExpressions, resolveExprRole } from "../model/expressions.js";
import { els, viewportSize } from "./dom.js";
import { state, FIT_DEBOUNCE_MS } from "./state.js";
import { setBoxSize } from "./scene.js";
import { compileAllExprs, layerRgbFromItem } from "./compile.js";
import {
  clipUniforms,
  clipQuad,
  bakeChebVolume,
  useGpuClipPath,
  syncClipCpuVolume,
  prepareClipGpuForDegree,
} from "./webglFallback.js";
import { resize, syncClipPresentation } from "./presentation.js";
import { clearClipGpuFrame } from "../render/webgpu/march.js";
import {
  setErr,
  setExprCompileOk,
  syncExprCompileState,
  refreshMetricsDump,
} from "./hud.js";

export function tickGpuKeyframeBlends() {
  const cons = state.lastSceneBake?.constraints;
  if (!cons?.length || !isClipBakeGpuReady()) return false;
  /** @type {{ id: string, i0: number, i1: number, t: number }[]} */
  const blends = [];
  for (const c of cons) {
    if (!c?.id || !Array.isArray(c.keyframes) || !c.keyframes.length) continue;
    const b = peekKeyframeBlend(c.id);
    if (!b) continue;
    c.blend = { i0: b.i0, i1: b.i1, t: b.t };
    blends.push(b);
  }
  if (!blends.length) return false;
  setConstraintKeyframeBlends(blends);
  state.clipDirty = true;
  return true;
}

export function uploadFit(opts = {}) {
  const fromAnim = !!opts.fromAnim;
  setErr("");
  try {
    const boxSize = Number(els.boxSize.value);
    const deg = Number(els.deg.value);
    const densScale = Number(els.scale.value);
    const steps = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
    els.steps.value = String(steps);
    if (!(boxSize > 0)) throw new Error("box size must be > 0");
    if (deg < 1 || deg > MAX_DEG) throw new Error(`poly deg must be 1…${MAX_DEG}`);
    const half = 0.5 * boxSize;

    const tUpload = performance.now();
    const { layers } = compileAllExprs({ rebuildUi: false });
    setExprCompileOk(true);

    // No visible / non-empty expressions → clear volume, draw nothing.
    if (!layers.length) {
      clearKeyframeCaches();
      state.lastSceneBake = { densLayers: [], constraints: [], M: Math.max(2, deg + 1), dens: null };
      state.lastFitTiming = null;
      state.lastNCoeff = 0;
      state.lastFitRel = NaN;
      state.worldCheb = null;
      state.fitDeg = deg;
      if (isClipBakeGpuReady()) {
        uploadSceneVolumes({ densLayers: [], constraints: [], M: state.lastSceneBake.M });
      }
      clipUniforms.uScale.value = densScale;
      clipUniforms.uSteps.value = steps;
      setBoxSize(boxSize);
      clipQuad.visible = false;
      state.clipDirty = true;
      if (!fromAnim) resize();
      syncClipPresentation();
      if (isClipBakeGpuReady()) {
        const { vw, vh } = viewportSize();
        clearClipGpuFrame(vw, vh);
      }
      return;
    }

    const densLayers = [];
    const constraints = [];
    let cheb = null;
    let fitRel = NaN;
    let timingAcc = { sampleMs: 0, chebMs: 0, monoMs: 0, l2Ms: 0, totalMs: 0 };
    let M = deg + 1;
    let fittedCount = 0;
    let keyframedCount = 0;
    let keyframeBaked = false;
    let densKeyframedCpu = false;

    // Anim ticks: only refit layers that depend on dirty params; reuse the rest.
    // Dirty layers with exactly one animating slider: GPU keyframe blend (iso) / CPU lerp (dens).
    const dirty = fromAnim ? collectAnimDirtyParams() : null;
    if (fromAnim) beginKeyframePass();
    else clearKeyframeCaches();

    /** @type {Map<string, any>} */
    const prevById = new Map();
    const canReuseCache =
      fromAnim &&
      dirty &&
      state.lastSceneBake &&
      state.lastSceneBake.deg === deg &&
      Math.abs((state.lastSceneBake.half ?? NaN) - half) < 1e-12;
    if (canReuseCache) {
      for (const d of state.lastSceneBake.densLayers) {
        if (d.id) prevById.set(d.id, { kind: "density", ...d });
      }
      for (const c of state.lastSceneBake.constraints) {
        if (c.id) prevById.set(c.id, { kind: "constraint", ...c });
      }
    }

    const baseParams = getParamValues();

    for (const L of layers) {
      const { color, color2, colors } = layerRgbFromItem(L.item);
      const depends =
        !dirty ||
        L.compiled.freeParams.some((p) => dirty.has(p));
      const prev = canReuseCache && !depends ? prevById.get(L.item.id) : null;
      const prevHasKf =
        prev && Array.isArray(prev.keyframes) && prev.keyframes.length > 0;
      const reuseDens =
        prev &&
        prev.kind === (L.role === "constraint" ? "constraint" : "density") &&
        (prev.dens instanceof Float32Array || prevHasKf);

      if (reuseDens) {
        if (prevHasKf) {
          M = Math.round(Math.cbrt(prev.keyframes[0].dens.length)) || M;
        } else {
          M = Math.round(Math.cbrt(prev.dens.length)) || M;
        }
        if (L.role === "constraint") {
          if (prevHasKf) {
            constraints.push({
              id: L.item.id,
              keyframes: prev.keyframes,
              blend: prev.blend || { i0: 0, i1: 0, t: 0 },
              color,
              color2,
              colors,
              isoLevel: L.compiled.isoLevel ?? prev.isoLevel ?? 0,
              cheb: prev.cheb,
              fitRel: prev.fitRel,
            });
          } else {
            constraints.push({
              id: L.item.id,
              dens: prev.dens,
              gx: prev.gx,
              gy: prev.gy,
              gz: prev.gz,
              color,
              color2,
              colors,
              isoLevel: L.compiled.isoLevel ?? prev.isoLevel ?? 0,
            });
          }
        } else {
          densLayers.push({ id: L.item.id, dens: prev.dens, color, color2 });
        }
        if (!cheb && prev.cheb) {
          cheb = prev.cheb;
          fitRel = prev.fitRel ?? fitRel;
        }
        continue;
      }

      // Keyframe path: one dirty animated slider → GPU blend (iso) / CPU lerp (dens).
      const kfParam =
        fromAnim && depends
          ? keyframeAnimParam(L.compiled.freeParams, dirty)
          : null;
      if (kfParam) {
        noteKeyframeLayer();
        keyframedCount++;
        if (L.role === "constraint") {
          const sample = ensureLayerKeyframes({
            layerId: L.item.id,
            latex: L.item.latex,
            role: "constraint",
            isoLevel: L.compiled.isoLevel ?? 0,
            paramName: kfParam,
            compiled: L.compiled,
            baseParams,
            half,
            deg,
          });
          if (sample.baked) keyframeBaked = true;
          M = sample.M || M;
          constraints.push({
            id: L.item.id,
            keyframes: sample.frames,
            blend: sample.blend,
            color,
            color2,
            colors,
            isoLevel: L.compiled.isoLevel ?? 0,
            cheb: sample.cheb,
            fitRel: sample.fitRel,
          });
          if (!cheb && sample.cheb) {
            cheb = sample.cheb;
            fitRel = sample.fitRel ?? fitRel;
          }
        } else {
          const sample = sampleLayerKeyframes({
            layerId: L.item.id,
            latex: L.item.latex,
            role: "density",
            isoLevel: L.compiled.isoLevel ?? 0,
            paramName: kfParam,
            compiled: L.compiled,
            baseParams,
            half,
            deg,
          });
          if (sample.baked) keyframeBaked = true;
          M = sample.M || M;
          densLayers.push({
            id: L.item.id,
            dens: sample.dens.slice(),
            color,
            color2,
            colors,
            cheb: sample.cheb,
            fitRel: sample.fitRel,
          });
          densKeyframedCpu = true;
          if (!cheb && sample.cheb) {
            cheb = sample.cheb;
            fitRel = sample.fitRel ?? fitRel;
          }
        }
        continue;
      }

      const skipHeavy = fromAnim || layers.length > 1;
      const fit = fitChebyshev3D(L.fn, half, deg, {
        skipL2: skipHeavy,
        skipMono: skipHeavy,
      });
      fittedCount++;
      const idct = idctCheb3D(fit.cheb, fit.deg, fit.deg + 1);
      M = idct.M;
      if (L.role === "constraint") {
        const grad = idctChebGrad3D(fit.cheb, fit.deg, fit.deg + 1);
        constraints.push({
          id: L.item.id,
          dens: idct.dens,
          gx: grad.gx,
          gy: grad.gy,
          gz: grad.gz,
          color,
          color2,
          colors,
          isoLevel: L.compiled.isoLevel ?? 0,
          cheb: fit.cheb,
          fitRel: fit.fitRelL2,
        });
      } else {
        densLayers.push({
          id: L.item.id,
          dens: idct.dens,
          color,
          color2,
          colors,
          cheb: fit.cheb,
          fitRel: fit.fitRelL2,
        });
      }
      if (!cheb) {
        cheb = fit.cheb;
        fitRel = fit.fitRelL2;
      }
      for (const k of Object.keys(timingAcc)) {
        timingAcc[k] += fit.timing?.[k] || 0;
      }
    }

    // WebGL preview texture: sum of density layers (skipped when WebGPU is active).
    let densSum = null;
    if (densLayers.length && !useGpuClipPath()) {
      densSum = new Float32Array(M * M * M);
      for (const d of densLayers) {
        for (let i = 0; i < densSum.length; i++) densSum[i] += d.dens[i] || 0;
      }
    }

    state.lastSceneBake = {
      densLayers,
      constraints,
      M,
      dens: densSum,
      deg,
      half,
      fittedCount,
      keyframedCount,
    };
    const uploadMs = performance.now() - tUpload;
    const kf = getKeyframeMetrics();
    state.lastFitTiming = {
      ...timingAcc,
      uploadMs,
      fittedCount,
      keyframedCount,
      kfBakeMs: kf.bakeMs,
      kfLerpMs: kf.lerpMs,
      kfK: kf.K,
      kfSampleMs: kf.stages?.sampleMs,
      kfChebMs: kf.stages?.chebMs,
      kfIdctMs: kf.stages?.idctMs,
      kfGradMs: kf.stages?.gradMs,
    };
    if (keyframeBaked) {
      logKeyframeBake(fromAnim ? "play/anim" : "bake");
    }
    state.lastNCoeff = (deg + 1) ** 3 * layers.length;
    if (Number.isFinite(fitRel)) state.lastFitRel = fitRel;

    if (cheb) state.worldCheb = cheb;
    else if (!fromAnim) state.worldCheb = null;
    state.fitDeg = deg;
    // GPU iso keyframes: upload only on bake / dens CPU lerp / full fit.
    // Warm anim ticks only update blend uniforms.
    const needUpload =
      fittedCount > 0 || keyframeBaked || densKeyframedCpu || !fromAnim;
    if (needUpload) {
      bakeChebVolume();
    } else if (keyframedCount > 0) {
      setConstraintKeyframeBlends(
        constraints
          .filter((c) => c.blend && c.id != null)
          .map((c) => ({ id: c.id, i0: c.blend.i0, i1: c.blend.i1, t: c.blend.t })),
      );
    }

    clipUniforms.uScale.value = densScale;
    clipUniforms.uSteps.value = steps;
    setBoxSize(boxSize);

    if (!fromAnim) resize();
    state.clipDirty = true;
    if (needUpload) {
      void prepareClipGpuForDegree(deg).then(() => {
        if (state.lastSceneBake) bakeChebVolume();
        syncClipPresentation();
        if (!useGpuClipPath()) {
          state.clipDirty = true;
          syncClipCpuVolume();
        }
      });
    } else {
      syncClipPresentation();
    }
  } catch (e) {
    try {
      compileAllExprs();
      setExprCompileOk(true);
    } catch {
      setExprCompileOk(false);
    }
    setErr(e instanceof Error ? e.message : String(e));
  }
}

export function scheduleUploadFit(delay = FIT_DEBOUNCE_MS, opts = {}) {
  state.pendingFitOpts = opts && typeof opts === "object" ? opts : {};
  if (state.fitTimer) clearTimeout(state.fitTimer);
  state.fitTimer = window.setTimeout(() => {
    state.fitTimer = 0;
    const fitOpts = state.pendingFitOpts;
    state.pendingFitOpts = {};
    if (!syncExprCompileState()) return;
    uploadFit(fitOpts);
  }, delay);
}

/** Scale / steps: no refit — update render uniforms only. */
export function applyRenderHyperparams() {
  const densScale = Number(els.scale.value) || 1;
  const steps = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
  els.steps.value = String(steps);
  clipUniforms.uScale.value = densScale;
  clipUniforms.uSteps.value = steps;
  state.clipDirty = true;
}

export function initKeyframeHandler() {
  /** Async keyframe fills: patch GPU slot + keep lastSceneBake in sync. */
  setKeyframeProgressHandler(({ layerId, index, frame, readyCount, K, done }) => {
    if (index >= 0 && frame && state.lastSceneBake?.constraints) {
      const c = state.lastSceneBake.constraints.find((x) => x.id === layerId);
      if (c && Array.isArray(c.keyframes) && index < c.keyframes.length) {
        c.keyframes[index] = frame;
      }
      if (isClipBakeGpuReady()) {
        patchConstraintKeyframeFrame(layerId, index, frame);
        state.clipDirty = true;
      }
    }
    if (done) {
      console.log(`[keyframes] async complete · ${layerId} · ${readyCount}/${K}`);
    }
  });
}

export function wirePipelineDom() {
  els.deg.addEventListener("input", () => scheduleUploadFit(200));
  els.deg.addEventListener("change", () => scheduleUploadFit(0));
  els.boxSize.addEventListener("input", () => scheduleUploadFit(200));
  els.boxSize.addEventListener("change", () => scheduleUploadFit(0));
  els.scale.addEventListener("input", applyRenderHyperparams);
  els.scale.addEventListener("change", applyRenderHyperparams);
  els.steps.addEventListener("input", applyRenderHyperparams);
  els.steps.addEventListener("change", applyRenderHyperparams);
  els.isoInterp?.addEventListener("change", async () => {
    const hermite = els.isoInterp.value === "hermite";
    if (!setIsoInterpHermite(hermite)) return;
    resetClipGpuProfile();
    state.clipDirty = true;
    await prepareClipGpuForDegree(state.fitDeg || Number(els.deg.value) || 23);
    refreshMetricsDump();
  });
}

export function handleColorChange() {
  // Colors only — skip Chebyshev refit; push RGB to GPU dens layers + constraints.
  if (state.lastSceneBake) {
    const items = listExpressions().filter((e) => e.enabled && String(e.latex || "").trim());
    const densCols = [];
    const consCols = [];
    for (const item of items) {
      let compiled;
      try {
        if (classifyExpr(item.latex).kind === "parameter") continue;
        compiled = compileExpr(item.latex);
      } catch {
        continue;
      }
      if (!compiled.usesSpace) continue;
      const role = resolveExprRole(item.role, compiled.kind);
      const rgb = layerRgbFromItem(item);
      if (role === "constraint") consCols.push(rgb);
      else densCols.push(rgb);
    }
    for (let i = 0; i < state.lastSceneBake.densLayers.length; i++) {
      if (densCols[i]) {
        state.lastSceneBake.densLayers[i].color = densCols[i].color;
        state.lastSceneBake.densLayers[i].color2 = densCols[i].color2;
        state.lastSceneBake.densLayers[i].colors = densCols[i].colors;
      }
    }
    for (let i = 0; i < state.lastSceneBake.constraints.length; i++) {
      if (consCols[i]) {
        state.lastSceneBake.constraints[i].color = consCols[i].color;
        state.lastSceneBake.constraints[i].color2 = consCols[i].color2;
        state.lastSceneBake.constraints[i].colors = consCols[i].colors;
      }
    }
    uploadSceneColors(state.lastSceneBake.densLayers.map((d) => d.colors || [d.color, d.color2]));
    if (isClipBakeGpuReady()) {
      uploadSceneVolumes({
        densLayers: state.lastSceneBake.densLayers,
        constraints: state.lastSceneBake.constraints,
        M: state.lastSceneBake.M,
      });
    }
    state.clipDirty = true;
  }
}
