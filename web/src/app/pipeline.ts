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
import { fitVectorField, seedFlowDye } from "../math/fitVector.js";
import { listExpressions, resolveExprRole } from "../model/expressions.js";
import { els, viewportSize } from "./dom.js";
import { state, FIT_DEBOUNCE_MS } from "./state.js";
import type {
  ChebFitTiming,
  ConstraintLayer,
  DensLayer,
  FlowLayer,
  KeyframeBlend,
  KeyframeFrame,
} from "../types/models.js";
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
import { tickFlowAdvectionGpu } from "../render/webgpu/flowAdvect.js";
import { clearClipGpuFrame } from "../render/webgpu/march.js";
import {
  setErr,
  setExprCompileOk,
  syncExprCompileState,
  refreshMetricsDump,
} from "./hud.js";

interface CachedLayer {
  kind: "density" | "constraint" | "flow";
  dens?: Float32Array;
  dye?: Float32Array;
  fx?: Float32Array;
  fy?: Float32Array;
  fz?: Float32Array;
  gx?: Float32Array;
  gy?: Float32Array;
  gz?: Float32Array;
  keyframes?: KeyframeFrame[];
  blend?: KeyframeBlend;
  cheb?: Float32Array;
  fitRel?: number;
  isoLevel?: number;
}

export function reseedFlowDye() {
  const bake = state.lastSceneBake;
  if (!bake?.flowLayers?.length) return;
  const half = bake.half ?? 0.5 * Number(els.boxSize.value);
  for (const f of bake.flowLayers) {
    f.dye = seedFlowDye(bake.M, half);
  }
  state.clipDirty = true;
  if (isClipBakeGpuReady()) bakeChebVolume();
}

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

export function tickFlowAdvection(dtMs: number) {
  if (!isClipBakeGpuReady() || !state.lastSceneBake?.flowLayers?.length) return false;
  return tickFlowAdvectionGpu(dtMs, state.lastSceneBake);
}

export function uploadFit(opts: { fromAnim?: boolean } = {}) {
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
      state.lastSceneBake = { densLayers: [], constraints: [], flowLayers: [], M: Math.max(2, deg + 1), dens: null };
      state.lastFitTiming = null;
      state.lastNCoeff = 0;
      state.lastFitRel = NaN;
      state.worldCheb = null;
      state.fitDeg = deg;
      if (isClipBakeGpuReady()) {
        uploadSceneVolumes({ densLayers: [], constraints: [], flowLayers: [], M: state.lastSceneBake.M });
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

    const densLayers: DensLayer[] = [];
    const constraints: ConstraintLayer[] = [];
    const flowLayers: FlowLayer[] = [];
    let cheb: Float32Array | null = null;
    let fitRel = NaN;
    let timingAcc: ChebFitTiming = { sampleMs: 0, chebMs: 0, monoMs: 0, l2Ms: 0, totalMs: 0 };
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

    const prevById = new Map<string, CachedLayer>();
    const lastBake = state.lastSceneBake;
    const canReuseCache =
      fromAnim &&
      dirty &&
      lastBake &&
      lastBake.deg === deg &&
      Math.abs((lastBake.half ?? NaN) - half) < 1e-12;
    if (canReuseCache && lastBake) {
      for (const d of lastBake.densLayers) {
        if (d.id) prevById.set(d.id, { kind: "density", ...d });
      }
      for (const c of lastBake.constraints) {
        if (c.id) prevById.set(c.id, { kind: "constraint", ...c });
      }
      for (const f of lastBake.flowLayers ?? []) {
        if (f.id) prevById.set(f.id, { kind: "flow", ...f, dens: f.dye });
      }
    }

    const baseParams = getParamValues();

    for (const L of layers) {
      const { color, color2, colors } = layerRgbFromItem(L.item);
      const depends =
        !dirty ||
        (L.role === "flow"
          ? L.vectorCompiled!.freeParams.some((p) => dirty.has(p))
          : L.compiled!.freeParams.some((p) => dirty.has(p)));
      const prev = canReuseCache && !depends ? prevById.get(L.item.id) : null;
      const prevHasKf =
        prev && Array.isArray(prev.keyframes) && prev.keyframes.length > 0;
      const reuseKind =
        L.role === "constraint" ? "constraint" : L.role === "flow" ? "flow" : "density";
      const reuseDens =
        prev &&
        prev.kind === reuseKind &&
        (prev.dens instanceof Float32Array ||
          (L.role === "flow" && prev.dye instanceof Float32Array) ||
          prevHasKf);

      if (reuseDens) {
        if (prevHasKf && prev.keyframes?.[0]) {
          M = Math.round(Math.cbrt(prev.keyframes[0].dens.length)) || M;
        } else if (prev.dens) {
          M = Math.round(Math.cbrt(prev.dens.length)) || M;
        } else if (prev.dye) {
          M = Math.round(Math.cbrt(prev.dye.length)) || M;
        }
        if (L.role === "constraint") {
          if (prevHasKf) {
            constraints.push({
              id: L.item.id,
              keyframes: prev.keyframes,
              blend: prev.blend ?? { i0: 0, i1: 0, t: 0 },
              color,
              color2,
              colors,
              isoLevel: L.compiled?.isoLevel ?? prev.isoLevel ?? 0,
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
              isoLevel: L.compiled?.isoLevel ?? prev.isoLevel ?? 0,
            });
          }
        } else if (L.role === "flow") {
          flowLayers.push({
            id: L.item.id,
            fx: prev.fx!,
            fy: prev.fy!,
            fz: prev.fz!,
            dye: prev.dye ?? prev.dens!,
            color,
            color2,
            colors,
            cheb: prev.cheb,
            fitRel: prev.fitRel,
          });
        } else {
          densLayers.push({ id: L.item.id, dens: prev.dens!, color, color2, colors });
        }
        if (!cheb && prev.cheb) {
          cheb = prev.cheb;
          fitRel = prev.fitRel ?? fitRel;
        }
        continue;
      }

      // Keyframe path: one dirty animated slider → GPU blend (iso) / CPU lerp (dens).
      const kfParam =
        fromAnim && depends && dirty && L.compiled
          ? keyframeAnimParam(L.compiled.freeParams, dirty)
          : null;
      if (kfParam && L.compiled && L.fn) {
        noteKeyframeLayer();
        keyframedCount++;
        if (L.role === "constraint") {
          const sample = ensureLayerKeyframes({
            layerId: L.item.id,
            latex: L.item.latex,
            role: "constraint",
            isoLevel: L.compiled?.isoLevel ?? 0,
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
            isoLevel: L.compiled?.isoLevel ?? 0,
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
            isoLevel: L.compiled?.isoLevel ?? 0,
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

      if (L.role === "flow") {
        const skipHeavy = fromAnim || layers.length > 1;
        const fit = fitVectorField(
          L.vectorCompiled!,
          L.vectorFn!,
          half,
          deg,
          { skipL2: skipHeavy },
        );
        fittedCount++;
        M = fit.M;
        const dye = seedFlowDye(M, half);
        flowLayers.push({
          id: L.item.id,
          fx: fit.fx,
          fy: fit.fy,
          fz: fit.fz,
          dye,
          color,
          color2,
          colors,
          cheb: fit.cheb,
          fitRel: fit.fitRel,
        });
        if (!cheb && fit.cheb) {
          cheb = fit.cheb;
          fitRel = fit.fitRel ?? fitRel;
        }
        continue;
      }

      const skipHeavy = fromAnim || layers.length > 1;
      const fit = fitChebyshev3D(L.fn!, half, deg, {
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
          isoLevel: L.compiled?.isoLevel ?? 0,
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
      for (const k of Object.keys(timingAcc) as (keyof ChebFitTiming)[]) {
        timingAcc[k] += fit.timing[k] || 0;
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
      flowLayers,
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
          .map((c) => ({
            id: c.id!,
            i0: c.blend!.i0,
            i1: c.blend!.i1,
            t: c.blend!.t,
          })),
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
  els.flowSpeed?.addEventListener("input", () => {
    state.flowSpeed = Math.max(0, Number(els.flowSpeed!.value) || 0);
  });
  els.flowDissipation?.addEventListener("input", () => {
    state.flowDissipation = Math.max(0, Math.min(0.5, Number(els.flowDissipation!.value) || 0));
  });
  els.reseedFlow?.addEventListener("click", () => reseedFlowDye());
}

export function handleColorChange() {
  // Colors only — skip Chebyshev refit; push RGB to GPU dens layers + constraints.
  if (state.lastSceneBake) {
    const items = listExpressions().filter((e) => e.enabled && String(e.latex || "").trim());
    const densCols: ReturnType<typeof layerRgbFromItem>[] = [];
    const consCols: ReturnType<typeof layerRgbFromItem>[] = [];
    const flowCols: ReturnType<typeof layerRgbFromItem>[] = [];
    for (const item of items) {
      let classified;
      try {
        classified = classifyExpr(item.latex);
        if (classified.kind === "parameter") continue;
      } catch {
        continue;
      }
      const role = resolveExprRole(item.role, classified.kind, item.latex);
      const rgb = layerRgbFromItem(item);
      if (role === "constraint") consCols.push(rgb);
      else if (role === "flow") flowCols.push(rgb);
      else {
        let compiled;
        try {
          compiled = compileExpr(item.latex);
        } catch {
          continue;
        }
        if (!compiled.usesSpace) continue;
        densCols.push(rgb);
      }
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
    for (let i = 0; i < (state.lastSceneBake.flowLayers?.length ?? 0); i++) {
      if (flowCols[i] && state.lastSceneBake.flowLayers?.[i]) {
        state.lastSceneBake.flowLayers[i].color = flowCols[i].color;
        state.lastSceneBake.flowLayers[i].color2 = flowCols[i].color2;
        state.lastSceneBake.flowLayers[i].colors = flowCols[i].colors;
      }
    }
    const allDensCols = [
      ...state.lastSceneBake.densLayers.map((d) => d.colors || [d.color, d.color2]),
      ...(state.lastSceneBake.flowLayers ?? []).map((f) => f.colors || [f.color, f.color2]),
    ];
    uploadSceneColors(allDensCols);
    if (isClipBakeGpuReady()) {
      uploadSceneVolumes({
        densLayers: state.lastSceneBake.densLayers,
        constraints: state.lastSceneBake.constraints,
        flowLayers: state.lastSceneBake.flowLayers,
        M: state.lastSceneBake.M,
        half: state.lastSceneBake.half,
      });
    }
    state.clipDirty = true;
  }
}
