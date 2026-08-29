import { MAX_DEG } from "../math/limits.js";
import { fitScalarField } from "../math/fit.js";
import { idctChebGrad3D } from "../math/idct.js";
import {
  ensureLobattoDegree,
  idctLobatto3D,
  lobattoChebToSeries,
} from "../math/chebLobatto.js";
import {
  clearLobattoFitCache,
  getLobattoLayerCache,
  isProgressiveLobattoEnabled,
  scheduleProgressiveUploadFit,
  setLobattoLayerCache,
} from "./progressiveFit.js";
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
  sampleFlowLayerKeyframes,
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
} from "../render/webgpu/march.js";
import { reseedFlowDyeBuffers } from "../render/webgpu/flowIbfv.js";
import { reseedFlowParticles } from "../render/webgpu/flowParticles.js";
import { compileExpr, classifyExpr } from "../math/fit.js";
import { fitVectorField } from "../math/fitVector.js";
import { listExpressions, resolveExprRole } from "../model/expressions.js";
import { els, viewportSize } from "./dom.js";
import { state, FIT_DEBOUNCE_MS } from "./state.js";
import type {
  ChebFitTiming,
  CloudLayer,
  IsosurfaceLayer,
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
import { resize, syncClipPresentation, syncShowGridAxesUi } from "./presentation.js";
import { clearClipGpuFrame } from "../render/webgpu/march.js";
import {
  setErr,
  setExprCompileOk,
  syncExprCompileState,
  refreshMetricsDump,
} from "./hud.js";
import { scheduleAutosave } from "./persistence/autosave.js";
import { tryMarkSplashBakeReady } from "./splash.js";
import { allKeyframesComplete } from "../model/keyframes.js";

interface CachedLayer {
  kind: "cloud" | "isosurface" | "flow";
  dens?: Float32Array;
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

export function tickGpuKeyframeBlends() {
  const isoLayers = state.lastSceneBake?.isosurfaceLayers;
  if (!isoLayers?.length || !isClipBakeGpuReady()) return false;
  /** @type {{ id: string, i0: number, i1: number, t: number }[]} */
  const blends = [];
  for (const c of isoLayers) {
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

export function uploadFit(
  opts: {
    fromAnim?: boolean;
    fitDeg?: number;
    progressive?: boolean;
    progressiveFinal?: boolean;
  } = {},
) {
  const fromAnim = !!opts.fromAnim;
  const progressive = !!opts.progressive;
  setErr("");
  try {
    const boxSize = Number(els.boxSize.value);
    const uiDeg = Number(els.deg.value);
    const deg = opts.fitDeg ?? uiDeg;
    const densScale = Number(els.scale.value);
    const steps = Math.min(96, Math.max(8, Number(els.steps.value) || 32));
    els.steps.value = String(steps);
    if (!(boxSize > 0)) throw new Error("box size must be > 0");
    if (deg < 1 || deg > MAX_DEG) throw new Error(`poly deg must be 1…${MAX_DEG}`);
    const half = 0.5 * boxSize;

    if (!progressive && !fromAnim) {
      clearLobattoFitCache();
    }

    const tUpload = performance.now();
    const { layers } = compileAllExprs({ rebuildUi: false });
    setExprCompileOk(true);

    // No visible / non-empty expressions → clear volume, draw nothing.
    if (!layers.length) {
      clearKeyframeCaches();
      state.lastSceneBake = { cloudLayers: [], isosurfaceLayers: [], flowLayers: [], M: Math.max(2, deg + 1), dens: null };
      state.lastFitTiming = null;
      state.lastNCoeff = 0;
      state.lastFitRel = NaN;
      state.worldCheb = null;
      state.fitDeg = deg;
      if (isClipBakeGpuReady()) {
        uploadSceneVolumes({ cloudLayers: [], isosurfaceLayers: [], flowLayers: [], M: state.lastSceneBake.M });
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
      tryMarkSplashBakeReady(false);
      return;
    }

    const cloudLayers: CloudLayer[] = [];
    const isosurfaceLayers: IsosurfaceLayer[] = [];
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
      for (const d of lastBake.cloudLayers) {
        if (d.id) prevById.set(d.id, { kind: "cloud", ...d });
      }
      for (const c of lastBake.isosurfaceLayers) {
        if (c.id) prevById.set(c.id, { kind: "isosurface", ...c });
      }
      for (const f of lastBake.flowLayers ?? []) {
        if (f.id) prevById.set(f.id, { kind: "flow", ...f });
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
        L.role === "isosurface" ? "isosurface" : L.role === "flow" ? "flow" : "cloud";
      const reuseDens =
        prev &&
        prev.kind === reuseKind &&
        (prev.dens instanceof Float32Array ||
          (L.role === "flow" && prev.fx instanceof Float32Array) ||
          prevHasKf);

      if (reuseDens) {
        if (prevHasKf && prev.keyframes?.[0]) {
          M = Math.round(Math.cbrt(prev.keyframes[0].dens.length)) || M;
        } else if (prev.dens) {
          M = Math.round(Math.cbrt(prev.dens.length)) || M;
        } else if (prev.fx) {
          M = Math.round(Math.cbrt(prev.fx.length)) || M;
        }
        if (L.role === "isosurface") {
          if (prevHasKf) {
            isosurfaceLayers.push({
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
            isosurfaceLayers.push({
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
            color,
            color2,
            colors,
            cheb: prev.cheb,
            fitRel: prev.fitRel,
          });
        } else {
          cloudLayers.push({ id: L.item.id, dens: prev.dens!, color, color2, colors });
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
        if (L.role === "isosurface") {
          const sample = ensureLayerKeyframes({
            layerId: L.item.id,
            latex: L.item.latex,
            role: "isosurface",
            isoLevel: L.compiled?.isoLevel ?? 0,
            paramName: kfParam,
            compiled: L.compiled,
            baseParams,
            half,
            deg,
          });
          if (sample.baked) keyframeBaked = true;
          M = sample.M || M;
          isosurfaceLayers.push({
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
            role: "cloud",
            isoLevel: L.compiled?.isoLevel ?? 0,
            paramName: kfParam,
            compiled: L.compiled,
            baseParams,
            half,
            deg,
          });
          if (sample.baked) keyframeBaked = true;
          M = sample.M || M;
          cloudLayers.push({
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
        const kfParam =
          fromAnim && depends && dirty && L.vectorCompiled
            ? keyframeAnimParam(L.vectorCompiled.freeParams, dirty)
            : null;
        if (kfParam && L.vectorCompiled && L.vectorFn) {
          noteKeyframeLayer();
          keyframedCount++;
          const sample = sampleFlowLayerKeyframes({
            layerId: L.item.id,
            latex: L.item.latex,
            role: "flow",
            paramName: kfParam,
            vectorCompiled: L.vectorCompiled,
            baseParams,
            half,
            deg,
          });
          if (sample.baked) keyframeBaked = true;
          M = sample.M || M;
          flowLayers.push({
            id: L.item.id,
            fx: sample.fx,
            fy: sample.fy,
            fz: sample.fz,
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
          continue;
        }

        const skipHeavy = fromAnim || layers.length > 1;
        const fit = fitVectorField(
          L.vectorCompiled!,
          L.vectorFn!,
          half,
          deg,
          { skipL2: skipHeavy, params: baseParams },
        );
        fittedCount++;
        M = fit.M;
        flowLayers.push({
          id: L.item.id,
          fx: fit.fx,
          fy: fit.fy,
          fz: fit.fz,
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

      const skipHeavy = fromAnim || layers.length > 1 || progressive;
      const useLobatto =
        progressive &&
        isProgressiveLobattoEnabled() &&
        L.role === "cloud" &&
        !L.compiled?.operator;

      let scalarFit;
      if (useLobatto && L.fn) {
        const cached = getLobattoLayerCache(L.item.id);
        const lob = ensureLobattoDegree(cached, L.fn, half, deg);
        setLobattoLayerCache(L.item.id, lob);
        const idct = idctLobatto3D(lob.cheb, lob.deg, lob.deg + 1);
        scalarFit = {
          dens: idct.dens,
          cheb: lob.cheb,
          fitRelL2: NaN,
          M: idct.M,
          deg: lob.deg,
        };
      } else {
        scalarFit = fitScalarField(L.compiled!, L.fn!, half, deg, {
          skipL2: skipHeavy,
          skipMono: skipHeavy,
        });
      }
      fittedCount++;
      M = scalarFit.M;
      if (L.role === "isosurface") {
        const gradCheb = useLobatto
          ? lobattoChebToSeries(scalarFit.cheb, scalarFit.deg)
          : scalarFit.cheb;
        const grad = idctChebGrad3D(gradCheb, scalarFit.deg, scalarFit.deg + 1);
        isosurfaceLayers.push({
          id: L.item.id,
          dens: scalarFit.dens,
          gx: grad.gx,
          gy: grad.gy,
          gz: grad.gz,
          color,
          color2,
          colors,
          isoLevel: L.compiled?.isoLevel ?? 0,
          cheb: scalarFit.cheb,
          fitRel: scalarFit.fitRelL2,
        });
      } else {
        cloudLayers.push({
          id: L.item.id,
          dens: scalarFit.dens,
          color,
          color2,
          colors,
          cheb: scalarFit.cheb,
          fitRel: scalarFit.fitRelL2,
        });
      }
      if (!cheb) {
        cheb = scalarFit.cheb;
        fitRel = scalarFit.fitRelL2;
      }
      if (scalarFit.timing) {
        for (const k of Object.keys(timingAcc) as (keyof ChebFitTiming)[]) {
          timingAcc[k] += scalarFit.timing[k] || 0;
        }
      }
    }

    // WebGL preview texture: sum of cloud layers (skipped when WebGPU is active).
    let densSum = null;
    if (cloudLayers.length && !useGpuClipPath()) {
      densSum = new Float32Array(M * M * M);
      for (const d of cloudLayers) {
        for (let i = 0; i < densSum.length; i++) densSum[i] += d.dens[i] || 0;
      }
    }

    state.lastSceneBake = {
      cloudLayers,
      isosurfaceLayers,
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
    else if (!fromAnim && opts.progressiveFinal) state.worldCheb = null;
    state.fitDeg = deg;
    // GPU iso keyframes: upload only on bake / dens CPU lerp / full fit.
    // Warm anim ticks only update blend uniforms.
    const needUpload =
      fittedCount > 0 || keyframeBaked || densKeyframedCpu || !fromAnim;
    if (needUpload) {
      bakeChebVolume();
    } else if (keyframedCount > 0) {
      setConstraintKeyframeBlends(
        isosurfaceLayers
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

    tryMarkSplashBakeReady(layers.length > 0);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    try {
      compileAllExprs();
      setExprCompileOk(true);
    } catch {
      setExprCompileOk(false);
    }
    setErr(e instanceof Error ? e.message : String(e));
    if (!state.lastSceneBake) {
      state.lastSceneBake = {
        cloudLayers: [],
        isosurfaceLayers: [],
        flowLayers: [],
        M: 2,
        dens: null,
      };
    }
    tryMarkSplashBakeReady(false);
  }
}

export function scheduleUploadFit(delay = FIT_DEBOUNCE_MS, opts = {}) {
  scheduleProgressiveUploadFit(uploadFit, delay, opts);
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
    if (index >= 0 && frame && state.lastSceneBake?.isosurfaceLayers) {
      const c = state.lastSceneBake.isosurfaceLayers.find((x) => x.id === layerId);
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
      if (allKeyframesComplete()) {
        tryMarkSplashBakeReady(true);
      }
    }
  });
}

function autosave() {
  scheduleAutosave();
}

export function wirePipelineDom() {
  els.deg.addEventListener("input", () => {
    scheduleUploadFit(200);
    autosave();
  });
  els.deg.addEventListener("change", () => {
    scheduleUploadFit(0);
    autosave();
  });
  els.boxSize.addEventListener("input", () => {
    scheduleUploadFit(200);
    autosave();
  });
  els.boxSize.addEventListener("change", () => {
    scheduleUploadFit(0);
    autosave();
  });
  els.scale.addEventListener("input", () => {
    applyRenderHyperparams();
    autosave();
  });
  els.scale.addEventListener("change", () => {
    applyRenderHyperparams();
    autosave();
  });
  els.steps.addEventListener("input", () => {
    applyRenderHyperparams();
    autosave();
  });
  els.steps.addEventListener("change", () => {
    applyRenderHyperparams();
    autosave();
  });
  els.marchDownscale.addEventListener("input", autosave);
  els.marchDownscale.addEventListener("change", autosave);
  syncShowGridAxesUi();
  els.toggleGridAxes?.addEventListener("click", () => {
    state.showGridAxes = !state.showGridAxes;
    syncShowGridAxesUi();
    syncClipPresentation();
    autosave();
  });
  els.flowAlpha?.addEventListener("input", () => {
    state.flowAlpha = Math.max(0, Math.min(1, Number(els.flowAlpha!.value) || 0));
    autosave();
  });
  els.flowVizMode?.addEventListener("change", () => {
    state.flowVizMode = els.flowVizMode!.value === "ibfv" ? "ibfv" : "particles";
    autosave();
  });
  els.flowParticleCount?.addEventListener("change", () => {
    state.flowParticleCount = Math.max(100, Math.min(32000, Number(els.flowParticleCount!.value) || 1000));
    if (state.lastSceneBake?.flowLayers?.length) {
      state.clipDirty = true;
      reseedFlowParticles();
    }
    autosave();
  });
  els.flowGridMode?.addEventListener("change", () => {
    state.flowGridPoints = els.flowGridMode!.value === "points";
    reseedFlowDyeBuffers();
    reseedFlowParticles();
    autosave();
  });
  els.flowNoiseScale?.addEventListener("input", () => {
    state.flowNoiseScale = Math.max(0.05, Number(els.flowNoiseScale!.value) || 0);
    reseedFlowDyeBuffers();
    reseedFlowParticles();
    autosave();
  });
  els.flowDt?.addEventListener("input", () => {
    state.flowDt = Math.max(0.001, Number(els.flowDt!.value) || 0);
    autosave();
  });
  els.flowSpeed?.addEventListener("input", () => {
    state.flowSpeed = Math.max(0.05, Math.min(10, Number(els.flowSpeed!.value) || 0.1));
    autosave();
  });
  els.flowTrailSteps?.addEventListener("change", () => {
    state.flowTrailSteps = Math.max(2, Math.min(32, Number(els.flowTrailSteps!.value) || 32));
    reseedFlowParticles();
    autosave();
  });
  els.flowTrailWidth?.addEventListener("input", () => {
    state.flowTrailWidth = Math.max(1, Math.min(32, Number(els.flowTrailWidth!.value) || 10));
    autosave();
  });
  els.flowVMax?.addEventListener("input", () => {
    state.flowVMax = Math.max(0, Number(els.flowVMax!.value) || 0);
    autosave();
  });
  els.flowOpacity?.addEventListener("input", () => {
    state.flowOpacity = Math.max(0.01, Math.min(2, Number(els.flowOpacity!.value) || 0.5));
    autosave();
  });
  els.flowAgeMax?.addEventListener("input", () => {
    state.flowAgeMax = Math.max(1, Math.min(120, Number(els.flowAgeMax!.value) || 30));
    autosave();
  });
}

export function handleColorChange() {
  // Colors only — skip Chebyshev refit; push RGB to GPU cloud + isosurface layers.
  if (state.lastSceneBake) {
    const items = listExpressions().filter((e) => e.enabled && String(e.latex || "").trim());
    const cloudCols: ReturnType<typeof layerRgbFromItem>[] = [];
    const isoCols: ReturnType<typeof layerRgbFromItem>[] = [];
    const flowCols: ReturnType<typeof layerRgbFromItem>[] = [];
    for (const item of items) {
      let classified;
      try {
        classified = classifyExpr(item.latex);
        if (classified.kind === "parameter") continue;
      } catch {
        continue;
      }
      const fieldLatex = classified.compileLatex;
      const role = resolveExprRole(item.role, classified.kind, fieldLatex);
      const rgb = layerRgbFromItem(item);
      if (role === "isosurface") isoCols.push(rgb);
      else if (role === "flow") flowCols.push(rgb);
      else {
        let compiled;
        try {
          compiled = compileExpr(fieldLatex);
        } catch {
          continue;
        }
        if (!compiled.usesSpace) continue;
        cloudCols.push(rgb);
      }
    }
    for (let i = 0; i < state.lastSceneBake.cloudLayers.length; i++) {
      if (cloudCols[i]) {
        state.lastSceneBake.cloudLayers[i].color = cloudCols[i].color;
        state.lastSceneBake.cloudLayers[i].color2 = cloudCols[i].color2;
        state.lastSceneBake.cloudLayers[i].colors = cloudCols[i].colors;
      }
    }
    for (let i = 0; i < state.lastSceneBake.isosurfaceLayers.length; i++) {
      if (isoCols[i]) {
        state.lastSceneBake.isosurfaceLayers[i].color = isoCols[i].color;
        state.lastSceneBake.isosurfaceLayers[i].color2 = isoCols[i].color2;
        state.lastSceneBake.isosurfaceLayers[i].colors = isoCols[i].colors;
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
      ...state.lastSceneBake.cloudLayers.map((d) => d.colors || [d.color, d.color2]),
      ...(state.lastSceneBake.flowLayers ?? []).map((f) => f.colors || [f.color, f.color2]),
    ];
    uploadSceneColors(allDensCols);
    state.clipDirty = true;
  }
}
