/**
 * Laplacian scene document — serialize, validate re-exports, apply.
 */
import { PRESETS } from "../../math/fit.js";
import { listExpressions, setExpressions, getExprWarning } from "../../model/expressions.js";
import { listParamNames, getParam } from "../../model/params.js";
import type { AnimMode, ExprItem, ParamState, PresetParamSeed } from "../../types/models.js";
import { els } from "../dom.js";
import { syncExprCompileState } from "../hud.js";
import { applyRenderHyperparams } from "../pipeline.js";
import {
  syncMarchSlider,
  syncClipPresentation,
  syncShowGridAxesUi,
  syncBoundsSlider,
  syncIsometricUi,
} from "../presentation.js";
import { syncQualitySlidersFromSettings } from "../quality.js";
import { camera, controls, DEFAULT_FOV } from "../scene.js";
import { state } from "../state.js";
import {
  bumpDocumentRevision,
  getDocumentRevision,
  setDocumentRevision,
  validateDocument,
  type LaplacianDocument,
  type LaplacianCameraSnapshot,
  type LaplacianFlowSnapshot,
  type LaplacianRenderSnapshot,
} from "./documentSchema.js";

export {
  DOCUMENT_VERSION,
  bumpDocumentRevision,
  getDocumentRevision,
  setDocumentRevision,
  validateDocument,
  type LaplacianDocument,
  type LaplacianCameraSnapshot,
  type LaplacianDocumentMeta,
  type LaplacianFlowSnapshot,
  type LaplacianRenderSnapshot,
} from "./documentSchema.js";

let persistSuspended = false;

export function isPersistSuspended() {
  return persistSuspended;
}

function stripTrailingBlank(exprs: ExprItem[]) {
  const copy = exprs.slice();
  while (copy.length > 0) {
    const last = copy[copy.length - 1]!;
    if (String(last.latex || "").trim()) break;
    copy.pop();
  }
  return copy;
}

function serializeExpression(item: ExprItem): Partial<ExprItem> {
  return {
    id: item.id,
    latex: item.latex,
    enabled: item.enabled,
    color: item.color,
    color2: item.color2,
    colors: item.colors?.slice(),
    sliderMin: item.sliderMin,
    sliderMax: item.sliderMax,
    sliderSpeed: item.sliderSpeed,
    sliderAnimating: item.sliderAnimating,
    sliderPhase: item.sliderPhase,
    sliderAnimMode: item.sliderAnimMode,
    autoParam: item.autoParam,
  };
}

/** WebMCP / persistence: expression row snapshot. */
export function serializeExpr(item: ExprItem) {
  return {
    ...serializeExpression(item),
    warning: getExprWarning(item.id) ?? null,
  };
}

/** WebMCP / persistence: named parameter snapshot. */
export function serializeParam(name: string) {
  const p = getParam(name);
  if (!p) return null;
  return {
    name,
    value: p.value,
    min: p.min,
    max: p.max,
    speed: p.speed,
    animating: p.animating,
    animMode: p.animMode,
    driven: p.driven,
    latex: p.latex,
    error: p.error,
    exprId: p.exprId,
  };
}

export function getRenderSettingsSnapshot(): LaplacianRenderSnapshot {
  return {
    deg: Number(els.deg.value),
    scale: Number(els.scale.value),
    steps: Number(els.steps.value),
    isoSteps: Number(els.isoSteps?.value ?? els.steps.value),
    boxSize: Number(els.boxSize.value),
    marchDownscale: Number(els.marchDownscale.value),
    isoMarchDownscale: Number(els.isoMarchDownscale?.value ?? els.marchDownscale.value),
    showGridAxes: state.showGridAxes,
    preset: els.preset.value,
  };
}

function getFlowSnapshot(): LaplacianFlowSnapshot {
  return {
    flowDt: state.flowDt,
    flowSpeed: state.flowSpeed,
    flowVMax: state.flowVMax,
    flowOpacity: state.flowOpacity,
    flowAgeMax: state.flowAgeMax,
    flowParticleCount: state.flowParticleCount,
    flowTrailSteps: state.flowTrailSteps,
    flowTrailWidth: state.flowTrailWidth,
  };
}

function getCameraSnapshot(): LaplacianCameraSnapshot {
  return {
    position: [camera.position.x, camera.position.y, camera.position.z],
    target: [controls.target.x, controls.target.y, controls.target.z],
    fov: camera.fov,
  };
}

function serializeParams(): Record<string, Partial<ParamState>> {
  const out: Record<string, Partial<ParamState>> = {};
  for (const name of listParamNames()) {
    const p = getParam(name);
    if (!p) continue;
    out[name] = {
      value: p.value,
      min: p.min,
      max: p.max,
      speed: p.speed,
      animating: p.animating,
      animMode: p.animMode,
      phase: p.phase,
    };
  }
  return out;
}

export function serializeDocument(): LaplacianDocument {
  const rev = bumpDocumentRevision();
  return {
    format: "laplacian",
    version: 1,
    revision: rev,
    savedAt: new Date().toISOString(),
    meta: { preset: els.preset.value },
    expressions: stripTrailingBlank(listExpressions()).map(serializeExpression),
    params: serializeParams(),
    render: getRenderSettingsSnapshot(),
    flow: getFlowSnapshot(),
    camera: getCameraSnapshot(),
  };
}

function paramSeedFromDoc(params: Record<string, Partial<ParamState>>): Record<string, PresetParamSeed> {
  const seed: Record<string, PresetParamSeed> = {};
  for (const [name, p] of Object.entries(params)) {
    seed[name] = {
      value: p.value,
      min: p.min,
      max: p.max,
      speed: p.speed,
      animating: p.animating,
      animate: p.animating,
      phase: p.phase,
      animMode: p.animMode as AnimMode | undefined,
    };
  }
  return seed;
}

function syncRenderDom(render: LaplacianRenderSnapshot) {
  els.deg.value = String(Math.round(render.deg));
  els.scale.value = String(render.scale);
  els.steps.value = String(Math.round(render.steps));
  if (els.isoSteps) els.isoSteps.value = String(Math.round(render.isoSteps));
  els.boxSize.value = String(render.boxSize);
  syncBoundsSlider();
  els.marchDownscale.value = String(Math.round(render.marchDownscale));
  if (els.isoMarchDownscale) {
    els.isoMarchDownscale.value = String(Math.round(render.isoMarchDownscale));
  }
  state.showGridAxes = render.showGridAxes;
  syncShowGridAxesUi();
  els.preset.value = render.preset in PRESETS ? render.preset : "sincos";
  syncMarchSlider();
  applyRenderHyperparams();
  syncClipPresentation();
  syncQualitySlidersFromSettings();
}

function syncFlowDom(flow: LaplacianFlowSnapshot) {
  state.flowDt = flow.flowDt;
  state.flowSpeed = flow.flowSpeed;
  state.flowVMax = flow.flowVMax;
  state.flowOpacity = flow.flowOpacity;
  state.flowAgeMax = flow.flowAgeMax;
  state.flowParticleCount = flow.flowParticleCount;
  state.flowTrailSteps = flow.flowTrailSteps;
  state.flowTrailWidth = flow.flowTrailWidth;
  if (els.flowDt) els.flowDt.value = String(flow.flowDt);
  if (els.flowSpeed) els.flowSpeed.value = String(flow.flowSpeed);
  if (els.flowVMax) els.flowVMax.value = String(flow.flowVMax);
  if (els.flowOpacity) els.flowOpacity.value = String(flow.flowOpacity);
  if (els.flowAgeMax) els.flowAgeMax.value = String(flow.flowAgeMax);
  if (els.flowParticleCount) els.flowParticleCount.value = String(flow.flowParticleCount);
  if (els.flowTrailSteps) els.flowTrailSteps.value = String(flow.flowTrailSteps);
  if (els.flowTrailWidth) els.flowTrailWidth.value = String(flow.flowTrailWidth);
}

/** Apply a validated document (DOM/state only; uploadFit performs refit). */
export async function applyDocument(doc: LaplacianDocument) {
  persistSuspended = true;
  try {
    setDocumentRevision(doc.revision);
    state.pendingParamSeed = paramSeedFromDoc(doc.params);
    syncRenderDom(doc.render);
    syncFlowDom(doc.flow);
    if (doc.expressions.length) {
      setExpressions(doc.expressions);
    } else {
      setExpressions([{ latex: "" }]);
    }
    if (doc.camera) {
      camera.position.set(doc.camera.position[0], doc.camera.position[1], doc.camera.position[2]);
      controls.target.set(doc.camera.target[0], doc.camera.target[1], doc.camera.target[2]);
      camera.fov = doc.camera.fov ?? DEFAULT_FOV;
      camera.updateProjectionMatrix();
      controls.update();
      state.clipDirty = true;
    }
    syncIsometricUi();
    syncExprCompileState();
  } finally {
    persistSuspended = false;
  }
}

export { getDocumentRevision as getCurrentRevision };
