/**
 * Production UI: quality sliders map to render/fit/flow hyperparameters.
 * Development keeps direct controls on the same underlying settings.
 */
import { els } from "./dom.js";
import {
  applyRenderHyperparams,
  scheduleUploadFit,
} from "./pipeline.js";
import { markMarchDirty, syncMarchSlider, syncSettingsLiquidThumbs } from "./presentation.js";
import { reseedFlowParticles } from "../render/webgpu/flowParticles.js";
import { state } from "./state.js";
import {
  inferQualityFromSettings,
  qualityToDeg,
  qualityToMarchDownscale,
  qualityToParticleCount,
  qualityToSteps,
  qualityToTrailSteps,
} from "./qualityMapping.js";

export {
  qualityToDeg,
  qualityToMarchDownscale,
  qualityToParticleCount,
  qualityToSteps,
  qualityToTrailSteps,
} from "./qualityMapping.js";

export function isProdUi(): boolean {
  return import.meta.env.PROD;
}

export function syncQualitySliderDom() {
  if (els.precisionQuality) els.precisionQuality.value = String(state.precisionQuality);
  if (els.scalarQuality) els.scalarQuality.value = String(state.scalarQuality);
  if (els.surfaceQuality) els.surfaceQuality.value = String(state.surfaceQuality);
  if (els.vectorQuality) els.vectorQuality.value = String(state.vectorQuality);
  syncSettingsLiquidThumbs();
}

/** Infer prod slider positions from current underlying render settings. */
export function syncQualitySlidersFromSettings() {
  if (!isProdUi()) return;
  const inferred = inferQualityFromSettings({
    marchDownscale: Number(els.marchDownscale.value) || 1,
    deg: Number(els.deg.value) || state.fitDeg,
    steps: Number(els.steps.value) || 16,
    flowParticleCount: state.flowParticleCount,
    flowTrailSteps: state.flowTrailSteps,
  });
  state.precisionQuality = inferred.precisionQuality;
  state.scalarQuality = inferred.scalarQuality;
  state.surfaceQuality = inferred.surfaceQuality;
  state.vectorQuality = inferred.vectorQuality;
  syncQualitySliderDom();
}

export function applyQualityFromState(opts: { refit?: boolean; reseedFlow?: boolean } = {}) {
  if (!isProdUi()) return;

  const marchDownscale = qualityToMarchDownscale(state.scalarQuality);
  const steps = qualityToSteps(state.surfaceQuality);
  const deg = qualityToDeg(state.precisionQuality);
  const particles = qualityToParticleCount(state.vectorQuality);
  const trailSteps = qualityToTrailSteps(state.vectorQuality);

  const prevDeg = Number(els.deg.value) || state.fitDeg;
  const prevParticles = state.flowParticleCount;
  const prevTrailSteps = state.flowTrailSteps;

  els.deg.value = String(deg);
  els.steps.value = String(steps);
  els.marchDownscale.value = String(marchDownscale);
  state.flowParticleCount = particles;
  state.flowTrailSteps = trailSteps;
  if (els.flowParticleCount) els.flowParticleCount.value = String(particles);
  if (els.flowTrailSteps) els.flowTrailSteps.value = String(trailSteps);

  applyRenderHyperparams();
  syncMarchSlider();
  markMarchDirty();

  const refit = opts.refit !== false && deg !== prevDeg;
  if (refit) scheduleUploadFit(0);

  const reseed =
    opts.reseedFlow !== false &&
    state.lastSceneBake?.flowLayers?.length &&
    (particles !== prevParticles || trailSteps !== prevTrailSteps);
  if (reseed) {
    state.clipDirty = true;
    reseedFlowParticles();
  }
}

export function initProdSettingsUi() {
  if (isProdUi()) {
    document.documentElement.dataset.prodUi = "true";
    syncQualitySlidersFromSettings();
  } else {
    delete document.documentElement.dataset.prodUi;
  }
}

function clampQ(q: number): number {
  return Math.min(100, Math.max(0, Math.round(q)));
}

type QualityKind = "precision" | "scalar" | "surface" | "vector";

export function wireQualityControls(onChange: () => void) {
  if (!isProdUi()) return;

  const onInput = (kind: QualityKind) => {
    const input =
      kind === "precision"
        ? els.precisionQuality
        : kind === "scalar"
          ? els.scalarQuality
          : kind === "surface"
            ? els.surfaceQuality
            : els.vectorQuality;
    if (!input) return;
    const q = clampQ(Number(input.value));
    if (kind === "precision") state.precisionQuality = q;
    else if (kind === "scalar") state.scalarQuality = q;
    else if (kind === "surface") state.surfaceQuality = q;
    else state.vectorQuality = q;
    applyQualityFromState();
    onChange();
  };

  els.precisionQuality?.addEventListener("input", () => onInput("precision"));
  els.scalarQuality?.addEventListener("input", () => onInput("scalar"));
  els.surfaceQuality?.addEventListener("input", () => onInput("surface"));
  els.vectorQuality?.addEventListener("input", () => onInput("vector"));
}
