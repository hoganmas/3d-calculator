import { bootQualityForTier, detectDeviceTier, type DeviceTier } from "./deviceTier.js";
import { els } from "./dom.js";
import { setPanelCollapsed } from "./panelLayout.js";
import { markMarchDirty } from "./presentation.js";
import { applyQualityFromState, isProdUi, syncQualitySliderDom } from "./quality.js";
import { state } from "./state.js";
import {
  PERF_ADAPT_MAX_STEPDOWNS,
  nextPerfAdaptStreak,
  perfAdaptBlockedByUserOverride,
  shouldTriggerPerfStepDown,
  stepDownQualityValues,
} from "./perfAdaptLogic.js";

export {
  PERF_ADAPT_FPS_THRESHOLD,
  PERF_ADAPT_FRAME_MS_THRESHOLD,
  PERF_ADAPT_STREAK_WINDOWS,
  PERF_ADAPT_STEP,
  PERF_ADAPT_MIN_QUALITY,
  PERF_ADAPT_MAX_STEPDOWNS,
  QUALITY_OVERRIDE_COOLDOWN_MS,
} from "./perfAdaptLogic.js";

export const MOBILE_INIT_KEY = "poly-cloud-mobile-init";

let lowPerfStreak = 0;
let perfHintUntil = 0;

export function getDeviceTier(): DeviceTier {
  return state.deviceTier;
}

/** Apply boot-time quality defaults for touch / low-end devices (fresh sessions only). */
export function applyBootPerfTier(restoredDocument: boolean) {
  const tier = detectDeviceTier({ webGpuFailed: state.webGpuFailed });
  state.deviceTier = tier;
  if (tier === "mobile") maybeAutoCollapsePanel();
  if (restoredDocument) return;
  if (!isProdUi()) return;

  const preset = bootQualityForTier(tier);
  state.precisionQuality = preset.precisionQuality;
  state.scalarQuality = preset.scalarQuality;
  state.surfaceQuality = preset.surfaceQuality;
  state.vectorQuality = preset.vectorQuality;
  applyQualityFromState({ refit: false });

  // Downscale overrides apply after quality so they don't perturb the
  // quality-derived step counts (scalarQuality/surfaceQuality also drive
  // volumeSteps/isoSteps — see BootQualityPreset).
  let overrode = false;
  if (preset.marchDownscaleOverride != null) {
    els.marchDownscale.value = String(preset.marchDownscaleOverride);
    overrode = true;
  }
  if (preset.isoMarchDownscaleOverride != null && els.isoMarchDownscale) {
    els.isoMarchDownscale.value = String(preset.isoMarchDownscaleOverride);
    overrode = true;
  }
  if (overrode) markMarchDirty();
}

function maybeAutoCollapsePanel() {
  try {
    if (localStorage.getItem(MOBILE_INIT_KEY)) return;
    setPanelCollapsed(true);
    localStorage.setItem(MOBILE_INIT_KEY, "1");
  } catch {
    /* ignore */
  }
}

function stepDownAllQualitySliders() {
  const next = stepDownQualityValues({
    precisionQuality: state.precisionQuality,
    scalarQuality: state.scalarQuality,
    surfaceQuality: state.surfaceQuality,
    vectorQuality: state.vectorQuality,
  });
  state.precisionQuality = next.precisionQuality;
  state.scalarQuality = next.scalarQuality;
  state.surfaceQuality = next.surfaceQuality;
  state.vectorQuality = next.vectorQuality;
  applyQualityFromState({ refit: true });
  syncQualitySliderDom();
}

/** Runtime FPS watchdog — call from the loop FPS window (~500ms). */
export function tickPerfAdapt(now: number) {
  if (!isProdUi()) return;
  // A step-down calls applyQualityFromState({ refit: true }), which
  // re-bakes the whole animation. On mobile that refit itself is often
  // the thing tanking the frame rate, so this watchdog can trigger a
  // refit, see fps stay low from the refit's own cost, and step down
  // again — making the perf problem it's meant to fix worse.
  if (state.deviceTier === "mobile") return;

  if (perfAdaptBlockedByUserOverride(now, state.qualityUserOverrideAt)) {
    lowPerfStreak = 0;
    return;
  }
  if (state.perfAdaptStepDownCount >= PERF_ADAPT_MAX_STEPDOWNS) return;

  lowPerfStreak = nextPerfAdaptStreak(lowPerfStreak, state.loopFps, state.frameDtSmooth);
  if (!shouldTriggerPerfStepDown(lowPerfStreak)) return;

  lowPerfStreak = 0;
  state.perfAdaptStepDownCount++;
  perfHintUntil = now + 3000;
  stepDownAllQualitySliders();
}

export function perfAdaptHudSuffix(now: number): string {
  if (now < perfHintUntil) return " · quality adjusted";
  return "";
}
