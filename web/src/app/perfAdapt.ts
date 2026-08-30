import { bootQualityForTier, detectDeviceTier, type DeviceTier } from "./deviceTier.js";
import { setPanelCollapsed } from "./panelLayout.js";
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
  if (!isProdUi() || restoredDocument) return;

  const tier = detectDeviceTier({ webGpuFailed: state.webGpuFailed });
  state.deviceTier = tier;

  const preset = bootQualityForTier(tier);
  state.precisionQuality = preset.precisionQuality;
  state.scalarQuality = preset.scalarQuality;
  state.surfaceQuality = preset.surfaceQuality;
  state.vectorQuality = preset.vectorQuality;
  applyQualityFromState({ refit: false });

  if (tier === "mobile") maybeAutoCollapsePanel();
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
