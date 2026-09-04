import "mathlive";
import "mathlive/static.css";
import "./ui/theme.css";
import { initTheme } from "./ui/theme.js";
import { mountExprList } from "./ui/expr-sidebar/mount.js";
import { setExpressionsOnChange } from "./model/expressions.js";
import { anyParamAnimating, ensureParamAnimationFromExprs } from "./model/params.js";
import { syncExprCompileState, getExpressionErrorReport } from "./app/hud.js";
import { initDom, els, initPanelResize, initPanelToggle, initPanelCollapse, initPanelDismiss, refreshPanelToggleChrome, runPanelTransition } from "./app/dom.js";
import { initScene, bindClipUniforms, resetCameraView } from "./app/scene.js";
import { initCompile, applyPreset } from "./app/compile.js";
import {
  scheduleUploadFit,
  uploadFit,
  initKeyframeHandler,
  wirePipelineDom,
  handleColorChange,
  invalidateLayerBakeFingerprint,
} from "./app/pipeline.js";
import { clipUniforms, initWebglFallback, ensureSceneGpuUpload, warmClipGpuInit } from "./app/webglFallback.js";
import { initPresentation, resize, bindHudText, syncBoundsSlider } from "./app/presentation.js";
import { hudText, copyMetricsToClipboard } from "./app/hud.js";
import { initProdSettingsUi } from "./app/quality.js";
import { applyBootPerfTier } from "./app/perfAdapt.js";
import { detectDeviceTier } from "./app/deviceTier.js";
import { cancelProgressiveFit } from "./app/progressiveFit.js";
import { startRenderLoop } from "./app/loop.js";
import { setPanelCollapsed } from "./app/panelLayout.js";
import { state } from "./app/state.js";
import { initWebMCP } from "./app/webmcp.js";
import { initWebmcpSetupDialog } from "./app/webmcpSetupDialog.js";
import { initSplash, markSplashSidebarReady, forceSplashDismiss } from "./app/splash.js";
import { initKeyframeLoadBar } from "./app/keyframeLoadBar.js";
import { initTearDebug } from "./app/tearDebug.js";
import { initIsoRefineDebug } from "./app/isoRefineDebug.js";
import { initStartupProfile, startupBegin, startupEnd, startupMark } from "./app/startupProfile.js";
import {
  initAutosave,
  persistNow,
  restoreAutosave,
  scheduleAutosave,
  setAutosaveError,
} from "./app/persistence/autosave.js";
import {
  applyExpressionsFromQuery,
  shareExpressionLink,
} from "./app/persistence/exprShare.js";
import { installOgCapture } from "./app/ogCapture.js";

initSplash();
initKeyframeLoadBar();
initTearDebug();
initIsoRefineDebug();
initStartupProfile();
initTheme();
initDom();
state.deviceTier = detectDeviceTier();
initProdSettingsUi();
initWebglFallback();
bindClipUniforms(clipUniforms);
initScene();
initPresentation();
bindHudText(hudText);
initKeyframeHandler();

state.latexChangeInvalidators.push(invalidateLayerBakeFingerprint);

function collapseToViewport() {
  setPanelCollapsed(true);
  refreshPanelToggleChrome();
  runPanelTransition(resize);
  resize();
  state.exprListApi?.render();
}

function onPanelDismissSettled(collapsed: boolean) {
  refreshPanelToggleChrome();
  resize();
  if (collapsed) state.exprListApi?.render();
}

state.exprListApi = mountExprList({
  root: els.exprList,
  footerRoot: els.exprFooter,
  onCollapsePanel: collapseToViewport,
  onExprChange: () => {
    syncExprCompileState();
    if (anyParamAnimating()) {
      cancelProgressiveFit();
      scheduleUploadFit(0, { fromAnim: false });
    } else {
      scheduleUploadFit();
    }
    scheduleAutosave();
  },
  onParamChange: () => {
    syncExprCompileState();
    const animating = anyParamAnimating();
    scheduleUploadFit(animating ? 0 : 80, { fromAnim: animating });
    scheduleAutosave();
  },
  onColorChange: () => {
    handleColorChange();
    scheduleAutosave();
  },
  onStructuralChange: () => {
    cancelProgressiveFit();
    scheduleUploadFit(0, { fromAnim: false });
    scheduleAutosave();
  },
});

async function bootstrap() {
  startupBegin("boot.bootstrap");
  wirePipelineDom();
  initPanelResize(resize);
  initPanelToggle(resize);
  initPanelCollapse(collapseToViewport);
  initPanelDismiss(onPanelDismissSettled, resize);
  els.clearExprs?.addEventListener("click", () => {
    state.exprListApi?.clearAll?.();
  });
  initAutosave();
  initProjectActions();

  setExpressionsOnChange(() => {
    scheduleAutosave();
  });

  els.preset.addEventListener("change", () => {
    applyPreset(els.preset.value);
    if (state.fitTimer) clearTimeout(state.fitTimer);
    uploadFit();
    scheduleAutosave();
  });

  els.reset.addEventListener("click", () => {
    resetCameraView();
    scheduleAutosave();
  });

  els.copyMetrics?.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    void copyMetricsToClipboard();
  });

  initWebmcpSetupDialog();
  void initWebMCP();

  if (els.hud) els.hud.textContent = "clip-grid · idct volume";

  startupMark("boot.before-render-loop");
  warmClipGpuInit("boot-early");
  startRenderLoop();

  let restored = false;
  try {
    restored = await applyExpressionsFromQuery(location.search);
    if (restored) {
      syncBoundsSlider();
      syncExprCompileState();
      await persistNow();
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setAutosaveError(`Could not load shared expressions: ${msg}`);
  }
  if (!restored) {
    startupBegin("boot.restore-autosave");
    restored = await restoreAutosave();
    startupEnd("boot.restore-autosave", { restored });
  }

  if (!restored) {
    startupBegin("boot.init-compile");
    initCompile();
    startupEnd("boot.init-compile");
  }
  applyBootPerfTier(restored);
  refreshPanelToggleChrome();
  ensureParamAnimationFromExprs();
  state.exprListApi?.render();
  markSplashSidebarReady();

  const errReport = getExpressionErrorReport();
  if (errReport && (errReport.errorCount > 0 || errReport.globalError)) {
    forceSplashDismiss("compile-errors");
  }

  startupMark("boot.schedule-first-uploadFit", { animating: anyParamAnimating() });
  window.setTimeout(() => {
    uploadFit({ fromAnim: anyParamAnimating() });
  }, 0);
  startupEnd("boot.bootstrap");
  if (new URLSearchParams(location.search).has("ogCapture")) {
    installOgCapture();
  }
}

function flashShareFeedback(btn: HTMLButtonElement, message: string) {
  const prevTip = btn.dataset.tooltip ?? "Share link";
  const prevLabel = btn.getAttribute("aria-label") ?? "Share link";
  btn.dataset.tooltip = message;
  btn.setAttribute("aria-label", message);
  window.setTimeout(() => {
    btn.dataset.tooltip = prevTip;
    btn.setAttribute("aria-label", prevLabel);
  }, 1600);
}

function initProjectActions() {
  els.shareLink?.addEventListener("click", () => {
    void shareExpressionLink().then((result) => {
      const btn = els.shareLink;
      if (!btn) return;
      if (result === "shared") flashShareFeedback(btn, "Shared");
      else if (result === "copied") flashShareFeedback(btn, "Link copied");
      else setAutosaveError("Could not share link");
    });
  });
}

void bootstrap();
