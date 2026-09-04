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
  handleVisibilityChange,
  invalidateLayerBakeFingerprint,
} from "./app/pipeline.js";
import { clipUniforms, initWebglFallback, ensureSceneGpuUpload, warmClipGpuInit } from "./app/webglFallback.js";
import { initPresentation, resize, bindHudText, syncBoundsSlider } from "./app/presentation.js";
import { hudText, copyMetricsToClipboard } from "./app/hud.js";
import { initProdSettingsUi } from "./app/quality.js";
import { applyBootPerfTier, initAutoQualityAdaptUi } from "./app/perfAdapt.js";
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
  applyExpressionsFromFragment,
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
  onVisibilityChange: () => {
    handleVisibilityChange();
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
      const url = new URL(location.href);
      if (url.searchParams.has("e")) {
        url.searchParams.delete("e");
        history.replaceState(null, "", url.pathname + url.hash);
      }
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    setAutosaveError(`Could not load shared expressions: ${msg}`);
  }
  if (!restored) {
    try {
      restored = await applyExpressionsFromFragment(location.hash);
      if (restored) {
        syncBoundsSlider();
        syncExprCompileState();
        await persistNow();
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setAutosaveError(`Could not load shared expressions: ${msg}`);
    }
  }
  if (!restored) {
    startupBegin("boot.restore-autosave");
    try {
      restored = await restoreAutosave();
    } catch (e) {
      // Safari/WebKit's IndexedDB backing store can transiently fail to open
      // right after a full reload ("Unable to open database file on disk").
      // Don't let that abort the rest of boot — fall through to a fresh
      // document instead of leaving the app half-initialized.
      const msg = e instanceof Error ? e.message : String(e);
      setAutosaveError(`Could not restore autosave: ${msg}`);
    }
    startupEnd("boot.restore-autosave", { restored });
  }

  if (!restored) {
    startupBegin("boot.init-compile");
    initCompile();
    startupEnd("boot.init-compile");
  }
  applyBootPerfTier(restored);
  initAutoQualityAdaptUi();
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

const SHARE_LINK_LABEL = "Share link";

function flashShareFeedback(btn: HTMLButtonElement, message: string) {
  btn.classList.add("is-done");
  btn.dataset.tooltip = message;
  btn.setAttribute("aria-label", message);
  window.setTimeout(() => {
    btn.classList.remove("is-done");
    btn.dataset.tooltip = SHARE_LINK_LABEL;
    btn.setAttribute("aria-label", SHARE_LINK_LABEL);
  }, 1600);
}

function setShareBusy(btn: HTMLButtonElement, busy: boolean) {
  btn.classList.toggle("is-busy", busy);
  btn.disabled = busy;
  const label = busy ? "Sharing…" : SHARE_LINK_LABEL;
  btn.dataset.tooltip = label;
  btn.setAttribute("aria-label", label);
}

/** Drop an optimistic checkmark that never got confirmed by a real result (cancel/failure). */
function clearShareDone(btn: HTMLButtonElement) {
  btn.classList.remove("is-done");
  btn.dataset.tooltip = SHARE_LINK_LABEL;
  btn.setAttribute("aria-label", SHARE_LINK_LABEL);
}

function initProjectActions() {
  els.shareLink?.addEventListener("click", () => {
    const btn = els.shareLink;
    if (!btn || btn.classList.contains("is-busy")) return;
    setShareBusy(btn, true);
    // The checkmark shows the moment rendering settles — it doesn't wait on
    // the native share sheet, which can block on user input for an
    // arbitrary amount of time afterward. But its *label* stays generic
    // until the actual copy/share result comes in, since we don't yet know
    // which (or whether it'll fail/get cancelled) at render time.
    void shareExpressionLink(() => {
      setShareBusy(btn, false);
      btn.classList.add("is-done");
    })
      .then((result) => {
        if (result === "shared") flashShareFeedback(btn, "Shared");
        else if (result === "copied") flashShareFeedback(btn, "Copied to clipboard");
        else {
          // "cancelled": user dismissed the share sheet — no error.
          // "failed": a real failure — drop the premature checkmark too.
          clearShareDone(btn);
          if (result === "failed") setAutosaveError("Could not share link");
        }
      })
      .catch(() => {
        setShareBusy(btn, false);
        setAutosaveError("Could not share link");
      });
  });
}

void bootstrap();
