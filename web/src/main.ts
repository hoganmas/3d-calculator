import "mathlive";
import "mathlive/static.css";
import "./ui/theme.css";
import { initTheme } from "./ui/theme.js";
import { mountExprList } from "./ui/expr-sidebar/mount.js";
import { setExpressionsOnChange } from "./model/expressions.js";
import { anyParamAnimating, ensureParamAnimationFromExprs } from "./model/params.js";
import { syncExprCompileState, getExpressionErrorReport } from "./app/hud.js";
import { initDom, els, initPanelResize, initPanelToggle } from "./app/dom.js";
import { initScene, bindClipUniforms, resetCameraView } from "./app/scene.js";
import { initCompile, applyPreset } from "./app/compile.js";
import {
  scheduleUploadFit,
  uploadFit,
  initKeyframeHandler,
  wirePipelineDom,
  handleColorChange,
} from "./app/pipeline.js";
import { clipUniforms, initWebglFallback, ensureSceneGpuUpload, warmClipGpuInit } from "./app/webglFallback.js";
import { initPresentation, resize, bindHudText } from "./app/presentation.js";
import { hudText, copyMetricsToClipboard } from "./app/hud.js";
import { startRenderLoop } from "./app/loop.js";
import { state } from "./app/state.js";
import { initWebMCP } from "./app/webmcp.js";
import { initWebmcpSetupDialog } from "./app/webmcpSetupDialog.js";
import { initSplash, markSplashSidebarReady, forceSplashDismiss } from "./app/splash.js";
import { initKeyframeLoadBar } from "./app/keyframeLoadBar.js";
import { initTearDebug } from "./app/tearDebug.js";
import { initStartupProfile, startupBegin, startupEnd, startupMark } from "./app/startupProfile.js";
import {
  initAutosave,
  persistNow,
  restoreAutosave,
  scheduleAutosave,
  setAutosaveError,
} from "./app/persistence/autosave.js";
import {
  downloadDocument,
  exportCurrentDocument,
  importDocumentFromFile,
  shareDocument,
  canShareFiles,
} from "./app/persistence/files.js";
import {
  applyExpressionsFromFragment,
  copyExpressionShareLink,
} from "./app/persistence/exprShare.js";

initSplash();
initKeyframeLoadBar();
initTearDebug();
initStartupProfile();
initTheme();
initDom();
initWebglFallback();
bindClipUniforms(clipUniforms);
initScene();
initPresentation();
bindHudText(hudText);
initKeyframeHandler();

state.exprListApi = mountExprList({
  root: els.exprList,
  onExprChange: () => {
    syncExprCompileState();
    scheduleUploadFit();
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
    scheduleUploadFit(0);
    scheduleAutosave();
  },
});

async function bootstrap() {
  startupBegin("boot.bootstrap");
  wirePipelineDom();
  initPanelResize(resize);
  initPanelToggle(resize);
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
    restored = await applyExpressionsFromFragment(location.hash);
    if (restored) {
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
}

function initProjectActions() {
  els.downloadScene?.addEventListener("click", () => {
    downloadDocument(exportCurrentDocument());
  });

  els.copyExprLink?.addEventListener("click", () => {
    void copyExpressionShareLink(els.copyExprLink ?? undefined).catch((e) => {
      const msg = e instanceof Error ? e.message : String(e);
      setAutosaveError(msg);
    });
  });

  els.shareScene?.addEventListener("click", async () => {
    try {
      const shared = await shareDocument(exportCurrentDocument());
      if (!shared) downloadDocument(exportCurrentDocument());
    } catch {
      downloadDocument(exportCurrentDocument());
    }
  });

  if (els.shareScene && !canShareFiles()) {
    els.shareScene.hidden = true;
  }

  els.sceneFileInput?.addEventListener("change", async () => {
    const input = els.sceneFileInput!;
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    try {
      if (state.fitTimer) clearTimeout(state.fitTimer);
      await importDocumentFromFile(file);
      uploadFit();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setAutosaveError(msg);
    }
  });

  els.openScene?.addEventListener("click", () => {
    els.sceneFileInput?.click();
  });
}

void bootstrap();
