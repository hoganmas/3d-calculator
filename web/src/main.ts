import "mathlive";
import "mathlive/static.css";
import "./ui/theme.css";
import { initTheme } from "./ui/theme.js";
import { mountExprList } from "./ui/expr-sidebar/mount.js";
import { setExpressionsOnChange } from "./model/expressions.js";
import { anyParamAnimating } from "./model/params.js";
import { syncExprCompileState } from "./app/hud.js";
import { initDom, els, initPanelResize, initPanelToggle } from "./app/dom.js";
import { initScene, bindClipUniforms, controls, camera } from "./app/scene.js";
import { initCompile, applyPreset } from "./app/compile.js";
import {
  scheduleUploadFit,
  uploadFit,
  initKeyframeHandler,
  wirePipelineDom,
  handleColorChange,
} from "./app/pipeline.js";
import { clipUniforms, initWebglFallback } from "./app/webglFallback.js";
import { initPresentation, resize, bindHudText } from "./app/presentation.js";
import { hudText, copyMetricsToClipboard } from "./app/hud.js";
import { startRenderLoop } from "./app/loop.js";
import { state } from "./app/state.js";
import { initWebMCP } from "./app/webmcp.js";
import { initWebmcpSetupDialog } from "./app/webmcpSetupDialog.js";
import { initSplash, markSplashSidebarReady } from "./app/splash.js";
import {
  initAutosave,
  restoreAutosave,
  scheduleAutosave,
} from "./app/persistence/autosave.js";
import {
  downloadDocument,
  exportCurrentDocument,
  importDocumentFromFile,
  shareDocument,
  canShareFiles,
} from "./app/persistence/files.js";

initSplash();
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
  const restored = await restoreAutosave();
  if (!restored) initCompile();
  state.exprListApi?.render();
  requestAnimationFrame(() => markSplashSidebarReady());

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
    camera.position.set(5.2, 4.0, 6.8);
    controls.target.set(0, 0, 0);
    controls.update();
    state.clipDirty = true;
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
  uploadFit();
  startRenderLoop();
}

function initProjectActions() {
  els.downloadScene?.addEventListener("click", () => {
    downloadDocument(exportCurrentDocument());
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
      if (els.autosaveStatus) {
        els.autosaveStatus.textContent = msg;
        els.autosaveStatus.dataset.state = "error";
      }
    }
  });

  els.openScene?.addEventListener("click", () => {
    els.sceneFileInput?.click();
  });
}

void bootstrap();
