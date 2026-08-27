import "mathlive";
import "mathlive/static.css";
import "./ui/theme.css";
import { initTheme } from "./ui/theme.js";
import { mountExprList } from "./ui/expr-sidebar/mount.js";
import { setExpressionsOnChange } from "./model/expressions.js";
import { anyParamAnimating } from "./model/params.js";
import { syncExprCompileState } from "./app/hud.js";
import { initDom, els, initPanelResize } from "./app/dom.js";
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

initTheme();
initDom();
initWebglFallback();
bindClipUniforms(clipUniforms);
initScene();
initPresentation();
bindHudText(hudText);
initCompile();
initKeyframeHandler();
wirePipelineDom();
initPanelResize(resize);

els.preset.addEventListener("change", () => {
  applyPreset(els.preset.value);
  if (state.fitTimer) clearTimeout(state.fitTimer);
  uploadFit();
});

els.reset.addEventListener("click", () => {
  camera.position.set(3.2, 2.4, 4.2);
  controls.target.set(0, 0, 0);
  controls.update();
  state.clipDirty = true;
});

els.copyMetrics?.addEventListener("click", (ev) => {
  ev.preventDefault();
  ev.stopPropagation();
  void copyMetricsToClipboard();
});

state.exprListApi = mountExprList({
  root: els.exprList,
  onExprChange: () => {
    syncExprCompileState();
    scheduleUploadFit();
  },
  onParamChange: () => {
    syncExprCompileState();
    // Play starts animation: bake keyframes immediately (fromAnim).
    // Avoid a non-anim fit that would clear the keyframe cache first.
    const animating = anyParamAnimating();
    scheduleUploadFit(animating ? 0 : 80, { fromAnim: animating });
  },
  onColorChange: () => {
    handleColorChange();
  },
  onStructuralChange: () => {
    scheduleUploadFit(0);
  },
});
state.exprListApi.render();
setExpressionsOnChange(() => {
  /* list mutations already call render from UI helpers */
});

if (els.hud) els.hud.textContent = "clip-grid · idct volume";
uploadFit();
startRenderLoop();
