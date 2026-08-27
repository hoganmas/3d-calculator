import { mountLiquidThumb, syncLiquidThumb } from "../ui/liquidSlider.js";
import {
  setClipGpuCanvasVisible,
  resizeClipGpuCanvas,
  hasUploadedVolume,
  resetClipGpuProfile,
  isClipBakeGpuReady,
} from "../render/webgpu/march.js";
import { els, viewportSize } from "./dom.js";
import { state, MARCH_DOWNSCALE_MIN, MARCH_DOWNSCALE_MAX, MARCH_DOWNSCALE_LABELS } from "./state.js";
import {
  renderer,
  labelRenderer,
  camera,
  worldGrid,
  worldLabels,
  boxHelper,
} from "./scene.js";
import { clipQuad, useGpuClipPath } from "./webglFallback.js";

function marchDownscaleTickPct(n) {
  return ((n - MARCH_DOWNSCALE_MIN) / (MARCH_DOWNSCALE_MAX - MARCH_DOWNSCALE_MIN)) * 100;
}

function initMarchSliderUi() {
  const ticks = document.getElementById("marchSliderTicks");
  const labels = document.getElementById("marchSliderLabels");
  if (!ticks || !labels) return;
  ticks.replaceChildren();
  labels.replaceChildren();
  for (let n = MARCH_DOWNSCALE_MIN; n <= MARCH_DOWNSCALE_MAX; n++) {
    const pct = `${marchDownscaleTickPct(n)}%`;
    const tick = document.createElement("span");
    tick.style.setProperty("--tick", pct);
    ticks.appendChild(tick);
    if (MARCH_DOWNSCALE_LABELS.has(n)) {
      const lab = document.createElement("span");
      lab.style.setProperty("--tick", pct);
      lab.textContent = `${n}×`;
      labels.appendChild(lab);
    }
  }
}

function readMarchDownscale() {
  if (!els.marchDownscale) return 1;
  const n = Math.round(Number(els.marchDownscale.value) || 1);
  return Math.min(MARCH_DOWNSCALE_MAX, Math.max(MARCH_DOWNSCALE_MIN, n));
}

export function syncMarchSlider() {
  const n = readMarchDownscale();
  if (els.marchDownscale) els.marchDownscale.value = String(n);
  if (els.marchScaleLabel) els.marchScaleLabel.textContent = `${n}×`;
  syncLiquidThumb(els.marchDownscale);
  return n;
}

export function marchDownscale() {
  return readMarchDownscale();
}

/** CSS px covered by the floating sidebar (0 on narrow layouts). */
export function compositionCoveredWidth(vw) {
  if (typeof window !== "undefined" && window.matchMedia("(max-width: 800px)").matches) {
    return 0;
  }
  const cs = getComputedStyle(document.documentElement);
  const inset = parseFloat(cs.getPropertyValue("--panel-inset"));
  const pw = parseFloat(cs.getPropertyValue("--panel-w"));
  const covered = (Number.isFinite(inset) ? inset : 12) + (Number.isFinite(pw) ? pw : 360);
  return Math.min(Math.max(0, covered), Math.max(0, vw - 160));
}

/** NDC x of the free-region center (0 when the panel does not inset composition). */
export function compositionNdcOffsetX(vw) {
  const covered = compositionCoveredWidth(vw);
  if (covered <= 1 || vw <= covered + 40) return 0;
  return covered / vw;
}

export function applyCameraComposition(vw, vh) {
  // Keep projection in sync with offsetDirMatrix used by volume rays:
  // rays aim forward at NDC x = +offset (free-region center to the right of the panel),
  // so world points on the view axis must project to that same NDC x.
  if (typeof camera.clearViewOffset === "function") camera.clearViewOffset();
  camera.aspect = vw / Math.max(vh, 1);
  camera.updateProjectionMatrix();
  const o = compositionNdcOffsetX(vw);
  if (Math.abs(o) > 1e-12) {
    const e = camera.projectionMatrix.elements;
    // Left-multiply by translate(x' = x + o*w): column c, row 0 += o * row 3.
    for (let c = 0; c < 4; c++) {
      e[c * 4 + 0] += o * e[c * 4 + 3];
    }
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
}

/** Raymarch internal resolution scale (display canvas stays full viewport). */
export function marchResolutionScale() {
  return 1 / marchDownscale();
}

export function marchFramebufferSize() {
  const { vw, vh } = viewportSize();
  const s = marchResolutionScale();
  return {
    mw: Math.max(1, Math.round(vw * s)),
    mh: Math.max(1, Math.round(vh * s)),
  };
}

/** @type {(() => string) | null} */
let getHudText = null;

export function bindHudText(getter) {
  getHudText = getter;
}

function applyDisplaySize(rw, rh, vw, vh, { markClipDirty = true } = {}) {
  applyCameraComposition(vw, vh);
  renderer.setSize(rw, rh, false);
  const canvas = renderer.domElement;
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.position = "absolute";
  canvas.style.inset = "0";
  labelRenderer.setSize(rw, rh, false);
  const labelCanvas = labelRenderer.domElement;
  labelCanvas.style.width = "100%";
  labelCanvas.style.height = "100%";
  if (markClipDirty) state.clipDirty = true;
  if (els.hud && getHudText) els.hud.textContent = getHudText();
}

export function resize() {
  const { vw, vh } = viewportSize();
  applyDisplaySize(vw, vh, vw, vh, { markClipDirty: true });
  // Present canvas at display resolution; raymarch targets stay at mw×mh.
  resizeClipGpuCanvas(vw, vh);
}

export function syncClipPresentation() {
  const hasVolume = hasUploadedVolume() || Boolean(
    state.lastSceneBake && (state.lastSceneBake.densLayers.length || state.lastSceneBake.constraints.length),
  );
  const gpu = useGpuClipPath() && hasVolume;
  clipQuad.visible = !gpu && hasVolume && Boolean(state.worldCheb);
  // Grid/box/labels depth-test on the WebGPU overlay; WebGL fallback keeps
  // labels on a higher canvas (isos don't write depth there).
  worldGrid.visible = !gpu;
  worldLabels.visible = !gpu;
  boxHelper.visible = !gpu;
  labelRenderer.domElement.style.visibility = gpu ? "hidden" : "visible";
  setClipGpuCanvasVisible(isClipBakeGpuReady());
}

export function markMarchDirty() {
  syncMarchSlider();
  resetClipGpuProfile();
  state.clipDirty = true;
}

export function initPresentation() {
  initMarchSliderUi();
  if (els.marchDownscale) {
    const wrap = els.marchDownscale.closest(".march-liquid-track") || els.marchDownscale.closest(".march-slider-wrap");
    if (wrap instanceof HTMLElement) mountLiquidThumb(wrap, els.marchDownscale);
  }
  syncMarchSlider();

  els.marchDownscale?.addEventListener("input", markMarchDirty);
  els.marchDownscale?.addEventListener("change", markMarchDirty);

  window.addEventListener("resize", resize);
  resize();
}
