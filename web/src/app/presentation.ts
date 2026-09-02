import { mountLiquidThumb, syncLiquidThumb } from "../ui/liquidSlider.js";
import {
  setClipGpuCanvasVisible,
  resizeClipGpuCanvas,
  hasUploadedVolume,
  resetClipGpuProfile,
  isClipBakeGpuReady,
} from "../render/webgpu/march.js";
import { els, viewportSize } from "./dom.js";
import { state, MARCH_DOWNSCALE_MIN, MARCH_DOWNSCALE_MAX, MARCH_DOWNSCALE_LABELS, BOUNDS_SIZE_MIN, BOUNDS_SIZE_MAX } from "./state.js";
import { ISO_COARSE_DOWNSCALE } from "./qualityMapping.js";
import { effectiveIsoComposeDownscale } from "./deviceTier.js";
import {
  isHorizontalPanelLayout,
  readPanelCoverHeight,
  readPanelCoverWidth,
  readPanelInset,
} from "./panelLayout.js";
import {
  renderer,
  labelRenderer,
  camera,
  worldGrid,
  worldLabels,
  boxHelper,
} from "./scene.js";
import { clipQuad, useGpuClipPath } from "./webglFallback.js";
import { initKeyboardInsets, VIEWPORT_SYNC_EVENT } from "./keyboardInsets.js";

function formatBoundsSize(n: number) {
  const rounded = Math.round(n * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

export function readBoundsSize() {
  const n = Number(els.boxSize.value) || 5;
  return Math.min(BOUNDS_SIZE_MAX, Math.max(BOUNDS_SIZE_MIN, Math.round(n * 10) / 10));
}

export function syncBoundsSlider() {
  const n = readBoundsSize();
  els.boxSize.value = String(n);
  if (els.boundsSizeLabel) els.boundsSizeLabel.textContent = formatBoundsSize(n);
  syncLiquidThumb(els.boxSize);
  return n;
}

function mountSettingsLiquidSlider(input: HTMLInputElement | null | undefined) {
  if (!input) return;
  const wrap =
    input.closest(".settings-liquid-track") ||
    input.closest(".march-liquid-track") ||
    input.closest(".march-slider-wrap");
  if (wrap instanceof HTMLElement) mountLiquidThumb(wrap, input);
}

function initSettingsLiquidSliders() {
  mountSettingsLiquidSlider(els.precisionQuality);
  mountSettingsLiquidSlider(els.boxSize);
  mountSettingsLiquidSlider(els.scalarQuality);
  mountSettingsLiquidSlider(els.surfaceQuality);
  mountSettingsLiquidSlider(els.vectorQuality);
  syncBoundsSlider();
}

export function syncSettingsLiquidThumbs() {
  syncLiquidThumb(els.precisionQuality);
  syncLiquidThumb(els.boxSize);
  syncLiquidThumb(els.scalarQuality);
  syncLiquidThumb(els.surfaceQuality);
  syncLiquidThumb(els.vectorQuality);
}

function marchDownscaleTickPct(n: number) {
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

function readIsoMarchDownscale() {
  if (!els.isoMarchDownscale) return readMarchDownscale();
  const n = Math.round(Number(els.isoMarchDownscale.value) || 1);
  return Math.min(MARCH_DOWNSCALE_MAX, Math.max(MARCH_DOWNSCALE_MIN, n));
}

export function syncMarchSlider() {
  const n = readMarchDownscale();
  if (els.marchDownscale) els.marchDownscale.value = String(n);
  if (els.marchScaleLabel) els.marchScaleLabel.textContent = `${n}×`;
  syncLiquidThumb(els.marchDownscale);
  const isoN = readIsoMarchDownscale();
  if (els.isoMarchDownscale) els.isoMarchDownscale.value = String(isoN);
  if (els.isoMarchScaleLabel) els.isoMarchScaleLabel.textContent = `${isoN}×`;
  syncLiquidThumb(els.isoMarchDownscale);
  return n;
}

export function syncShowGridAxesUi() {
  const btn = els.toggleGridAxes;
  if (!btn) return;
  const on = state.showGridAxes;
  const label = on ? "Hide grid & axes" : "Show grid & axes";
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  btn.setAttribute("aria-label", label);
  btn.dataset.tooltip = label;
}

/** Volume (Beer) march downscale. */
export function marchDownscale() {
  return readMarchDownscale();
}

/** Iso-surface occupancy (coarse) downscale — always 16×. */
export function isoCoarseDownscale() {
  return ISO_COARSE_DOWNSCALE;
}

/** Iso-surface compose (finest) downscale — surface quality / iso slider. */
export function isoMarchDownscale() {
  return readIsoMarchDownscale();
}

/** Slider value raised to the device-tier compose floor (GPU + HUD). */
export function effectiveIsoMarchDownscale() {
  return effectiveIsoComposeDownscale(readIsoMarchDownscale(), state.deviceTier);
}

/** CSS px covered by the floating sidebar along the dock axis (0 on narrow full-width layouts). */
export function compositionCoveredWidth(vw: number) {
  if (isHorizontalPanelLayout()) return 0;
  const covered = readPanelInset() + readPanelCoverWidth();
  return Math.min(Math.max(0, covered), Math.max(0, vw - 160));
}

/** CSS px covered by a top-docked panel strip (0 in vertical layout). */
export function compositionCoveredHeight(vh: number) {
  if (!isHorizontalPanelLayout()) return 0;
  const covered = readPanelInset() + readPanelCoverHeight();
  return Math.min(Math.max(0, covered), Math.max(0, vh - 160));
}

/** NDC x of the free-region center (0 when the panel does not inset composition). */
export function compositionNdcOffsetX(vw: number) {
  const covered = compositionCoveredWidth(vw);
  if (covered <= 1 || vw <= covered + 40) return 0;
  return covered / vw;
}

/** NDC y of the free-region center (0 when the panel does not inset composition). */
export function compositionNdcOffsetY(vh: number) {
  const covered = compositionCoveredHeight(vh);
  if (covered <= 1 || vh <= covered + 40) return 0;
  // Top panel → free region sits below viewport center → negative NDC y.
  return -covered / vh;
}

export function applyCameraComposition(vw: number, vh: number) {
  // Keep projection in sync with offsetDirMatrix used by volume rays:
  // rays aim forward at the free-region center (right of a left panel, or below a top panel).
  if (typeof camera.clearViewOffset === "function") camera.clearViewOffset();
  camera.aspect = vw / Math.max(vh, 1);
  camera.updateProjectionMatrix();
  const ox = compositionNdcOffsetX(vw);
  const oy = compositionNdcOffsetY(vh);
  if (Math.abs(ox) > 1e-12 || Math.abs(oy) > 1e-12) {
    const e = camera.projectionMatrix.elements;
    if (Math.abs(ox) > 1e-12) {
      for (let c = 0; c < 4; c++) e[c * 4 + 0] += ox * e[c * 4 + 3];
    }
    if (Math.abs(oy) > 1e-12) {
      for (let c = 0; c < 4; c++) e[c * 4 + 1] += oy * e[c * 4 + 3];
    }
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
}

/** Coarse iso occupancy resolution. Display canvas stays full viewport. */
export function marchResolutionScale() {
  return 1 / isoCoarseDownscale();
}

export function volumeResolutionScale() {
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

export function volumeFramebufferSize() {
  const { vw, vh } = viewportSize();
  const s = volumeResolutionScale();
  return {
    mw: Math.max(1, Math.round(vw * s)),
    mh: Math.max(1, Math.round(vh * s)),
  };
}

let getHudText: (() => string) | null = null;

export function bindHudText(getter: () => string) {
  getHudText = getter;
}

function applyDisplaySize(
  rw: number,
  rh: number,
  vw: number,
  vh: number,
  { markClipDirty = true }: { markClipDirty?: boolean } = {},
) {
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

let lastDisplayW = 0;
let lastDisplayH = 0;

export function resize() {
  const { vw, vh } = viewportSize();
  // iOS visualViewport scroll/URL-bar animation fires resize constantly.
  // Rebuilding the march targets on a no-op size change makes the coarse
  // grid crawl, which shows up as cyan flicker along the box on mobile.
  if (vw === lastDisplayW && vh === lastDisplayH) return;
  lastDisplayW = vw;
  lastDisplayH = vh;
  applyDisplaySize(vw, vh, vw, vh, { markClipDirty: true });
  // Present canvas at display resolution; raymarch targets stay at mw×mh.
  resizeClipGpuCanvas(vw, vh);
}

export function syncClipPresentation() {
  const hasVolume = hasUploadedVolume() || Boolean(
    state.lastSceneBake &&
      (state.lastSceneBake.cloudLayers.length ||
        state.lastSceneBake.isosurfaceLayers.length ||
        (state.lastSceneBake.flowLayers?.length ?? 0) > 0),
  );
  const gpu = useGpuClipPath() && hasVolume;
  clipQuad.visible = !gpu && hasVolume && Boolean(state.worldCheb);
  // Grid/box/labels depth-test on the WebGPU overlay; WebGL fallback keeps
  // labels on a higher canvas (isos don't write depth there).
  const showGrid = state.showGridAxes;
  worldGrid.visible = !gpu && showGrid;
  worldLabels.visible = !gpu && showGrid;
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
  initSettingsLiquidSliders();
  if (els.marchDownscale) {
    const wrap = els.marchDownscale.closest(".march-liquid-track") || els.marchDownscale.closest(".march-slider-wrap");
    if (wrap instanceof HTMLElement) mountLiquidThumb(wrap, els.marchDownscale);
  }
  if (els.isoMarchDownscale) {
    const wrap =
      els.isoMarchDownscale.closest(".march-liquid-track") ||
      els.isoMarchDownscale.closest(".march-slider-wrap");
    if (wrap instanceof HTMLElement) mountLiquidThumb(wrap, els.isoMarchDownscale);
  }
  syncMarchSlider();

  els.marchDownscale?.addEventListener("input", markMarchDirty);
  els.marchDownscale?.addEventListener("change", markMarchDirty);
  els.isoMarchDownscale?.addEventListener("input", markMarchDirty);
  els.isoMarchDownscale?.addEventListener("change", markMarchDirty);
  els.boxSize.addEventListener("input", syncBoundsSlider);
  els.boxSize.addEventListener("change", syncBoundsSlider);

  window.addEventListener("resize", resize);
  initVisualViewportResize();
  initKeyboardInsets();
  window.addEventListener(VIEWPORT_SYNC_EVENT, resize);
  resize();
}

function pinDocumentScroll() {
  if (window.scrollX !== 0 || window.scrollY !== 0) {
    window.scrollTo(0, 0);
  }
}

function initVisualViewportResize() {
  window.addEventListener("scroll", pinDocumentScroll, { passive: true });
  const vv = window.visualViewport;
  if (!vv) return;
  vv.addEventListener("resize", resize);
  vv.addEventListener("scroll", () => {
    // When the keyboard is open, visualViewport scroll drives layout — don't reset it.
    if (document.documentElement.dataset.keyboardOpen === "true") {
      resize();
      return;
    }
    pinDocumentScroll();
    resize();
  });
}
