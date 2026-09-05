import { mountLiquidThumb, mountStaticLiquidThumb, syncLiquidThumb } from "../ui/liquidSlider.js";
import { attachInfoContextMenu } from "../ui/infoMenu.js";
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
  isIsometric,
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

/** Toggle switches snap between two fixed states — no pointer-follow drag. */
function mountSettingsToggleThumb(input: HTMLInputElement | null | undefined) {
  if (!input) return;
  const wrap = input.closest(".settings-liquid-track");
  if (wrap instanceof HTMLElement) mountStaticLiquidThumb(wrap, input);
}

const QUALITY_INFO: [HTMLInputElement | null | undefined, string][] = [
  [
    els.precisionQuality,
    "Polynomial fit degree for expression fields. Higher values resolve sharper detail at a higher compute cost.",
  ],
  [
    els.scalarQuality,
    "Resolution and step count for scalar volume (Beer) ray-marching.",
  ],
  [
    els.surfaceQuality,
    "Resolution and step count for iso-surface ray-marching.",
  ],
  [
    els.vectorQuality,
    "Particle count for flow / vector field animation.",
  ],
  [
    els.autoQualityAdapt,
    "Lowers quality automatically if frame rate drops.",
  ],
];

function initSettingsInfoMenus() {
  for (const [input, text] of QUALITY_INFO) {
    if (!input) continue;
    const row = input.closest("label");
    attachInfoContextMenu(row instanceof HTMLElement ? row : input, text);
  }
}

export function syncToggleSwitchColor(input: HTMLInputElement | null | undefined) {
  if (!input) return;
  input.classList.toggle("is-on", input.value === "1");
}

function initQualityToggleSwitch() {
  const input = els.autoQualityAdapt;
  if (!input) return;
  syncToggleSwitchColor(input);
  input.addEventListener("input", () => syncToggleSwitchColor(input));

  // A 2-state switch shouldn't need a drag to land on a position — treat any
  // click/tap as a flip instead of the native range "jump to click point".
  input.addEventListener("pointerdown", (ev) => ev.preventDefault());
  input.addEventListener("click", () => {
    input.value = input.value === "1" ? "0" : "1";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    input.focus({ preventScroll: true });
  });
}

function initSettingsLiquidSliders() {
  mountSettingsLiquidSlider(els.precisionQuality);
  mountSettingsLiquidSlider(els.boxSize);
  mountSettingsLiquidSlider(els.scalarQuality);
  mountSettingsLiquidSlider(els.surfaceQuality);
  mountSettingsLiquidSlider(els.vectorQuality);
  mountSettingsToggleThumb(els.autoQualityAdapt);
  syncBoundsSlider();
  initSettingsInfoMenus();
  initQualityToggleSwitch();
}

/**
 * The sidebar footer floats over the expression list (see `.panel-status` in
 * app.css) so the list can scroll beneath its translucent background. Its
 * height varies (a wrapped compile error grows it), so keep `--panel-status-h`
 * in sync — `#exprList`'s bottom padding reads it to let the last expression
 * still scroll fully clear of the overlay.
 */
function initPanelStatusOverlay() {
  const status = document.getElementById("panelStatus");
  if (!status) return;
  const sync = () => {
    const h = Math.ceil(status.getBoundingClientRect().height);
    if (h > 0) document.documentElement.style.setProperty("--panel-status-h", `${h}px`);
  };
  sync();
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(sync).observe(status);
}

export function syncSettingsLiquidThumbs() {
  syncLiquidThumb(els.precisionQuality);
  syncLiquidThumb(els.boxSize);
  syncLiquidThumb(els.scalarQuality);
  syncLiquidThumb(els.surfaceQuality);
  syncLiquidThumb(els.vectorQuality);
  syncLiquidThumb(els.autoQualityAdapt);
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

export function syncIsometricUi() {
  const btn = els.toggleIsometric;
  if (!btn) return;
  const on = isIsometric();
  const label = on ? "Switch to perspective view" : "Switch to isometric view";
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

/** Slider value, with any device-tier compose floor (GPU + HUD). */
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

/**
 * @param liveGpuPath When supplied (the render loop passes its already-computed
 * per-frame `useGpuClipPath()` result), the WebGL fallback's visibility flags
 * are derived from that exact value instead of a fresh call — so they can
 * never lag a frame behind the live check `drawClipGpuFrame` uses to decide
 * whether to draw the WebGPU grid/label overlay. A stale cached `false` here
 * while the WebGPU path was already live drawing its own grid+labels was
 * causing both to render at once (doubled axes/gridlines on mobile, where
 * the async pipeline-ready → resync events land less predictably).
 */
export function syncClipPresentation(liveGpuPath?: boolean) {
  const hasVolume = hasUploadedVolume() || Boolean(
    state.lastSceneBake &&
      (state.lastSceneBake.cloudLayers.length ||
        state.lastSceneBake.isosurfaceLayers.length ||
        (state.lastSceneBake.flowLayers?.length ?? 0) > 0),
  );
  const gpu = (liveGpuPath ?? useGpuClipPath()) && hasVolume;
  clipQuad.visible = !gpu && hasVolume && Boolean(state.worldCheb);
  // Grid/box/labels depth-test on the WebGPU overlay; WebGL fallback keeps
  // labels on a higher canvas (isos don't write depth there).
  const showGrid = state.showGridAxes;
  worldGrid.visible = !gpu && showGrid;
  worldLabels.visible = !gpu && showGrid;
  boxHelper.visible = !gpu;
  // `display:none` (not `visibility:hidden`) — the label canvas sits at a
  // higher z-index than the WebGPU canvas underneath it, and mobile Safari
  // has been known to keep compositing a `visibility:hidden` WebGL canvas
  // for a frame or two after it stops rendering, ghosting stale axis labels
  // over the live WebGPU-drawn ones. `display:none` fully removes it from
  // paint instead of relying on the compositor to honor visibility.
  labelRenderer.domElement.style.display = gpu ? "none" : "block";
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
  initPanelStatusOverlay();
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
