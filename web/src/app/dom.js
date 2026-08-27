import { PRESETS } from "../math/fit.js";
import { getThemePref, setThemePref } from "../ui/theme.js";

export const els = {
  preset: document.getElementById("preset"),
  exprList: document.getElementById("exprList"),
  deg: document.getElementById("deg"),
  scale: document.getElementById("scale"),
  steps: document.getElementById("steps"),
  boxSize: document.getElementById("boxSize"),
  isoInterp: document.getElementById("isoInterp"),
  marchDownscale: document.getElementById("marchDownscale"),
  marchScaleLabel: document.getElementById("marchScaleLabel"),
  reset: document.getElementById("reset"),
  err: document.getElementById("err"),
  viewport: document.getElementById("viewport"),
  hud: document.getElementById("hud"),
  metricsDump: document.getElementById("metricsDump"),
  copyMetrics: document.getElementById("copyMetrics"),
  openSettings: document.getElementById("openSettings"),
  closeSettings: document.getElementById("closeSettings"),
  settingsDialog: document.getElementById("settingsDialog"),
  themePref: document.getElementById("themePref"),
};

export function viewportSize() {
  const vw = els.viewport.clientWidth;
  const vh = Math.max(els.viewport.clientHeight, 1);
  return { vw, vh };
}

export function initDom() {
  for (const [key, p] of Object.entries(PRESETS)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = p.label;
    els.preset.appendChild(opt);
  }

  if (els.themePref) {
    els.themePref.value = getThemePref();
    els.themePref.addEventListener("change", () => {
      setThemePref(/** @type {import("../ui/theme.js").ThemePref} */ (els.themePref.value));
    });
  }

  els.openSettings?.addEventListener("click", () => openSettingsDialog());
  els.closeSettings?.addEventListener("click", () => closeSettingsDialog());
  els.settingsDialog?.addEventListener("click", (ev) => {
    if (ev.target === els.settingsDialog) closeSettingsDialog();
  });
}

export function openSettingsDialog() {
  if (!els.settingsDialog || typeof els.settingsDialog.showModal !== "function") return;
  if (!els.settingsDialog.open) els.settingsDialog.showModal();
}

export function closeSettingsDialog() {
  if (!els.settingsDialog?.open) return;
  els.settingsDialog.close();
}

/** Drag the sidebar edge to change --panel-w; persists in localStorage. */
export function initPanelResize(onResize) {
  const handle = document.getElementById("panelResize");
  if (!handle) return;
  const PANEL_MIN = 240;
  const PANEL_MAX = 720;
  const STORAGE_KEY = "poly-cloud-panel-w";

  function panelInset() {
    const raw = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-inset"));
    return Number.isFinite(raw) ? raw : 12;
  }

  function clampW(w) {
    const max = Math.min(PANEL_MAX, Math.max(PANEL_MIN, window.innerWidth - 2 * panelInset() - 160));
    return Math.round(Math.min(max, Math.max(PANEL_MIN, w)));
  }

  function applyW(w) {
    const px = clampW(w);
    document.documentElement.style.setProperty("--panel-w", `${px}px`);
    handle.setAttribute("aria-valuenow", String(px));
    return px;
  }

  try {
    const saved = Number(localStorage.getItem(STORAGE_KEY));
    if (Number.isFinite(saved) && saved > 0) applyW(saved);
  } catch (_) {
    /* ignore */
  }

  handle.setAttribute("aria-valuemin", String(PANEL_MIN));
  handle.setAttribute("aria-valuemax", String(PANEL_MAX));

  let dragging = false;

  function onMove(ev) {
    if (!dragging) return;
    const x = ev.touches ? ev.touches[0].clientX : ev.clientX;
    applyW(x - panelInset());
    onResize();
  }

  function onUp() {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.classList.remove("panel-resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
    try {
      const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-w"));
      if (Number.isFinite(cur)) localStorage.setItem(STORAGE_KEY, String(cur));
    } catch (_) {
      /* ignore */
    }
    onResize();
  }

  handle.addEventListener("pointerdown", (ev) => {
    if (window.matchMedia("(max-width: 800px)").matches) return;
    ev.preventDefault();
    dragging = true;
    handle.classList.add("dragging");
    document.body.classList.add("panel-resizing");
    handle.setPointerCapture?.(ev.pointerId);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });

  handle.addEventListener("keydown", (ev) => {
    const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-w")) || 360;
    const step = ev.shiftKey ? 40 : 16;
    if (ev.key === "ArrowLeft") {
      ev.preventDefault();
      applyW(cur - step);
      onResize();
    } else if (ev.key === "ArrowRight") {
      ev.preventDefault();
      applyW(cur + step);
      onResize();
    }
  });
}
