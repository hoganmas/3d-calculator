import { PRESETS } from "../math/fit.js";
import { getThemePref, setThemePref, type ThemePref } from "../ui/theme.js";
import {
  isHorizontalPanelLayout,
  isPanelCollapsed,
  panelTransitionMs,
  PANEL_LAYOUT_MQ,
  readPanelCollapsedPref,
  readPanelInset,
  setPanelCollapsed,
} from "./panelLayout.js";

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

export interface DomElements {
  preset: HTMLSelectElement;
  exprList: HTMLElement;
  exprFooter: HTMLElement | null;
  deg: HTMLInputElement;
  scale: HTMLInputElement;
  steps: HTMLInputElement;
  isoSteps: HTMLInputElement | null;
  boxSize: HTMLInputElement;
  boundsSizeLabel: HTMLElement | null;
  marchDownscale: HTMLInputElement;
  isoMarchDownscale: HTMLInputElement | null;
  toggleGridAxes: HTMLButtonElement | null;
  shareLink: HTMLButtonElement | null;
  marchScaleLabel: HTMLElement | null;
  isoMarchScaleLabel: HTMLElement | null;
  reset: HTMLButtonElement;
  togglePanel: HTMLButtonElement | null;
  collapsePanel: HTMLButtonElement | null;
  clearExprs: HTMLButtonElement | null;
  panelDismissHandle: HTMLElement | null;
  err: HTMLElement;
  viewport: HTMLElement;
  hud: HTMLElement;
  metricsDump: HTMLElement | null;
  copyMetrics: HTMLButtonElement | null;
  openSettings: HTMLButtonElement | null;
  openSettingsViewport: HTMLButtonElement | null;
  closeSettings: HTMLButtonElement | null;
  settingsDialog: HTMLDialogElement | null;
  themePref: HTMLSelectElement | null;
  flowDt: HTMLInputElement | null;
  flowSpeed: HTMLInputElement | null;
  flowVMax: HTMLInputElement | null;
  flowOpacity: HTMLInputElement | null;
  flowAgeMax: HTMLInputElement | null;
  flowParticleCount: HTMLInputElement | null;
  flowTrailSteps: HTMLInputElement | null;
  flowTrailWidth: HTMLInputElement | null;
  autosaveStatus: HTMLElement | null;
  autosaveStatusBar: HTMLElement | null;
  scalarQuality: HTMLInputElement | null;
  surfaceQuality: HTMLInputElement | null;
  vectorQuality: HTMLInputElement | null;
  precisionQuality: HTMLInputElement | null;
}

export const els: DomElements = {
  preset: el("preset"),
  exprList: el("exprList"),
  exprFooter: document.getElementById("exprFooter") as HTMLElement | null,
  deg: el("deg"),
  scale: el("scale"),
  steps: el("steps"),
  isoSteps: document.getElementById("isoSteps") as HTMLInputElement | null,
  boxSize: el("boxSize"),
  boundsSizeLabel: document.getElementById("boundsSizeLabel"),
  marchDownscale: el("marchDownscale"),
  isoMarchDownscale: document.getElementById("isoMarchDownscale") as HTMLInputElement | null,
  toggleGridAxes: document.getElementById("toggleGridAxes") as HTMLButtonElement | null,
  shareLink: document.getElementById("shareLink") as HTMLButtonElement | null,
  marchScaleLabel: document.getElementById("marchScaleLabel"),
  isoMarchScaleLabel: document.getElementById("isoMarchScaleLabel"),
  reset: el("reset"),
  togglePanel: document.getElementById("togglePanel") as HTMLButtonElement | null,
  collapsePanel: document.getElementById("collapsePanel") as HTMLButtonElement | null,
  clearExprs: document.getElementById("clearExprs") as HTMLButtonElement | null,
  panelDismissHandle: document.getElementById("panelDismissHandle") as HTMLElement | null,
  err: el("err"),
  viewport: el("viewport"),
  hud: el("hud"),
  metricsDump: document.getElementById("metricsDump"),
  copyMetrics: document.getElementById("copyMetrics") as HTMLButtonElement | null,
  openSettings: document.getElementById("openSettings") as HTMLButtonElement | null,
  openSettingsViewport: document.getElementById("openSettingsViewport") as HTMLButtonElement | null,
  closeSettings: document.getElementById("closeSettings") as HTMLButtonElement | null,
  settingsDialog: document.getElementById("settingsDialog") as HTMLDialogElement | null,
  themePref: document.getElementById("themePref") as HTMLSelectElement | null,
  flowDt: document.getElementById("flowDt") as HTMLInputElement | null,
  flowSpeed: document.getElementById("flowSpeed") as HTMLInputElement | null,
  flowVMax: document.getElementById("flowVMax") as HTMLInputElement | null,
  flowOpacity: document.getElementById("flowOpacity") as HTMLInputElement | null,
  flowAgeMax: document.getElementById("flowAgeMax") as HTMLInputElement | null,
  flowParticleCount: document.getElementById("flowParticleCount") as HTMLInputElement | null,
  flowTrailSteps: document.getElementById("flowTrailSteps") as HTMLInputElement | null,
  flowTrailWidth: document.getElementById("flowTrailWidth") as HTMLInputElement | null,
  autosaveStatus: document.getElementById("autosaveStatus"),
  autosaveStatusBar: document.getElementById("autosaveStatusBar"),
  scalarQuality: document.getElementById("scalarQuality") as HTMLInputElement | null,
  surfaceQuality: document.getElementById("surfaceQuality") as HTMLInputElement | null,
  vectorQuality: document.getElementById("vectorQuality") as HTMLInputElement | null,
  precisionQuality: document.getElementById("precisionQuality") as HTMLInputElement | null,
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
      setThemePref(els.themePref!.value as ThemePref);
    });
  }

  els.openSettings?.addEventListener("click", () => openSettingsDialog());
  els.openSettingsViewport?.addEventListener("click", () => openSettingsDialog());
  els.closeSettings?.addEventListener("click", () => closeSettingsDialog());
  els.settingsDialog?.addEventListener("click", (ev) => {
    if (ev.target === els.settingsDialog) closeSettingsDialog();
  });

  initViewportToolbarTouch();
}

/** On touch, release focus after toolbar taps so tooltips / focus rings do not stick. */
function initViewportToolbarTouch() {
  const toolbar = document.querySelector(".viewport-toolbar");
  if (!toolbar) return;
  const mq = window.matchMedia("(pointer: coarse)");

  const releaseFocus = (ev: Event) => {
    if (!mq.matches) return;
    const btn = (ev.target as Element | null)?.closest("button");
    if (!(btn instanceof HTMLButtonElement) || !toolbar.contains(btn)) return;
    queueMicrotask(() => btn.blur());
  };

  toolbar.addEventListener("click", releaseFocus);
}

export function openSettingsDialog() {
  if (!els.settingsDialog || typeof els.settingsDialog.showModal !== "function") return;
  if (!els.settingsDialog.open) els.settingsDialog.showModal();
}

export function closeSettingsDialog() {
  if (!els.settingsDialog?.open) return;
  els.settingsDialog.close();
}

/** Drag the sidebar edge to change --panel-w (vertical) or --panel-h (horizontal). */
export function initPanelResize(onResize: () => void) {
  const el = document.getElementById("panelResize");
  if (!el) return;
  const grip: HTMLElement = el;
  const PANEL_W_MIN = 240;
  const PANEL_W_MAX = 720;
  const PANEL_H_MIN = 160;
  const PANEL_H_MAX = 720;
  const STORAGE_W = "poly-cloud-panel-w";
  const STORAGE_H = "poly-cloud-panel-h";

  function panelInset() {
    return readPanelInset();
  }

  function clampW(w: number) {
    const max = Math.min(PANEL_W_MAX, Math.max(PANEL_W_MIN, window.innerWidth - 2 * panelInset() - 160));
    return Math.round(Math.min(max, Math.max(PANEL_W_MIN, w)));
  }

  function clampH(h: number) {
    const max = Math.min(
      PANEL_H_MAX,
      Math.max(PANEL_H_MIN, window.innerHeight - 2 * panelInset() - 160),
    );
    return Math.round(Math.min(max, Math.max(PANEL_H_MIN, h)));
  }

  function applyW(w: number) {
    const px = clampW(w);
    document.documentElement.style.setProperty("--panel-w", `${px}px`);
    grip.setAttribute("aria-valuenow", String(px));
    return px;
  }

  function applyH(h: number) {
    const px = clampH(h);
    document.documentElement.style.setProperty("--panel-h", `${px}px`);
    grip.setAttribute("aria-valuenow", String(px));
    return px;
  }

  function syncGripChrome() {
    const horizontal = isHorizontalPanelLayout();
    grip.setAttribute("aria-orientation", horizontal ? "horizontal" : "vertical");
    document.documentElement.dataset.panelLayout = horizontal ? "horizontal" : "vertical";
    if (horizontal) {
      grip.setAttribute("aria-valuemin", String(PANEL_H_MIN));
      grip.setAttribute("aria-valuemax", String(PANEL_H_MAX));
      const cur =
        parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-h")) || 280;
      grip.setAttribute("aria-valuenow", String(Math.round(cur)));
    } else {
      grip.setAttribute("aria-valuemin", String(PANEL_W_MIN));
      grip.setAttribute("aria-valuemax", String(PANEL_W_MAX));
      const cur =
        parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-w")) || 360;
      grip.setAttribute("aria-valuenow", String(Math.round(cur)));
    }
  }

  try {
    const savedW = Number(localStorage.getItem(STORAGE_W));
    if (Number.isFinite(savedW) && savedW > 0) applyW(savedW);
  } catch {
    /* ignore */
  }
  try {
    const savedH = Number(localStorage.getItem(STORAGE_H));
    if (Number.isFinite(savedH) && savedH > 0) applyH(savedH);
  } catch {
    /* ignore */
  }

  syncGripChrome();

  let dragging = false;

  function onMove(ev: PointerEvent | TouchEvent) {
    if (!dragging) return;
    if (isHorizontalPanelLayout()) {
      const y = "touches" in ev ? ev.touches[0].clientY : ev.clientY;
      applyH(y - panelInset());
    } else {
      const x = "touches" in ev ? ev.touches[0].clientX : ev.clientX;
      applyW(x - panelInset());
    }
    onResize();
  }

  function onUp() {
    if (!dragging) return;
    dragging = false;
    grip.classList.remove("dragging");
    document.body.classList.remove("panel-resizing");
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
    try {
      if (isHorizontalPanelLayout()) {
        const cur = parseFloat(
          getComputedStyle(document.documentElement).getPropertyValue("--panel-h"),
        );
        if (Number.isFinite(cur)) localStorage.setItem(STORAGE_H, String(cur));
      } else {
        const cur = parseFloat(
          getComputedStyle(document.documentElement).getPropertyValue("--panel-w"),
        );
        if (Number.isFinite(cur)) localStorage.setItem(STORAGE_W, String(cur));
      }
    } catch {
      /* ignore */
    }
    onResize();
  }

  grip.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    dragging = true;
    grip.classList.add("dragging");
    document.body.classList.add("panel-resizing");
    grip.setPointerCapture?.(ev.pointerId);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });

  grip.addEventListener("keydown", (ev) => {
    const horizontal = isHorizontalPanelLayout();
    const step = ev.shiftKey ? 40 : 16;
    if (horizontal) {
      const cur =
        parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-h")) || 280;
      if (ev.key === "ArrowUp") {
        ev.preventDefault();
        applyH(cur - step);
        onResize();
      } else if (ev.key === "ArrowDown") {
        ev.preventDefault();
        applyH(cur + step);
        onResize();
      }
      return;
    }
    const cur =
      parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-w")) || 360;
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

  const mq = window.matchMedia(PANEL_LAYOUT_MQ);
  mq.addEventListener("change", () => {
    syncGripChrome();
    onResize();
  });
}

function syncPanelToggleChrome(btn: HTMLButtonElement) {
  const collapsed = isPanelCollapsed();
  const horizontal = isHorizontalPanelLayout();
  const label = collapsed
    ? horizontal
      ? "Show expressions"
      : "Show sidebar"
    : horizontal
      ? "Hide expressions"
      : "Hide sidebar";
  btn.setAttribute("aria-label", label);
  btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  btn.dataset.tooltip = label;
}

function runPanelTransition(onResize: () => void) {
  const duration = panelTransitionMs();
  if (duration <= 0) {
    onResize();
    return;
  }
  const start = performance.now();
  const tick = (now: number) => {
    onResize();
    if (now - start < duration + 32) requestAnimationFrame(tick);
    else onResize();
  };
  requestAnimationFrame(tick);
}

/** Toggle sidebar visibility (wide left dock or narrow top strip). */
export function initPanelCollapse(onCollapse: () => void) {
  els.collapsePanel?.addEventListener("click", onCollapse);
}

/** Swipe up on the mobile expression-list footer to collapse back to the scene. */
export function initPanelDismiss(onCollapse: () => void) {
  const status = document.getElementById("panelStatus");
  if (!status) return;

  let startY = 0;
  let tracking = false;

  function onPointerDown(ev: PointerEvent) {
    if (!isHorizontalPanelLayout()) return;
    if (ev.button !== 0) return;
    tracking = true;
    startY = ev.clientY;
    status.setPointerCapture(ev.pointerId);
  }

  function onPointerUp(ev: PointerEvent) {
    if (!tracking) return;
    tracking = false;
    const dy = ev.clientY - startY;
    if (dy < -44) onCollapse();
  }

  function onPointerCancel() {
    tracking = false;
  }

  status.addEventListener("pointerdown", onPointerDown);
  status.addEventListener("pointerup", onPointerUp);
  status.addEventListener("pointercancel", onPointerCancel);

  els.panelDismissHandle?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      onCollapse();
    }
  });
}

/** Toggle sidebar visibility (wide left dock or narrow top strip). */
export function refreshPanelToggleChrome() {
  const btn = els.togglePanel;
  if (btn) syncPanelToggleChrome(btn);
}

/** Toggle sidebar visibility (wide left dock or narrow top strip). */
export function initPanelToggle(onResize: () => void) {
  const btn = els.togglePanel;
  if (!btn) return;

  const root = document.documentElement;
  root.dataset.panelInit = "";
  setPanelCollapsed(readPanelCollapsedPref());
  syncPanelToggleChrome(btn);
  requestAnimationFrame(() => {
    delete root.dataset.panelInit;
  });

  btn.addEventListener("click", () => {
    setPanelCollapsed(!isPanelCollapsed());
    syncPanelToggleChrome(btn);
    runPanelTransition(onResize);
  });
}
