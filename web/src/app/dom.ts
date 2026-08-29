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
  deg: HTMLInputElement;
  scale: HTMLInputElement;
  steps: HTMLInputElement;
  boxSize: HTMLInputElement;
  marchDownscale: HTMLInputElement;
  marchScaleLabel: HTMLElement | null;
  reset: HTMLButtonElement;
  togglePanel: HTMLButtonElement | null;
  err: HTMLElement;
  viewport: HTMLElement;
  hud: HTMLElement;
  metricsDump: HTMLElement | null;
  copyMetrics: HTMLButtonElement | null;
  openSettings: HTMLButtonElement | null;
  closeSettings: HTMLButtonElement | null;
  settingsDialog: HTMLDialogElement | null;
  themePref: HTMLSelectElement | null;
  flowAlpha: HTMLInputElement | null;
  flowGridMode: HTMLSelectElement | null;
  flowNoiseScale: HTMLInputElement | null;
  flowDt: HTMLInputElement | null;
  flowSpeed: HTMLInputElement | null;
  flowVMax: HTMLInputElement | null;
  flowOpacity: HTMLInputElement | null;
  flowAgeMax: HTMLInputElement | null;
  flowVizMode: HTMLSelectElement | null;
  flowParticleCount: HTMLInputElement | null;
  flowTrailSteps: HTMLInputElement | null;
  flowTrailWidth: HTMLInputElement | null;
  downloadScene: HTMLButtonElement | null;
  openScene: HTMLButtonElement | null;
  shareScene: HTMLButtonElement | null;
  sceneFileInput: HTMLInputElement | null;
  autosaveStatus: HTMLElement | null;
  autosaveStatusBar: HTMLElement | null;
}

export const els: DomElements = {
  preset: el("preset"),
  exprList: el("exprList"),
  deg: el("deg"),
  scale: el("scale"),
  steps: el("steps"),
  boxSize: el("boxSize"),
  marchDownscale: el("marchDownscale"),
  marchScaleLabel: document.getElementById("marchScaleLabel"),
  reset: el("reset"),
  togglePanel: document.getElementById("togglePanel") as HTMLButtonElement | null,
  err: el("err"),
  viewport: el("viewport"),
  hud: el("hud"),
  metricsDump: document.getElementById("metricsDump"),
  copyMetrics: document.getElementById("copyMetrics") as HTMLButtonElement | null,
  openSettings: document.getElementById("openSettings") as HTMLButtonElement | null,
  closeSettings: document.getElementById("closeSettings") as HTMLButtonElement | null,
  settingsDialog: document.getElementById("settingsDialog") as HTMLDialogElement | null,
  themePref: document.getElementById("themePref") as HTMLSelectElement | null,
  flowAlpha: document.getElementById("flowAlpha") as HTMLInputElement | null,
  flowGridMode: document.getElementById("flowGridMode") as HTMLSelectElement | null,
  flowNoiseScale: document.getElementById("flowNoiseScale") as HTMLInputElement | null,
  flowDt: document.getElementById("flowDt") as HTMLInputElement | null,
  flowSpeed: document.getElementById("flowSpeed") as HTMLInputElement | null,
  flowVMax: document.getElementById("flowVMax") as HTMLInputElement | null,
  flowOpacity: document.getElementById("flowOpacity") as HTMLInputElement | null,
  flowAgeMax: document.getElementById("flowAgeMax") as HTMLInputElement | null,
  flowVizMode: document.getElementById("flowVizMode") as HTMLSelectElement | null,
  flowParticleCount: document.getElementById("flowParticleCount") as HTMLInputElement | null,
  flowTrailSteps: document.getElementById("flowTrailSteps") as HTMLInputElement | null,
  flowTrailWidth: document.getElementById("flowTrailWidth") as HTMLInputElement | null,
  downloadScene: document.getElementById("downloadScene") as HTMLButtonElement | null,
  openScene: document.getElementById("openScene") as HTMLButtonElement | null,
  shareScene: document.getElementById("shareScene") as HTMLButtonElement | null,
  sceneFileInput: document.getElementById("sceneFileInput") as HTMLInputElement | null,
  autosaveStatus: document.getElementById("autosaveStatus"),
  autosaveStatusBar: document.getElementById("autosaveStatusBar"),
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
  const label = collapsed ? "Show sidebar" : "Hide sidebar";
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
