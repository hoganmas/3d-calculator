import { PRESETS } from "../math/fit.js";
import { getThemePref, setThemePref } from "../ui/theme.js";

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
  isoInterp: HTMLSelectElement;
  marchDownscale: HTMLInputElement;
  marchScaleLabel: HTMLElement | null;
  reset: HTMLButtonElement;
  err: HTMLElement;
  viewport: HTMLElement;
  hud: HTMLElement;
  metricsDump: HTMLElement | null;
  copyMetrics: HTMLButtonElement | null;
  openSettings: HTMLButtonElement | null;
  closeSettings: HTMLButtonElement | null;
  settingsDialog: HTMLDialogElement | null;
  themePref: HTMLSelectElement | null;
}

export const els: DomElements = {
  preset: el("preset"),
  exprList: el("exprList"),
  deg: el("deg"),
  scale: el("scale"),
  steps: el("steps"),
  boxSize: el("boxSize"),
  isoInterp: el("isoInterp"),
  marchDownscale: el("marchDownscale"),
  marchScaleLabel: document.getElementById("marchScaleLabel"),
  reset: el("reset"),
  err: el("err"),
  viewport: el("viewport"),
  hud: el("hud"),
  metricsDump: document.getElementById("metricsDump"),
  copyMetrics: document.getElementById("copyMetrics") as HTMLButtonElement | null,
  openSettings: document.getElementById("openSettings") as HTMLButtonElement | null,
  closeSettings: document.getElementById("closeSettings") as HTMLButtonElement | null,
  settingsDialog: document.getElementById("settingsDialog") as HTMLDialogElement | null,
  themePref: document.getElementById("themePref") as HTMLSelectElement | null,
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
      setThemePref(els.themePref!.value as "dark" | "light" | "system");
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
export function initPanelResize(onResize: () => void) {
  const el = document.getElementById("panelResize");
  if (!el) return;
  const grip: HTMLElement = el;
  const PANEL_MIN = 240;
  const PANEL_MAX = 720;
  const STORAGE_KEY = "poly-cloud-panel-w";

  function panelInset() {
    const raw = parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue("--panel-inset"),
    );
    return Number.isFinite(raw) ? raw : 12;
  }

  function clampW(w: number) {
    const max = Math.min(PANEL_MAX, Math.max(PANEL_MIN, window.innerWidth - 2 * panelInset() - 160));
    return Math.round(Math.min(max, Math.max(PANEL_MIN, w)));
  }

  function applyW(w: number) {
    const px = clampW(w);
    document.documentElement.style.setProperty("--panel-w", `${px}px`);
    grip.setAttribute("aria-valuenow", String(px));
    return px;
  }

  try {
    const saved = Number(localStorage.getItem(STORAGE_KEY));
    if (Number.isFinite(saved) && saved > 0) applyW(saved);
  } catch {
    /* ignore */
  }

  grip.setAttribute("aria-valuemin", String(PANEL_MIN));
  grip.setAttribute("aria-valuemax", String(PANEL_MAX));

  let dragging = false;

  function onMove(ev: PointerEvent | TouchEvent) {
    if (!dragging) return;
    const x = "touches" in ev ? ev.touches[0].clientX : ev.clientX;
    applyW(x - panelInset());
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
      const cur = parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue("--panel-w"),
      );
      if (Number.isFinite(cur)) localStorage.setItem(STORAGE_KEY, String(cur));
    } catch {
      /* ignore */
    }
    onResize();
  }

  grip.addEventListener("pointerdown", (ev) => {
    if (window.matchMedia("(max-width: 800px)").matches) return;
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
    const cur =
      parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--panel-w")) || 360;
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
