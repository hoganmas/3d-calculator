/** Sidebar docked left (wide) vs top strip (narrow). */
export const PANEL_LAYOUT_MQ = "(max-width: 800px)";
export const PANEL_COLLAPSED_KEY = "poly-cloud-panel-collapsed";
export const PANEL_TRANSITION_MS = 340;

export function isHorizontalPanelLayout() {
  return typeof window !== "undefined" && window.matchMedia(PANEL_LAYOUT_MQ).matches;
}

export function isPanelCollapsed() {
  return typeof document !== "undefined" && document.documentElement.dataset.panelCollapsed === "true";
}

export function readPanelProgress() {
  const raw = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--panel-progress"),
  );
  if (Number.isFinite(raw)) return Math.min(1, Math.max(0, raw));
  return isPanelCollapsed() ? 1 : 0;
}

export function setPanelDragging(dragging: boolean) {
  const root = document.documentElement;
  if (dragging) root.dataset.panelDragging = "true";
  else delete root.dataset.panelDragging;
}

export function setPanelProgress(progress: number) {
  const p = Math.min(1, Math.max(0, progress));
  document.documentElement.style.setProperty("--panel-progress", String(p));
}

export function readPanelCoverWidth() {
  if (isHorizontalPanelLayout()) return 0;
  return readPanelWidth() * (1 - readPanelProgress());
}

export function readPanelCoverHeight() {
  if (!isHorizontalPanelLayout()) return 0;
  // Full-screen mobile overlay — scene keeps the full viewport.
  return 0;
}

export function panelTransitionMs() {
  if (typeof window === "undefined") return 0;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return 0;
  const raw = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--panel-transition-duration"),
  );
  if (Number.isFinite(raw) && raw > 0) return Math.round(raw * 1000);
  return PANEL_TRANSITION_MS;
}

export function setPanelCollapsed(collapsed: boolean) {
  const root = document.documentElement;
  const wasCollapsed = isPanelCollapsed();
  root.dataset.panelCollapsed = collapsed ? "true" : "false";
  setPanelProgress(collapsed ? 1 : 0);
  try {
    localStorage.setItem(PANEL_COLLAPSED_KEY, collapsed ? "1" : "0");
  } catch {
    /* ignore */
  }
  if (wasCollapsed !== collapsed) {
    window.dispatchEvent(
      new CustomEvent("laplaci:panel-collapsed", { detail: { collapsed } }),
    );
  }
}

export function readPanelCollapsedPref(): boolean {
  try {
    return localStorage.getItem(PANEL_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function readPanelInset() {
  const raw = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--panel-inset"),
  );
  return Number.isFinite(raw) ? raw : 12;
}

export function readPanelWidth() {
  const raw = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--panel-w"),
  );
  return Number.isFinite(raw) ? raw : 360;
}

export function readPanelHeight() {
  const raw = parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--panel-h"),
  );
  return Number.isFinite(raw) ? raw : 280;
}
