/**
 * Light / dark / system theme — CSS tokens + JS color reads for Three.js / WebGPU.
 */

const STORAGE_KEY = "poly-cloud-theme";

export type ThemePref = "dark" | "light" | "system";
export type ThemeResolved = "dark" | "light";

const listeners = new Set<(resolved: ThemeResolved, pref: ThemePref) => void>();

export function getThemePref(): ThemePref {
  const v = localStorage.getItem(STORAGE_KEY);
  if (v === "light" || v === "system") return v;
  return "dark";
}

export function resolveTheme(pref: ThemePref): ThemeResolved {
  if (pref === "system") {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  return pref;
}

export function setThemePref(pref: ThemePref) {
  localStorage.setItem(STORAGE_KEY, pref);
  applyTheme(pref);
}

export function onThemeChange(fn: (resolved: ThemeResolved, pref: ThemePref) => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function applyTheme(pref: ThemePref = getThemePref()) {
  const resolved = resolveTheme(pref);
  document.documentElement.setAttribute("data-theme", resolved);
  document.documentElement.dataset.themePref = pref;
  for (const fn of listeners) fn(resolved, pref);
}

export function initTheme() {
  applyTheme(getThemePref());
  const mq = window.matchMedia("(prefers-color-scheme: light)");
  mq.addEventListener("change", () => {
    if (getThemePref() === "system") applyTheme("system");
  });
}

function cssHex(name: string) {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (!raw) return 0;
  if (raw.startsWith("#")) {
    const h = raw.slice(1);
    const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
    return parseInt(full, 16) || 0;
  }
  return 0;
}

function cssFloat(name: string, fallback = 0) {
  const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
  return Number.isFinite(v) ? v : fallback;
}

function cssColor(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function cssHexToRgb01(hexStr: string) {
  const h = hexStr.replace("#", "").trim();
  if (!h) return [1, 1, 1];
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(full, 16) || 0;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** Snapshot of theme colors for WebGL / WebGPU. */
export function readThemeColors() {
  const axisX = cssColor("--axis-x");
  const axisY = cssColor("--axis-y");
  const axisZ = cssColor("--axis-z");
  const boxEdge = cssHex("--box-edge");
  return {
    gridMajor: cssHex("--grid-major"),
    gridMinor: cssHex("--grid-minor"),
    boxEdge,
    boxEdgeRgb: cssHexToRgb01(cssColor("--box-edge")),
    axisX,
    axisY,
    axisZ,
    axisXRgb: cssHexToRgb01(axisX),
    axisYRgb: cssHexToRgb01(axisY),
    axisZRgb: cssHexToRgb01(axisZ),
    labelStroke: cssColor("--label-stroke"),
    lava1: cssColor("--lava-1"),
    lava2: cssColor("--lava-2"),
    lava3: cssColor("--lava-3"),
    isoAbsorb: [
      cssFloat("--iso-absorb-r", 0.22),
      cssFloat("--iso-absorb-g", 0.14),
      cssFloat("--iso-absorb-b", 0.28),
    ],
    isoEmit: [
      cssFloat("--iso-emit-r", 0.92),
      cssFloat("--iso-emit-g", 0.66),
      cssFloat("--iso-emit-b", 0.48),
    ],
    beerAbsorb: [
      cssFloat("--beer-absorb-r", 0.18),
      cssFloat("--beer-absorb-g", 0.12),
      cssFloat("--beer-absorb-b", 0.22),
    ],
    beerEmit: [
      cssFloat("--beer-emit-r", 0.85),
      cssFloat("--beer-emit-g", 0.62),
      cssFloat("--beer-emit-b", 0.45),
    ],
  };
}
