import { getKeyframeLoadSummary } from "../model/keyframes.js";

let barRoot: HTMLElement | null = null;
let barTrack: HTMLElement | null = null;
let barFill: HTMLElement | null = null;
let barLabel: HTMLElement | null = null;
let splashRoot: HTMLElement | null = null;
let splashTrack: HTMLElement | null = null;
let splashFill: HTMLElement | null = null;
let splashLabel: HTMLElement | null = null;

function applyProgress(
  track: HTMLElement | null,
  fill: HTMLElement | null,
  label: HTMLElement | null,
  pct: number,
  text: string,
) {
  if (fill) fill.style.width = `${pct}%`;
  if (track) {
    track.setAttribute("aria-valuenow", String(pct));
    track.setAttribute("aria-valuetext", text);
  }
  if (label) label.textContent = text;
}

export function initKeyframeLoadBar() {
  barRoot = document.getElementById("kfLoadBar");
  barTrack = document.getElementById("kfLoadTrack");
  barFill = document.getElementById("kfLoadFill");
  barLabel = document.getElementById("kfLoadLabel");
  splashRoot = document.getElementById("splashKfLoad");
  splashTrack = document.getElementById("splashKfTrack");
  splashFill = document.getElementById("splashKfFill");
  splashLabel = document.getElementById("splashKfLabel");
}

export function syncKeyframeLoadBar() {
  const summary = getKeyframeLoadSummary();
  const pct = Math.min(100, Math.max(0, Math.round(summary.fraction * 100)));
  const shortLabel = summary.complete
    ? ""
    : summary.label || `Loading animation · ${pct}%`;
  const showViewport = summary.active;
  const booting = document.documentElement.hasAttribute("data-booting");
  const showSplash = summary.active && booting;

  if (barRoot) {
    barRoot.hidden = !showViewport;
    barRoot.setAttribute("aria-hidden", showViewport ? "false" : "true");
    if (showViewport) {
      applyProgress(barTrack, barFill, barLabel, pct, shortLabel);
    }
  }

  if (splashRoot) {
    splashRoot.hidden = !showSplash;
    if (showSplash) {
      applyProgress(splashTrack, splashFill, splashLabel, pct, shortLabel);
    }
  }
}
