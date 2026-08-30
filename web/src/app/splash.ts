import { setErr } from "./hud.js";

import { allKeyframesComplete, hasActiveKeyframeCaches, keyframesSplashReady } from "../model/keyframes.js";
import { startupMark, startupReport } from "./startupProfile.js";

const SPLASH_TIMEOUT_MS = 15_000;
const SPLASH_EXIT_MS = 400;

let splashEl: HTMLElement | null = null;
let timeoutId = 0;
let dismissed = false;
let sidebarReady = false;
let contentReady = false;
let frameReady = false;

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function getSplashSvg(): SVGSVGElement | null {
  if (!splashEl) return null;
  const obj = splashEl.querySelector("object.splash-logo");
  if (!(obj instanceof HTMLObjectElement)) return null;
  const root = obj.contentDocument?.documentElement;
  return root instanceof SVGSVGElement ? root : null;
}

function removeSplash() {
  if (!splashEl?.isConnected) return;
  splashEl.remove();
  splashEl = null;
}

function fadeSplashOut() {
  const root = document.documentElement;
  delete root.dataset.booting;

  if (!splashEl) return;
  splashEl.setAttribute("aria-busy", "false");

  if (prefersReducedMotion()) {
    removeSplash();
    return;
  }

  const onEnd = (ev: TransitionEvent) => {
    if (ev.target !== splashEl || ev.propertyName !== "opacity") return;
    splashEl?.removeEventListener("transitionend", onEnd);
    removeSplash();
  };
  splashEl.addEventListener("transitionend", onEnd);
  window.setTimeout(removeSplash, 500);
}

function beginSplashExit(svg: SVGSVGElement) {
  svg.classList.add("splash-exit");
  for (const el of svg.querySelectorAll(".splash-nabla-sweep, .splash-two-stroke, .splash-two-layer")) {
    el.getAnimations().forEach((anim) => anim.cancel());
    if (el instanceof SVGElement) el.style.animation = "none";
  }
  void svg.getBoundingClientRect();
  for (const el of svg.querySelectorAll(".splash-nabla-sweep, .splash-two-stroke, .splash-two-layer")) {
    if (el instanceof SVGElement) el.style.removeProperty("animation");
  }
}

function dismissSplash() {
  if (dismissed) return;
  dismissed = true;
  startupMark("splash.dismiss");
  startupReport("splash-dismiss");
  if (timeoutId) {
    clearTimeout(timeoutId);
    timeoutId = 0;
  }

  if (prefersReducedMotion()) {
    fadeSplashOut();
    return;
  }

  const svg = getSplashSvg();
  if (svg) {
    beginSplashExit(svg);
    window.setTimeout(fadeSplashOut, SPLASH_EXIT_MS);
    return;
  }

  fadeSplashOut();
}

function splashDebugState(extra: Record<string, unknown> = {}) {
  return {
    dismissed,
    sidebarReady,
    contentReady,
    frameReady,
    keyframeCaches: hasActiveKeyframeCaches(),
    keyframesComplete: allKeyframesComplete(),
    keyframesSplashReady: keyframesSplashReady(),
    ...extra,
  };
}

function tryDismiss() {
  if (dismissed) return;
  if (sidebarReady && contentReady && frameReady) {
    dismissSplash();
  }
}

/** True once the initial scene bake has finished (keyframe fill may still run). */
export function isSplashContentReady() {
  return contentReady;
}

/** Snapshot for console debugging: copy(JSON.stringify(window.__laplaciSplash())) */
export function getSplashDebugSnapshot() {
  return splashDebugState();
}

/**
 * Gate splash on the first scene bake only. Keyframe grid refinement continues
 * after dismiss (viewport + splash load bars).
 */
export function tryMarkSplashBakeReady(_hasSceneLayers: boolean) {
  if (contentReady) return;
  contentReady = true;
  startupMark("splash.content-ready");
  tryDismiss();
}

export function initSplash() {
  splashEl = document.getElementById("splash");
  if (typeof window !== "undefined") {
    (window as Window & { __laplaciSplash?: () => ReturnType<typeof getSplashDebugSnapshot> }).__laplaciSplash =
      getSplashDebugSnapshot;
  }
  timeoutId = window.setTimeout(() => {
    if (!contentReady) {
      setErr("Startup took longer than expected — some features may still be loading.");
    }
    sidebarReady = true;
    contentReady = true;
    frameReady = true;
    dismissSplash();
  }, SPLASH_TIMEOUT_MS);
}

export function markSplashSidebarReady() {
  if (sidebarReady) return;
  sidebarReady = true;
  tryDismiss();
}

export function markSplashFrameReady() {
  if (frameReady) return;
  frameReady = true;
  startupMark("splash.frame-ready");
  tryDismiss();
}

/** Boot escape hatch when compile failed or we must not block on fit/GPU present. */
export function forceSplashDismiss(_reason?: string) {
  sidebarReady = true;
  if (!contentReady) contentReady = true;
  if (!frameReady) frameReady = true;
  dismissSplash();
}
