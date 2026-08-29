import { setErr } from "./hud.js";

import { allKeyframesComplete, hasActiveKeyframeCaches } from "../model/keyframes.js";

const SPLASH_TIMEOUT_MS = 15_000;
const SPLASH_EXIT_MS = 700;

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

function dismissSplash() {
  if (dismissed) return;
  dismissed = true;
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
    svg.classList.add("splash-exit");
    window.setTimeout(fadeSplashOut, SPLASH_EXIT_MS);
    return;
  }

  fadeSplashOut();
}

function tryDismiss() {
  if (dismissed) return;
  if (sidebarReady && contentReady && frameReady) dismissSplash();
}

/** True once the initial fit (and any keyframe cache) is ready. */
export function isSplashContentReady() {
  return contentReady;
}

/**
 * Gate splash on the first scene bake. When keyframes are building, waits until
 * every frame in the cache is ready so the default animation is fully loaded.
 */
export function tryMarkSplashBakeReady(hasSceneLayers: boolean) {
  if (contentReady) return;
  if (!hasSceneLayers) {
    contentReady = true;
    tryDismiss();
    return;
  }
  if (hasActiveKeyframeCaches() && !allKeyframesComplete()) return;
  contentReady = true;
  tryDismiss();
}

export function initSplash() {
  splashEl = document.getElementById("splash");
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
  tryDismiss();
}
