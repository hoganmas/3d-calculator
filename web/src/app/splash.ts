import { setErr } from "./hud.js";

const SPLASH_TIMEOUT_MS = 15_000;
const SPLASH_EXIT_MS = 700;

let splashEl: HTMLElement | null = null;
let timeoutId = 0;
let dismissed = false;
let sidebarReady = false;
let sceneReady = false;
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
  if (sidebarReady && sceneReady && frameReady) dismissSplash();
}

export function initSplash() {
  splashEl = document.getElementById("splash");
  timeoutId = window.setTimeout(() => {
    if (!sceneReady) {
      setErr("Startup took longer than expected — some features may still be loading.");
    }
    sidebarReady = true;
    sceneReady = true;
    frameReady = true;
    dismissSplash();
  }, SPLASH_TIMEOUT_MS);
}

export function markSplashSidebarReady() {
  if (sidebarReady) return;
  sidebarReady = true;
  tryDismiss();
}

export function markSplashSceneReady() {
  if (sceneReady) return;
  sceneReady = true;
  tryDismiss();
}

export function markSplashFrameReady() {
  if (frameReady) return;
  frameReady = true;
  tryDismiss();
}
