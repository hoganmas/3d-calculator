/**
 * Headset-agnostic immersive-vr session shell (Three.js WebXRManager).
 * Volumes render via the WebGL Beer path while presenting; DOM UI stays flat.
 */
import { state } from "../state.js";
import { els } from "../dom.js";
import { renderer, controls } from "../scene.js";
import { setClipGpuCanvasVisible, isClipBakeGpuReady } from "../../render/webgpu/march.js";
import { syncClipPresentation, resize } from "../presentation.js";
import { initXrNav, resetXrView, resetXrWorldDesktop } from "./nav.js";

const SESSION_OPTIONS: XRSessionInit = {
  optionalFeatures: ["local-floor", "bounded-floor", "hand-tracking", "layers"],
};

let sessionHooked = false;

function updateEnterXrButton() {
  const btn = els.enterXr;
  if (!btn) return;
  const presenting = renderer.xr.isPresenting;
  btn.setAttribute("aria-pressed", presenting ? "true" : "false");
  const label = presenting ? "Exit XR" : "Enter XR";
  btn.setAttribute("aria-label", label);
  btn.dataset.tooltip = label;
}

function onSessionStart() {
  state.xrActive = true;
  controls.enabled = false;
  resetXrView();
  setClipGpuCanvasVisible(false);
  state.clipDirty = true;
  syncClipPresentation();
  updateEnterXrButton();
}

function onSessionEnd() {
  state.xrActive = false;
  controls.enabled = true;
  resetXrWorldDesktop();
  setClipGpuCanvasVisible(isClipBakeGpuReady());
  state.clipDirty = true;
  syncClipPresentation();
  resize();
  updateEnterXrButton();
}

async function enterXr() {
  if (!navigator.xr) return;
  const session = await navigator.xr.requestSession("immersive-vr", SESSION_OPTIONS);
  await renderer.xr.setSession(session);
}

async function exitXr() {
  const session = renderer.xr.getSession();
  if (session) await session.end();
}

async function toggleXr() {
  try {
    if (renderer.xr.isPresenting) await exitXr();
    else await enterXr();
  } catch (e) {
    console.warn("[xr] session request failed", e);
  }
}

/** Feature-detect immersive-vr and wire the toolbar Enter XR control. */
export function initXr() {
  initXrNav(renderer);

  if (!sessionHooked) {
    sessionHooked = true;
    renderer.xr.addEventListener("sessionstart", onSessionStart);
    renderer.xr.addEventListener("sessionend", onSessionEnd);
  }

  const btn = els.enterXr;
  if (!btn) return;

  btn.hidden = true;
  btn.addEventListener("click", () => {
    void toggleXr();
  });

  const xr = navigator.xr;
  if (!xr?.isSessionSupported) return;

  void xr.isSessionSupported("immersive-vr").then((ok) => {
    btn.hidden = !ok;
    if (ok) updateEnterXrButton();
  });
}

export { resetXrView, resetXrWorldDesktop };
