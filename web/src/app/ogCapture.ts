/**
 * Query-param hook for scripts/render-og-image.mjs (?ogCapture=1).
 */
import { setExpressions, EXPR_GRADIENTS } from "../model/expressions.js";
import { compileAllExprs } from "./compile.js";
import { uploadFit } from "./pipeline.js";
import { camera, controls, resetCameraView } from "./scene.js";
import { resize } from "./presentation.js";
import { els } from "./dom.js";
import { state } from "./state.js";

type Vec3 = [number, number, number];

type LoadOpts = {
  /** Built-in palette index (0–4). */
  palette?: number;
};

function paletteAt(index: number) {
  return EXPR_GRADIENTS[((index % EXPR_GRADIENTS.length) + EXPR_GRADIENTS.length) % EXPR_GRADIENTS.length]!;
}

function prepareViewport() {
  document.documentElement.dataset.panelCollapsed = "true";
  resize();
}

function wait(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}

async function waitForBake(timeoutMs = 60_000) {
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    const err = els.err?.textContent?.trim();
    if (err) throw new Error(err);
    const bake = state.lastSceneBake;
    const hasLayers =
      !!bake &&
      (bake.isosurfaceLayers.length > 0 || bake.cloudLayers.length > 0 || bake.flowLayers.length > 0);
    if (hasLayers && !state.fitTimer) {
      await wait(2500);
      return;
    }
    await wait(200);
    if (!state.fitTimer && !hasLayers) uploadFit();
  }
  throw new Error("Timed out waiting for scene bake");
}

export function installOgCapture() {
  const api = {
    async load(latex: string, opts: LoadOpts = {}) {
      prepareViewport();
      const grad = paletteAt(opts.palette ?? 0);
      setExpressions([
        {
          latex,
          color: grad.color,
          color2: grad.color2,
          colors: [grad.color, grad.color2],
        },
      ]);
      compileAllExprs({ rebuildUi: false });
      state.exprListApi?.render();
      if (state.fitTimer) window.clearTimeout(state.fitTimer);
      uploadFit();
      await waitForBake();
    },
    setCamera(position: Vec3, target: Vec3 = [0, 0, 0]) {
      camera.up.set(0, 0, 1);
      camera.position.set(position[0], position[1], position[2]);
      controls.target.set(target[0], target[1], target[2]);
      controls.update();
      state.clipDirty = true;
    },
    resetCamera() {
      prepareViewport();
      resetCameraView();
    },
    async waitFrame(ms = 1500) {
      await wait(ms);
    },
  };

  (window as Window & { __laplacianOgCapture?: typeof api }).__laplacianOgCapture = api;
}
