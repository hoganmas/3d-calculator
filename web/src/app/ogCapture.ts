/**
 * Query-param hook for scripts/render-og-image.mjs and /api/og (?ogCapture=1).
 */
import { setExpressions, EXPR_GRADIENTS } from "../model/expressions.js";
import type { ExprItem } from "../types/models.js";
import { decodeCompactFragment } from "./persistence/exprShareCodec.js";
import { normalizeSharePayload } from "./persistence/exprShare.js";
import { compileAllExprs } from "./compile.js";
import { uploadFit } from "./pipeline.js";
import { camera, controls, resetCameraView } from "./scene.js";
import { resize } from "./presentation.js";
import { applyQualityFromState, markQualityUserOverride } from "./quality.js";
import { els } from "./dom.js";
import { state } from "./state.js";

type Vec3 = [number, number, number];

type LoadOpts = {
  /** Built-in palette index (0–4). */
  palette?: number;
  /**
   * Parameter rows (e.g. `t=0.35`) the loaded expression may reference.
   * Needed when loading a single row in isolation (as og.ts's per-panel
   * capture does) — otherwise an animated/parametric expression fails to
   * compile with an "Undefined parameter" error.
   */
  paramRows?: Partial<ExprItem>[];
};

function paletteAt(index: number) {
  return EXPR_GRADIENTS[((index % EXPR_GRADIENTS.length) + EXPR_GRADIENTS.length) % EXPR_GRADIENTS.length]!;
}

function prepareViewport() {
  document.documentElement.dataset.panelCollapsed = "true";
  resize();
}

/**
 * A capture is a one-shot server render, not an interactive session — there's
 * no reason to settle for the tier-adaptive boot defaults (the desktop boot
 * preset caps volume march at 8× downscale, see deviceTier.ts) or let
 * perf-adapt step quality back down mid-capture if a slow/software-rendered
 * headless GPU dips below its FPS threshold. Push every quality axis to its
 * ceiling and hide the grid/axes overlay (share images should show just the
 * surface, not editor chrome). `?ogDeg=` remains as an explicit override for
 * local testing/fast iteration when max quality isn't wanted.
 */
function prepareOgCaptureDefaults() {
  state.precisionQuality = 100;
  state.scalarQuality = 100;
  state.surfaceQuality = 100;
  state.vectorQuality = 100;
  applyQualityFromState({ force: true, refit: false, reseedFlow: false });
  markQualityUserOverride();

  const params = new URLSearchParams(location.search);
  const ogDeg = Number(params.get("ogDeg"));
  if (Number.isFinite(ogDeg) && ogDeg >= 2) {
    state.fitDeg = Math.round(ogDeg);
    if (els.deg) els.deg.value = String(state.fitDeg);
  }

  state.showGridAxes = false;
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

async function loadExpressions(rows: Partial<ExprItem>[]) {
  prepareViewport();
  setExpressions(rows);
  compileAllExprs({ rebuildUi: false });
  state.exprListApi?.render();
  if (state.fitTimer) window.clearTimeout(state.fitTimer);
  uploadFit();
  await waitForBake();
}

export function installOgCapture() {
  prepareOgCaptureDefaults();

  const api = {
    async load(latex: string, opts: LoadOpts = {}) {
      const grad = paletteAt(opts.palette ?? 0);
      await loadExpressions([
        ...(opts.paramRows ?? []),
        {
          latex,
          color: grad.color,
          color2: grad.color2,
          colors: [grad.color, grad.color2],
        },
      ]);
    },
    async loadExpressions(rows: Partial<ExprItem>[]) {
      await loadExpressions(rows);
    },
    async loadFromSharePayload(payload: string) {
      const decoded = await decodeCompactFragment(`e=${normalizeSharePayload(payload)}`);
      if (!decoded?.rows?.length) throw new Error("Invalid share payload");
      await loadExpressions(decoded.rows);
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
