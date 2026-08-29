import { getClipGpuProfile } from "../render/webgpu/march.js";
import { getFlowParticleMetrics } from "../render/webgpu/flowParticles.js";
import { hasFlowGpuLayers } from "../render/webgpu/flowGpu.js";
import { getParamValues } from "../model/params.js";
import { getExprWarning } from "../model/expressions.js";
import { els } from "./dom.js";
import { state } from "./state.js";
import { renderer } from "./scene.js";
import { clipUniforms, useGpuClipPath } from "./webglFallback.js";
import {
  marchDownscale,
  marchFramebufferSize,
} from "./presentation.js";
import { compileAllExprs, fmtParamNum } from "./compile.js";

export function fmtRel(v: number) {
  if (!Number.isFinite(v)) return "∞";
  if (v < 1e-3) return v.toExponential(2);
  return v.toPrecision(3);
}

export function setErr(msg: string) {
  els.err.textContent = msg || "";
}

/** Highlight expression fields when compile fails (preserve duplicate-var warnings). */
export function setExprCompileOk(ok: boolean) {
  els.exprList?.querySelectorAll(".expr-field").forEach((mf) => {
    const row = mf.closest?.(".expr-row");
    const id = row instanceof HTMLElement ? row.dataset.id : null;
    const warn = id ? getExprWarning(id) : null;
    if (warn) {
      mf.classList.add("invalid");
      if (mf instanceof HTMLElement) mf.title = warn;
      return;
    }
    mf.classList.toggle("invalid", !ok);
    if (mf instanceof HTMLElement && !ok) {
      /* keep existing title if any */
    } else if (mf instanceof HTMLElement) {
      mf.removeAttribute("title");
    }
  });
}

export function syncExprCompileState() {
  try {
    const result = compileAllExprs({ rebuildUi: true });
    setExprCompileOk(true);
    const warn = result?.warnings?.length ? result.warnings.join(" · ") : "";
    setErr(warn);
    return true;
  } catch (e) {
    setExprCompileOk(false);
    setErr(e instanceof Error ? e.message : String(e));
    return false;
  }
}

export function readIsoLevel() {
  return Number.isFinite(state.lastExprMeta.isoLevel) ? state.lastExprMeta.isoLevel : 0;
}

/** FPS shown in HUD — GPU present rate when WebGPU is active, else rAF spacing. */
function hudFpsText() {
  const loop = 1000 / Math.max(1, state.frameDtSmooth);
  if (useGpuClipPath()) {
    const p = getClipGpuProfile();
    const presentFresh =
      p.presentIntervalMs > 0 && performance.now() - p.lastPresentAt < 1200;
    if (presentFresh) {
      const gpu = 1000 / p.presentIntervalMs;
      if (loop > gpu * 1.12) {
        return `${Math.round(gpu)} fps · loop ${Math.round(loop)}`;
      }
      return `${Math.round(gpu)} fps`;
    }
  }
  return `${Math.round(loop)} fps`;
}

export function hudText() {
  const w = els.viewport.clientWidth;
  const h = Math.max(els.viewport.clientHeight, 1);
  const pr = renderer.getPixelRatio();
  const submit = state.densSubmittedThisFrame
    ? `submit ${state.bakeMsSmooth.toFixed(0)}ms`
    : `submit miss (last ${state.lastDensSubmitMs.toFixed(0)}ms)`;
  const p = getClipGpuProfile();
  const gpuSplit = p.timestamps
    ? ` · idct ${p.idctMs.toFixed(2)}/march ${p.marchMs.toFixed(1)}`
    : "";
  const present =
    p.presentIntervalMs > 0 && performance.now() - p.lastPresentAt < 1200
      ? ` · present ${p.presentIntervalMs.toFixed(0)}ms`
      : "";
  const clip = ` · rAF ${state.frameDtSmooth.toFixed(0)}ms · ${submit}${gpuSplit}${present} · vol ${state.lastVolumeM}³`;
  return `clip-grid · ${hudFpsText()} · ${state.cpuMsSmooth.toFixed(1)}ms js${clip} · ${Math.round(w * pr)}×${Math.round(h * pr)}`;
}

export function buildMetricsReport() {
  const fbW = Math.max(1, renderer.domElement.width);
  const fbH = Math.max(1, renderer.domElement.height);
  const p = getClipGpuProfile();
  const lines = [
    `poly-cloud metrics  ${new Date().toISOString()}`,
    `deg             ${state.fitDeg}`,
    `scale           ${clipUniforms.uScale.value}`,
    `steps           ${clipUniforms.uSteps.value}`,
    `box_size        ${2 * clipUniforms.uHalf.value}`,
    `march_downscale ${marchDownscale()}×`,
    `march_resolution ${(100 / marchDownscale()).toFixed(1)}%`,
    `viewport        ${fbW}×${fbH}`,
    `march_fb_req    ${marchFramebufferSize().mw}×${marchFramebufferSize().mh}`,
    `gpu_march_fb    ${p.marchFbW && p.marchFbH ? `${p.marchFbW}×${p.marchFbH}` : "—"}`,
    `loop_fps        ${Math.round(state.loopFps)}`,
    `loop_ms         ${state.frameDtSmooth.toFixed(2)}`,
    `js_frame_ms     ${state.cpuMsSmooth.toFixed(2)}`,
    `gpu_path        ${useGpuClipPath() ? "webgpu" : "cpu/webgl"}`,
    `gpu_method      ${p.method || "—"}`,
    `iso_interp      tricubic Hermite`,
    `expr_kind       ${state.lastExprMeta.kind}`,
    `shade           ${state.lastExprMeta.shade}`,
    `iso_level       ${readIsoLevel()}`,
    `grid_m          ${state.lastVolumeM || p.gridM || "—"}`,
    `march_submit_ms ${state.densSubmittedThisFrame ? state.bakeMsSmooth.toFixed(2) : "—"}`,
    `march_last_ms   ${state.lastDensSubmitMs.toFixed(2)}`,
    `gpu_timestamps  ${p.timestamps ? "yes" : "no"}`,
    `idct_ms         ${p.idctMs ? p.idctMs.toFixed(3) : "n/a"}`,
    `gpu_march_ms    ${p.timestamps ? p.marchMs.toFixed(3) : "n/a"}`,
    `gpu_present_ms  ${p.presentWallMs > 0 ? p.presentWallMs.toFixed(2) : "n/a"}`,
    `gpu_present_iv  ${p.presentIntervalMs > 0 ? p.presentIntervalMs.toFixed(2) : "n/a"}`,
    `gpu_present_fps ${
      p.presentIntervalMs > 0 ? Math.round(1000 / p.presentIntervalMs) : "n/a"
    }`,
    `fit_rel_L2      ${Number.isFinite(state.lastFitRel) ? fmtRel(state.lastFitRel) : "—"}`,
    `n_coeffs        ${state.lastNCoeff || "—"}`,
  ];
  if (state.lastFitTiming) {
    const t = state.lastFitTiming;
    lines.push(
      `fit_total_ms    ${t.totalMs.toFixed(2)}`,
      `fit_sample_ms   ${t.sampleMs.toFixed(2)}`,
      `fit_cheb_ms     ${t.chebMs.toFixed(2)}`,
      `fit_mono_ms     ${t.monoMs.toFixed(2)}`,
      `fit_l2_ms       ${t.l2Ms.toFixed(2)}`,
      `fit_upload_ms   ${(t.uploadMs ?? 0).toFixed(2)}`,
    );
    if (Number.isFinite(t.fittedCount)) {
      lines.push(`fit_layers      ${t.fittedCount}`);
    }
    if (Number.isFinite(t.keyframedCount)) {
      lines.push(`kf_layers       ${t.keyframedCount}`);
    }
    if (t.kfBakeMs != null && t.kfBakeMs > 0) {
      lines.push(`kf_bake_ms      ${t.kfBakeMs.toFixed(2)}`);
    }
    if (t.kfLerpMs != null && t.kfLerpMs > 0) {
      lines.push(`kf_lerp_ms      ${t.kfLerpMs.toFixed(2)}`);
    } else if (t.keyframedCount != null && t.keyframedCount > 0) {
      lines.push(`kf_blend        gpu`);
    }
    if (t.kfK != null && (t.keyframedCount ?? 0) > 0) {
      lines.push(`kf_K            ${t.kfK}`);
    }
  }
  const pv = getParamValues();
  const pNames = Object.keys(pv);
  if (pNames.length) {
    lines.push(
      `params          ${pNames.map((n) => `${n}=${fmtParamNum(pv[n])}`).join(" ")}`,
    );
  }
  if (hasFlowGpuLayers()) {
    const fp = getFlowParticleMetrics();
    lines.push(
      `flow_viz_mode     ${fp.vizMode}`,
      `flow_layers       ${fp.layerCount}`,
      `flow_per_layer    ${fp.perLayer}`,
      `flow_particles    ${fp.total}`,
      `flow_trail_steps  ${fp.trailSteps}`,
      `flow_trail_segs   ${fp.trailSegCount}`,
      `flow_draw_segs    ${fp.drawSegCount}${fp.segStride > 1 ? ` (stride ${fp.segStride})` : ""}`,
      `flow_ribbon_verts ${fp.ribbonDrawVerts}`,
      `flow_trail_buf_kb ${(fp.trailBufBytes / 1024).toFixed(1)}`,
      `flow_trail_push   every ${fp.trailPushInterval} frames`,
    );
    if (fp.active) {
      lines.push(
        `flow_tick_ms      ${fp.tickMs.toFixed(2)}`,
        `flow_tick_density ${fp.tickDensityMs.toFixed(2)}`,
        `flow_tick_advect  ${fp.tickAdvectMs.toFixed(2)}`,
        `flow_tick_redist  ${fp.tickRedistributeMs.toFixed(2)}`,
        `flow_tick_trail   ${fp.tickTrailMs.toFixed(2)}`,
        `flow_tick_sort    ${fp.tickSortMs.toFixed(2)}`,
        `flow_tick_upload  ${fp.tickUploadMs.toFixed(2)}`,
        `flow_draw_ms      ${fp.drawMs.toFixed(2)}`,
        `flow_speed_rng_ms ${fp.speedRangeMs.toFixed(2)}`,
        `flow_vmin         ${fp.speedMin.toFixed(4)}`,
        `flow_vmax         ${fp.speedMax.toFixed(4)}`,
      );
    }
  }
  return lines.join("\n");
}

export function refreshMetricsDump() {
  state.lastMetricsText = buildMetricsReport();
  if (els.metricsDump) (els.metricsDump as HTMLTextAreaElement).value = state.lastMetricsText;
}

export async function copyMetricsToClipboard() {
  const text = state.lastMetricsText || buildMetricsReport();
  try {
    await navigator.clipboard.writeText(text);
    if (els.copyMetrics) {
      els.copyMetrics.textContent = "Copied";
      if (state.copyMetricsResetTimer) clearTimeout(state.copyMetricsResetTimer);
      state.copyMetricsResetTimer = window.setTimeout(() => {
        if (els.copyMetrics) els.copyMetrics.textContent = "Copy";
      }, 1200);
    }
  } catch (e) {
    setErr(e instanceof Error ? e.message : "clipboard failed");
  }
}
