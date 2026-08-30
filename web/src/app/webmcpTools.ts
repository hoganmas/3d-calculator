/**
 * Laplacian WebMCP tools — imperative registrations wrapping app/model APIs.
 */
import { PRESETS } from "../math/fit.js";
import {
  listExpressions,
  getSelectedId,
  updateExpr,
  insertExprAt,
  removeExpr,
} from "../model/expressions.js";
import {
  listParamNames,
  getParam,
  setParamValue,
  updateParam,
  toggleParamAnimate,
  stopParamAnimation,
  anyParamAnimating,
  normalizeAnimMode,
} from "../model/params.js";
import type { AnimMode, ExprItem } from "../types/models.js";
import { applyPreset } from "./compile.js";
import { getRenderSettingsSnapshot, serializeExpr, serializeParam } from "./persistence/document.js";
import { els } from "./dom.js";
import { syncExprCompileState, buildMetricsReport, refreshMetricsDump, getExpressionErrorReport } from "./hud.js";
import {
  scheduleUploadFit,
  uploadFit,
  applyRenderHyperparams,
} from "./pipeline.js";
import { syncMarchSlider, syncBoundsSlider } from "./presentation.js";
import { camera, controls, resetCameraView } from "./scene.js";
import { state } from "./state.js";
import { buildCapabilities } from "./webmcpCapabilities.js";

const PRESET_KEYS = Object.keys(PRESETS);

const CAPS_HINT = "Call laplacian_get_capabilities for full preset list, LaTeX syntax, limits, and tool reference.";

export type ToolResult = { ok: true; data: unknown } | { ok: false; error: string };

export function ok(data: unknown): ToolResult {
  return { ok: true, data };
}

export function err(message: string): ToolResult {
  return { ok: false, error: message };
}

function refreshAfterStructuralChange(): boolean {
  const compileOk = syncExprCompileState();
  state.exprListApi?.render();
  if (compileOk) scheduleUploadFit(0);
  return compileOk;
}

function refreshAfterParamChange(fromAnim = false): boolean {
  const compileOk = syncExprCompileState();
  state.exprListApi?.syncAllParamSliders?.();
  if (compileOk) {
    scheduleUploadFit(fromAnim ? 0 : 80, { fromAnim });
  }
  return compileOk;
}

export async function setRenderSettings(patch: {
  deg?: number;
  scale?: number;
  steps?: number;
  boxSize?: number;
  marchDownscale?: number;
}) {
  let refit = false;
  if (patch.deg != null && Number.isFinite(patch.deg)) {
    els.deg.value = String(Math.round(patch.deg));
    refit = true;
  }
  if (patch.boxSize != null && Number.isFinite(patch.boxSize)) {
    els.boxSize.value = String(patch.boxSize);
    syncBoundsSlider();
    refit = true;
  }
  if (patch.scale != null && Number.isFinite(patch.scale)) {
    els.scale.value = String(patch.scale);
    applyRenderHyperparams();
  }
  if (patch.steps != null && Number.isFinite(patch.steps)) {
    els.steps.value = String(Math.round(patch.steps));
    applyRenderHyperparams();
  }
  if (patch.marchDownscale != null && Number.isFinite(patch.marchDownscale)) {
    els.marchDownscale.value = String(Math.round(patch.marchDownscale));
    syncMarchSlider();
    state.clipDirty = true;
  }
  if (refit) {
    const compileOk = syncExprCompileState();
    if (compileOk) scheduleUploadFit(0);
    else return err(els.err.textContent || "Compile failed");
  }
  return ok(getRenderSettingsSnapshot());
}

export function resetCamera() {
  resetCameraView();
  return ok({
    position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
    target: { x: controls.target.x, y: controls.target.y, z: controls.target.z },
  });
}

/** Load lava-lamp preset and ensure all blob position params are animating. */
export async function setupLavaLamp(opts: { tuneRender?: boolean } = {}) {
  applyPreset("lavalamp");
  if (state.fitTimer) clearTimeout(state.fitTimer);
  syncExprCompileState();
  for (const name of listParamNames()) {
    const p = getParam(name);
    if (p && !p.driven && !p.animating) toggleParamAnimate(name);
  }
  if (opts.tuneRender !== false) {
    const tune = await setRenderSettings({
      deg: 20,
      scale: 3.2,
      steps: 24,
      boxSize: 6,
      marchDownscale: 1,
    });
    if (!tune.ok) return tune;
  }
  uploadFit({ fromAnim: anyParamAnimating() });
  state.exprListApi?.render();
  const compileOk = !els.err.textContent;
  return ok({
    preset: "lavalamp",
    expressions: listExpressions().map(serializeExpr),
    params: listParamNames().map((n) => serializeParam(n)).filter(Boolean),
    animating: anyParamAnimating(),
    compileOk,
    error: compileOk ? null : els.err.textContent || null,
  });
}

export function setAllParamAnimation(animating: boolean) {
  for (const name of listParamNames()) {
    const p = getParam(name);
    if (!p || p.driven) continue;
    if (animating && !p.animating) toggleParamAnimate(name);
    else if (!animating && p.animating) stopParamAnimation(name);
  }
  const compileOk = refreshAfterParamChange(anyParamAnimating());
  return ok({
    animating: anyParamAnimating(),
    params: listParamNames().map((n) => serializeParam(n)).filter(Boolean),
    compileOk,
  });
}

type ToolDef = {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  readOnly?: boolean;
  execute: (input: Record<string, unknown>) => Promise<ToolResult>;
};

function tools(): ToolDef[] {
  return [
    {
      name: "laplacian_get_capabilities",
      description:
        "Complete reference for MCP agents: all presets (keys, labels, latex, param seeds), LaTeX/expression syntax, spatial symbols, animation modes, render setting limits, tool catalog, and WebMCP setup notes. Read this first.",
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () => ok(buildCapabilities()),
    },
    {
      name: "laplacian_get_state",
      description: `Snapshot of expressions, parameters, render settings, selection, and compile meta. ${CAPS_HINT}`,
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () =>
        ok({
          selectedId: getSelectedId(),
          expressions: listExpressions().map(serializeExpr),
          params: listParamNames().map((n) => serializeParam(n)).filter(Boolean),
          render: getRenderSettingsSnapshot(),
          compile: {
            label: state.lastExprMeta.label,
            kind: state.lastExprMeta.kind,
            shade: state.lastExprMeta.shade,
            error: els.err.textContent || null,
          },
        }),
    },
    {
      name: "laplacian_get_metrics",
      description: "Diagnostics metrics report (same as Settings → Diagnostics). Read-only.",
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () => {
        refreshMetricsDump();
        return ok({ report: buildMetricsReport() });
      },
    },
    {
      name: "laplacian_get_compile_status",
      description: "Run compile sync and return ok/error plus structured expression errors.",
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () => {
        try {
          const result = syncExprCompileState();
          const report = getExpressionErrorReport();
          return ok({
            ok: result,
            error: els.err.textContent || null,
            errorCount: report?.errorCount ?? 0,
            errors: report?.errors ?? [],
            meta: state.lastExprMeta,
          });
        } catch (e) {
          const report = getExpressionErrorReport();
          return ok({
            ok: false,
            error: e instanceof Error ? e.message : String(e),
            errorCount: report?.errorCount ?? 0,
            errors: report?.errors ?? [],
            meta: state.lastExprMeta,
          });
        }
      },
    },
    {
      name: "laplacian_get_expression_errors",
      description:
        "Structured report of expression compile warnings, parameter errors, and global compile failures. Runs compile sync first.",
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () => {
        syncExprCompileState();
        const report = getExpressionErrorReport();
        return ok(report ?? { compileOk: true, globalError: null, errors: [], expressionCount: 0, errorCount: 0 });
      },
    },
    {
      name: "laplacian_list_expressions",
      description: "List all expression rows with id, latex, enabled, warnings.",
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () => ok({ expressions: listExpressions().map(serializeExpr) }),
    },
    {
      name: "laplacian_list_params",
      description: "List named parameters with values, ranges, animation state.",
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () =>
        ok({ params: listParamNames().map((n) => serializeParam(n)).filter(Boolean) }),
    },
    {
      name: "laplacian_get_render_settings",
      description: "Read poly deg, scale, steps, box size, march downscale, iso interpolation, preset.",
      readOnly: true,
      inputSchema: { type: "object", properties: {} },
      execute: async () => ok(getRenderSettingsSnapshot()),
    },
    {
      name: "laplacian_set_expression",
      description:
        "Update an expression row by id. Triggers Chebyshev refit when latex changes. May take seconds on large deg.",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string", description: "Expression row id" },
          latex: { type: "string" },
          enabled: { type: "boolean" },
          color: { type: "string", description: "Gradient start hex e.g. #ff4500" },
          color2: { type: "string", description: "Gradient end hex e.g. #ffec00" },
        },
        required: ["id"],
      },
      execute: async (input) => {
        const id = String(input.id ?? "");
        if (!id) return err("id is required");
        const patch: Partial<ExprItem> = {};
        if (input.latex != null) patch.latex = String(input.latex);
        if (input.enabled != null) patch.enabled = Boolean(input.enabled);
        if (input.color != null) patch.color = String(input.color);
        if (input.color2 != null) patch.color2 = String(input.color2);
        const row = updateExpr(id, patch);
        if (!row) return err(`Expression not found: ${id}`);
        const compileOk = refreshAfterStructuralChange();
        return ok({
          expression: serializeExpr(row),
          compileOk,
          error: compileOk ? null : els.err.textContent || "Compile failed",
        });
      },
    },
    {
      name: "laplacian_add_expression",
      description: "Insert a new expression row. Triggers refit when latex is non-empty.",
      inputSchema: {
        type: "object",
        properties: {
          latex: { type: "string", default: "" },
          index: { type: "number", description: "Insert index (default: end)" },
        },
      },
      execute: async (input) => {
        const index =
          input.index != null && Number.isFinite(Number(input.index))
            ? Number(input.index)
            : listExpressions().length;
        const row = insertExprAt(index, {
          latex: input.latex != null ? String(input.latex) : "",
        });
        const compileOk = refreshAfterStructuralChange();
        return ok({
          expression: serializeExpr(row),
          compileOk,
          error: compileOk ? null : els.err.textContent || null,
        });
      },
    },
    {
      name: "laplacian_remove_expression",
      description: "Remove an expression row by id. Triggers refit.",
      inputSchema: {
        type: "object",
        properties: { id: { type: "string" } },
        required: ["id"],
      },
      execute: async (input) => {
        const id = String(input.id ?? "");
        if (!id) return err("id is required");
        removeExpr(id);
        const compileOk = refreshAfterStructuralChange();
        return ok({ removed: id, compileOk });
      },
    },
    {
      name: "laplacian_apply_preset",
      description: `Load a built-in preset by key. Keys: ${PRESET_KEYS.join(", ")}. Triggers full Chebyshev refit. ${CAPS_HINT}`,
      inputSchema: {
        type: "object",
        properties: {
          preset: {
            type: "string",
            enum: PRESET_KEYS,
            description: "Preset key — see laplacian_get_capabilities for labels and latex.",
          },
        },
        required: ["preset"],
      },
      execute: async (input) => {
        const key = String(input.preset ?? "");
        if (!(key in PRESETS)) return err(`Unknown preset: ${key}`);
        applyPreset(key);
        if (state.fitTimer) clearTimeout(state.fitTimer);
        const compileOk = syncExprCompileState();
        uploadFit();
        return ok({
          preset: key,
          expressions: listExpressions().map(serializeExpr),
          compileOk,
          error: compileOk ? null : els.err.textContent || null,
        });
      },
    },
    {
      name: "laplacian_setup_lava_lamp",
      description:
        "Demo/test: load the animated 3-blob lava-lamp preset (9 drifting position params), start all animations, and tune render settings for smooth volume view. Equivalent to apply_preset lavalamp + start all param animations.",
      inputSchema: {
        type: "object",
        properties: {
          tuneRender: {
            type: "boolean",
            description: "When true (default), set deg=20, scale=3.2, steps=24, marchDownscale=1.",
            default: true,
          },
        },
      },
      execute: async (input) =>
        setupLavaLamp({ tuneRender: input.tuneRender !== false }),
    },
    {
      name: "laplacian_set_param",
      description: "Set a named parameter value. Stops animation unless keepAnimating is true.",
      inputSchema: {
        type: "object",
        properties: {
          name: { type: "string" },
          value: { type: "number" },
          keepAnimating: { type: "boolean" },
        },
        required: ["name", "value"],
      },
      execute: async (input) => {
        const name = String(input.name ?? "");
        if (!name) return err("name is required");
        const next = setParamValue(name, Number(input.value), {
          stopAnim: !input.keepAnimating,
        });
        if (!next) return err(`Parameter not found: ${name}`);
        const compileOk = refreshAfterParamChange(false);
        return ok({ param: serializeParam(name), compileOk });
      },
    },
    {
      name: "laplacian_set_param_range",
      description: "Set min/max slider range for a parameter.",
      inputSchema: {
        type: "object",
        properties: {
          name: { type: "string" },
          min: { type: "number" },
          max: { type: "number" },
        },
        required: ["name"],
      },
      execute: async (input) => {
        const name = String(input.name ?? "");
        if (!name) return err("name is required");
        const patch: { min?: number; max?: number } = {};
        if (input.min != null) patch.min = Number(input.min);
        if (input.max != null) patch.max = Number(input.max);
        const next = updateParam(name, patch);
        if (!next) return err(`Parameter not found: ${name}`);
        const compileOk = refreshAfterParamChange(false);
        return ok({ param: serializeParam(name), compileOk });
      },
    },
    {
      name: "laplacian_set_param_animation",
      description: "Start, stop, or toggle parameter slider animation.",
      inputSchema: {
        type: "object",
        properties: {
          name: { type: "string" },
          animating: { type: "boolean", description: "true=start, false=stop, omit=toggle" },
          speed: { type: "number" },
          animMode: { type: "string", enum: ["pingpong", "loop"] },
        },
        required: ["name"],
      },
      execute: async (input) => {
        const name = String(input.name ?? "");
        if (!name) return err("name is required");
        let p = getParam(name);
        if (!p) return err(`Parameter not found: ${name}`);
        if (input.speed != null && Number.isFinite(Number(input.speed))) {
          p = updateParam(name, { speed: Number(input.speed) }) ?? p;
        }
        if (input.animMode != null) {
          p = updateParam(name, { animMode: normalizeAnimMode(input.animMode) as AnimMode }) ?? p;
        }
        if (input.animating === true) {
          if (!p.animating) p = toggleParamAnimate(name) ?? p;
        } else if (input.animating === false) {
          p = stopParamAnimation(name) ?? p;
        } else if (input.animating === undefined) {
          p = toggleParamAnimate(name) ?? p;
        }
        const fromAnim = anyParamAnimating();
        const compileOk = refreshAfterParamChange(fromAnim);
        return ok({ param: serializeParam(name), compileOk });
      },
    },
    {
      name: "laplacian_set_all_param_animation",
      description:
        "Start or stop slider animation on every non-driven parameter. Use after loading animated presets (pulse, twist, lavalamp).",
      inputSchema: {
        type: "object",
        properties: {
          animating: {
            type: "boolean",
            description: "true = start all, false = stop all.",
          },
        },
        required: ["animating"],
      },
      execute: async (input) => {
        if (input.animating !== true && input.animating !== false) {
          return err("animating (boolean) is required");
        }
        return setAllParamAnimation(Boolean(input.animating));
      },
    },
    {
      name: "laplacian_set_render_settings",
      description:
        "Update render/fit settings (deg, scale, steps, box size, march downscale). deg/box trigger refit.",
      inputSchema: {
        type: "object",
        properties: {
          deg: { type: "number" },
          scale: { type: "number" },
          steps: { type: "number" },
          boxSize: { type: "number" },
          marchDownscale: { type: "number" },
        },
      },
      execute: async (input) => setRenderSettings(input),
    },
    {
      name: "laplacian_reset_camera",
      description: "Reset orbit camera to the default view.",
      inputSchema: { type: "object", properties: {} },
      execute: async () => resetCamera(),
    },
  ];
}

/** Register all Laplacian tools; abort `signal` to unregister. */
export async function registerLaplacianTools(signal: AbortSignal) {
  const ctx = document.modelContext;
  if (!ctx?.registerTool) {
    console.warn("[WebMCP] document.modelContext.registerTool unavailable");
    return 0;
  }

  let count = 0;
  for (const tool of tools()) {
    if (signal.aborted) break;
    await ctx.registerTool(
      {
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
        annotations: { readOnlyHint: !!tool.readOnly },
        execute: async (input, execCtx) => {
          if (execCtx?.signal?.aborted) throw new DOMException("Aborted", "AbortError");
          try {
            return await tool.execute(input ?? {});
          } catch (e) {
            return err(e instanceof Error ? e.message : String(e));
          }
        },
      },
      { signal },
    );
    count++;
  }
  console.info(`[WebMCP] Registered ${count} Laplacian tools`);
  return count;
}
