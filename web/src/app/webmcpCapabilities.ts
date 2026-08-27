/**
 * Machine-readable reference for MCP clients — presets, syntax, limits, tools.
 */
import { PRESETS } from "../math/fit.js";
import { MAX_DEG } from "../math/limits.js";
import { getPageOrigin, getWidgetOriginsForRelay } from "./webmcpConfig.js";

const PRESET_KEYS = Object.keys(PRESETS);

const TOOL_CATALOG = [
  {
    name: "laplacian_get_capabilities",
    readOnly: true,
    summary: "Full reference: presets, LaTeX syntax, params, render limits, all tools.",
  },
  {
    name: "laplacian_get_state",
    readOnly: true,
    summary: "Snapshot: expressions, params, render settings, selection, compile meta.",
  },
  {
    name: "laplacian_get_metrics",
    readOnly: true,
    summary: "Diagnostics report (Settings → Diagnostics).",
  },
  {
    name: "laplacian_get_compile_status",
    readOnly: true,
    summary: "Run compile sync; return ok/error and layer meta.",
  },
  {
    name: "laplacian_list_expressions",
    readOnly: true,
    summary: "All expression rows (id, latex, enabled, role, colors, warnings).",
  },
  {
    name: "laplacian_list_params",
    readOnly: true,
    summary: "All named parameters (value, range, speed, animation state).",
  },
  {
    name: "laplacian_get_render_settings",
    readOnly: true,
    summary: "Read deg, scale, steps, boxSize, marchDownscale, isoInterp, preset key.",
  },
  {
    name: "laplacian_set_expression",
    summary: "Patch a row by id (latex, enabled, role, color, color2). Refits on latex change.",
  },
  {
    name: "laplacian_add_expression",
    summary: "Insert a row at index (default end). Refits when latex is non-empty.",
  },
  {
    name: "laplacian_remove_expression",
    summary: "Delete a row by id and refit.",
  },
  {
    name: "laplacian_apply_preset",
    summary: "Load a built-in preset by key; full refit.",
  },
  {
    name: "laplacian_setup_lava_lamp",
    summary: "Load animated 3-blob lava-lamp preset and start all param animations.",
  },
  {
    name: "laplacian_set_param",
    summary: "Set param value; stops animation unless keepAnimating=true.",
  },
  {
    name: "laplacian_set_param_range",
    summary: "Set param min/max slider range.",
  },
  {
    name: "laplacian_set_param_animation",
    summary: "Start/stop/toggle one param; optional speed and animMode.",
  },
  {
    name: "laplacian_set_all_param_animation",
    summary: "Start or stop animation on every parameter at once.",
  },
  {
    name: "laplacian_set_render_settings",
    summary: "Patch deg/scale/steps/boxSize/marchDownscale/isoInterp; deg/box refit.",
  },
  {
    name: "laplacian_reset_camera",
    summary: "Reset orbit camera to default view.",
  },
] as const;

/** @returns Structured capability document for MCP clients. */
export function buildCapabilities() {
  const presets = PRESET_KEYS.map((key) => {
    const p = PRESETS[key];
    return {
      key,
      label: p.label,
      latex: p.latex ?? null,
      paramNames: p.params ? Object.keys(p.params) : [],
      paramSeeds: p.params ?? {},
      hasCustomExpressions: Array.isArray(p.expressions) && p.expressions.length > 0,
    };
  });

  return {
    app: {
      name: "Laplacian",
      description:
        "Multi-expression 3D polynomial calculator. Each field expression is Chebyshev-fitted to a density volume (Beer–Lambert) or isosurface manifold. Shared named parameters animate over time via slider keyframes.",
      pipeline:
        "LaTeX → compile → Chebyshev fit (on structural changes) → IDCT volume → GPU march (every frame). Animated params refit only dirty layers via keyframe blend.",
    },
    expressionSyntax: {
      rows: [
        {
          form: "name=value",
          kind: "parameter",
          description:
            "Defines a named slider parameter. RHS may reference other params (e.g. a=2b). Gets min/max/speed/animate controls.",
        },
        {
          form: "A=B",
          kind: "constraint",
          description: "Implicit manifold → opaque isosurface at residual zero.",
        },
        {
          form: "bare expression or f=…",
          kind: "density",
          description: "Scalar field → volumetric density cloud (Beer–Lambert).",
        },
      ],
      roles: {
        auto: "Infer density vs constraint from syntax.",
        density: "Force volume rendering.",
        constraint: "Force isosurface manifold.",
      },
      freeSymbols:
        "Any identifier not a known function becomes a parameter (auto row inserted). Param names must match /^[A-Za-z][A-Za-z0-9_]*$/ on the LHS of name=value (e.g. a, u, d — not y1 which parses as subscript). Reserved spatial: x, y, z, r, theta, phi, rho.",
      polarCoords: {
        r: "Spherical radius √(x²+y²+z²)",
        theta: "Polar angle from +z (LaTeX \\theta)",
        phi: "Azimuth atan2(y,x) (LaTeX \\phi)",
        rho: "Cylindrical radius √(x²+y²) (LaTeX \\rho)",
      },
      functions: [
        "sin", "cos", "tan", "exp", "ln", "log", "sqrt", "abs", "max", "min", "hypot",
        "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "floor", "ceil", "round", "sign",
      ],
      colors: "Each row has color and color2 (hex) for gradient stops on its layer.",
    },
    parameters: {
      animation: {
        pingpong: "Smooth oscillation between min and max (sinusoidal easing). Default mode.",
        loop: "Linear wrap from min→max, repeating (good for rising/falling lava-lamp motion on y).",
      },
      speed: "Cycles per second scale factor (default ~0.35).",
      driven: "Params whose RHS references other params cannot animate until isolated.",
      keyframes:
        "One animating param per layer enables GPU keyframe blend (fast). Multiple animating params trigger Chebyshev refit per tick.",
    },
    renderSettings: {
      deg: { min: 1, max: MAX_DEG, description: "Chebyshev polynomial degree; triggers refit." },
      scale: { description: "Density falloff scale in march shader." },
      steps: { min: 8, max: 96, description: "Ray-march step count." },
      boxSize: { description: "World cube half-extent multiplier; triggers refit." },
      marchDownscale: { enum: [1, 2, 4, 8, 16], description: "March framebuffer resolution divisor." },
      isoInterp: { enum: ["trilinear", "hermite"], description: "Isosurface / density interpolation." },
      preset: { enum: PRESET_KEYS, description: "Last loaded preset key (informational)." },
    },
    presets,
    presetKeys: PRESET_KEYS,
    lavaLamp: {
      presetKey: "lavalamp",
      tool: "laplacian_setup_lava_lamp",
      description:
        "Three Gaussian density blobs on separate layers; each blob's vertical position (params u, v, w) animates in loop mode for rising/falling lava-lamp motion. One animating param per layer enables GPU keyframe blending. Param names must be simple identifiers (a, u, v) — not y1-style subscripts.",
      suggestedRender: { deg: 20, scale: 2.8, steps: 20, isoInterp: "hermite" },
    },
    tools: TOOL_CATALOG,
    webmcp: {
      enable:
        "On by default in dev and production builds (any http(s) host). Opt out with ?webmcp=0 or localStorage laplacian-webmcp=0.",
      relay:
        `@mcp-b/webmcp-local-relay with --widget-origin ${getWidgetOriginsForRelay()}. Local dev includes both 127.0.0.1 and localhost when applicable.`,
      setupDialog:
        "Setup MCP button in the header shows copy-paste server config for the current page origin.",
      singleTab:
        "Keep exactly ONE browser tab open on the app URL. The relay attaches to open tabs automatically.",
      doNotOpenTabs:
        "Never call relay webmcp_open_page unless the user explicitly asks — it spawns a new browser tab each time. Use webmcp_list_sources to verify connection; ask the user to reload the existing tab if disconnected.",
      singleRelay:
        "Only ONE webmcp-local-relay process should run (from your MCP client config). Extra relay instances cause ws port-scan errors (9333–9348) in the browser console.",
      refresh:
        "If needed, ask the user to reload their tab, or call webmcp_open_page with refresh:true (server relay only) — never bare open_page.",
      browserNote:
        "Relay requires a system browser tab (Chrome/Firefox/Safari). IDE embedded browser panels may not connect.",
      refitTimeout: "Large deg refits may take 10–120s; relay request timeout is 120s.",
      firstStep: "Call laplacian_get_capabilities, then verify webmcp_list_sources count >= 1 before mutating.",
    },
  };
}
