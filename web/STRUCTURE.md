# Source layout (`src/`)

Vite entry: [`index.html`](index.html) → [`src/main.ts`](src/main.ts) (thin bootstrap).

```
src/
  main.ts                 Wire modules, mount expression list, start loop
  app/
    state.ts              Shared mutable runtime state
    dom.ts                  DOM refs, settings dialog, panel resize
    scene.ts                THREE scene, camera, grid, lava background
    compile.ts              Expression compile + preset helpers
    persistence/            Scene document serialize, IndexedDB autosave, file I/O
    pipeline.ts             uploadFit, bakeChebVolume, keyframes
    webglFallback.ts        WebGL Beer march (no WebGPU)
    presentation.ts         Resize, march downscale, GPU/CPU presentation
    hud.ts                  Metrics HUD + compile status
    loop.ts                 setAnimationLoop (XR-aware) frame loop
    xr/
      session.ts            immersive-vr session + Enter XR button
      nav.ts                Controller grab-orbit + recenter
    persistence/            Scene document v1, IndexedDB autosave, file import/export
  math/
    fit.ts                  LaTeX compile + Chebyshev least-squares fit
    fitVector.ts            Vector field tuple/grad parse + fit
    idct.ts                 Separable Chebyshev IDCT → dens grid (+ grad)
    limits.ts               MAX_DEG and shared math constants
  model/
    expressions.ts          Expression list state, colors
    params.ts               Free-parameter values + animation
    keyframes.ts            Animated-param keyframe cache + GPU blend
  render/
    camera.ts               NDC → world ray helpers
    background.ts           Three.js fullscreen backdrop shader
    webgl/
      marchShaders.ts       WebGL Beer–Lambert fallback GLSL
    webgpu/
      march.ts              Public API (re-exports)
      marchTypes.ts         Shared march types + GPU handle guards
      marchReadiness.ts     Ready checks + iso interp mode
      marchProfile.ts       GPU timing / profiling
      marchCanvas.ts        Overlay canvas + offscreen targets
      marchInit.ts          Device init + pipeline bootstrap
      gridOverlay.ts        World grid, axes, axis-label billboards
      renderFrame.ts        Per-frame iso / SSAO / Beer / FXAA pass graph
      gpuState.ts           Shared GPU device / texture state
      uniforms.ts           Draw-param packing + layer color upload
      sceneUpload.ts        Volume upload + keyframe patch
      flowIbfv.ts           3D IBFV dye advection (compute)
      pipelines.ts          Shader module compile + pipeline creation
      shaders/
        compose.ts          Load .wgls ?raw, inject constants
        common/gradient.wgsl
        flowIbfv.wgsl
        isoHermite.wgsl
        isoTrilinear.wgsl
        beer.wgsl, grid.wgsl, axisLabel.wgsl, fxaa.wgsl, ssao.wgsl
  types/
    models.ts               Shared TS domain types (ExprItem, SceneBake, …)
  ui/
    expr-sidebar/           Svelte expression list (MathLive, drag, popovers)
      mount.ts              Public mountExprList API
      ExprSidebar.svelte    List host: render/sync API, drag controller
      ExprRow.svelte        Single expression row + math-field
      ParamRail.svelte      Parameter slider / animate controls
      helpers.ts            MathLive + row helpers
      popovers.ts           Gradient / animation popovers
      dragReorder.ts        Pointer drag-reorder controller
    expressionList.ts       Re-export mountExprList
    popover.ts              Floating UI popover helper
    liquidSlider.ts         Glass-style range slider thumbs
    theme.ts / theme.css
    app.css                 Panel / expr-list / settings layout styles
```

## Data flow

```
MathLive (ui/expr-sidebar)
  → compile + classify (app/compile, math/fit)
  → fit Chebyshev coeffs (math/fit)
  → IDCT volumes (math/idct)
  → upload + march (app/pipeline → render/webgpu/march, or app/webglFallback)
```

Animated free parameters bake keyframes in `model/keyframes.ts` (cold fit, hot GPU lerp).

## Layer rules

| Layer | May import from |
|---|---|
| `math/` | npm deps only |
| `model/` | `math/` |
| `render/` | `math/`, `model/`, sibling render modules |
| `ui/` | `math/`, `model/`, sibling `ui/` |
| `app/` | all of the above (orchestration) |
| `main.ts` | `app/`, `ui/` |

Avoid importing `app/` or `main.ts` from `math/`, `model/`, `render/`, or `ui/`.

## Notes

- **WGSL lives in `.wgsl` files** under `render/webgpu/shaders/`; Hermite vs trilinear iso uses two separate shaders selected at pipeline build.
- **Legacy names** — some internals still say `clip*` (e.g. `clipQuad`, `[clip-grid]` logs).
- **TS migration** — `web/src/` is fully TypeScript; imports use `.js` extensions. Vite resolves missing `.js` to sibling `.ts` via `resolveTsFromJs` in `vite.config.js`. WebGPU types from `@webgpu/types`.
- **Pipeline write-up** — [`research/poly/notes/cheb-idct-volume.md`](../research/poly/notes/cheb-idct-volume.md).
