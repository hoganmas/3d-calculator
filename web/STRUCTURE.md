# Source layout (`src/`)

Vite entry: [`index.html`](index.html) → [`src/main.js`](src/main.js) (thin bootstrap).

```
src/
  main.js                 Wire modules, mount expression list, start loop
  app/
    state.js              Shared mutable runtime state
    dom.js                  DOM refs, settings dialog, panel resize
    scene.js                THREE scene, camera, grid, lava background
    compile.js              Expression compile + preset helpers
    pipeline.js             uploadFit, bakeChebVolume, keyframes
    webglFallback.js        WebGL Beer march (no WebGPU)
    presentation.js         Resize, march downscale, GPU/CPU presentation
    hud.js                  Metrics HUD + compile status
    loop.js                 requestAnimationFrame loop
  math/
    fit.js                  LaTeX compile + Chebyshev least-squares fit
    idct.js                 Separable Chebyshev IDCT → dens grid (+ grad)
    limits.js               MAX_DEG and shared math constants
  model/
    expressions.js          Expression list state, colors, roles
    params.js               Free-parameter values + animation
    keyframes.js            Animated-param keyframe cache + GPU blend
  render/
    camera.js               NDC → world ray helpers
    background.js           Three.js fullscreen backdrop shader
    webgl/
      marchShaders.js       WebGL Beer–Lambert fallback GLSL
    webgpu/
      march.js              WebGPU init, frame graph, public exports
      gpuState.js           Shared GPU device / texture state
      uniforms.js           Draw-param packing + layer color upload
      sceneUpload.js        Volume upload + keyframe patch
      pipelines.js          Shader module compile + pipeline creation
      shaders/
        compose.js          Load .wgsl ?raw, inject constants
        common/gradient.wgsl
        isoHermite.wgsl
        isoTrilinear.wgsl
        beer.wgsl, grid.wgsl, axisLabel.wgsl, fxaa.wgsl, ssao.wgsl
  ui/
    expressionList.js       Sidebar: MathLive rows, sliders, play
    liquidSlider.js         Glass-style range slider thumbs
    theme.js / theme.css
```

## Data flow

```
MathLive (ui/expressionList)
  → compile + classify (app/compile, math/fit)
  → fit Chebyshev coeffs (math/fit)
  → IDCT volumes (math/idct)
  → upload + march (app/pipeline → render/webgpu/march, or app/webglFallback)
```

Animated free parameters bake keyframes in `model/keyframes.js` (cold fit, hot GPU lerp).

## Layer rules

| Layer | May import from |
|---|---|
| `math/` | npm deps only |
| `model/` | `math/` |
| `render/` | `math/`, `model/`, sibling render modules |
| `ui/` | `math/`, `model/`, sibling `ui/` |
| `app/` | all of the above (orchestration) |
| `main.js` | `app/`, `ui/` |

Avoid importing `app/` or `main.js` from `math/`, `model/`, `render/`, or `ui/`.

## Notes

- **WGSL lives in `.wgsl` files** under `render/webgpu/shaders/`; Hermite vs trilinear iso uses two separate shaders selected at pipeline build.
- **Legacy names** — some internals still say `clip*` (e.g. `clipQuad`, `[clip-grid]` logs).
- **Pipeline write-up** — [`research/poly/notes/cheb-idct-volume.md`](../research/poly/notes/cheb-idct-volume.md).
