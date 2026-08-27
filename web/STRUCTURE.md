# Source layout (`src/`)

Vite entry: [`index.html`](index.html) → [`src/main.js`](src/main.js).

```
src/
  main.js                 App orchestrator (scene, fit pipeline, render loop, HUD)
  math/
    fit.js                LaTeX compile + Chebyshev least-squares fit
    idct.js               Separable Chebyshev IDCT → dens grid (+ grad for isos)
    limits.js             Shared constants (e.g. MAX_DEG)
  model/
    expressions.js        Expression list state, colors, roles (manifold vs dens)
    params.js             Free-parameter values, sliders, ping-pong / loop animation
    keyframes.js          Animated-param keyframe cache + GPU blend sampling
  render/
    camera.js             NDC → world ray helpers for volume march
    background.js         Three.js fullscreen backdrop shader
    webgl/
      marchShaders.js     WebGL Beer–Lambert fallback (single summed dens texture)
    webgpu/
      march.js            WebGPU iso manifolds + multi-layer dens march
  ui/
    expressionList.js     Sidebar: MathLive rows, param sliders, play controls
    liquidSlider.js       Glass-style range slider thumbs
    theme.js              Light / dark / system theme + CSS variables
    theme.css
```

## Data flow

```
MathLive (ui/expressionList)
  → compile + classify (math/fit)
  → fit Chebyshev coeffs (math/fit)
  → IDCT volumes (math/idct)
  → upload + march (render/webgpu/march, or render/webgl fallback)
```

Animated free parameters bake keyframes in `model/keyframes.js` (cold fit, hot GPU lerp). Expression and param state live in `model/`; rendering code reads baked volumes and colors only.

## Layer rules

| Layer | May import from |
|---|---|
| `math/` | npm deps only |
| `model/` | `math/` |
| `render/` | `math/`, `model/` (colors), `render/camera.js` |
| `ui/` | `math/`, `model/`, sibling `ui/` |
| `main.js` | all of the above |

Avoid importing `ui/` or `main.js` from lower layers.

## Notes

- **`main.js` is large** — scene setup, `uploadFit`, WebGL fallback wiring, settings, and the animation loop still live here. A future split into `src/app/` is optional.
- **Legacy names in code** — some internals still say `clip*` (e.g. `clipQuad`, `[clip-grid]` logs); file paths use the names above.
- **Pipeline write-up** — [`research/poly/notes/cheb-idct-volume.md`](../research/poly/notes/cheb-idct-volume.md).
