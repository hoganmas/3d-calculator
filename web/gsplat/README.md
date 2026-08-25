# web/gsplat

Realtime browser research viewer for **classic 3DGS front-to-back compositing** with tunable transmittance early-out:

\[
T \leftarrow T(1-\alpha),\quad \text{stop when } T < \varepsilon
\]

No Unity. No CUDA. Kerbl-style **tile + sorted** software raster in a **Web Worker** so the UI stays live while frames render.

## Run

```bash
# from repo root
.venv/bin/python web/gsplat/gen_scene.py

cd web/gsplat
npm install
npm run dev
```

Open the printed localhost URL. Drag to orbit, scroll to zoom. Use **Bench ε sweep** to compare `1e-4 / 1e-3 / 1e-2 / 5e-2` (ms + splat evals + early-out pixels).

Default res is **320×180** for interactive FPS; bump resolution in the panel when you want quality over speed.

## Custom scenes

Upload in the panel:

| Format | Notes |
|--|--|
| `.ply` | INRIA / nerfstudio 3DGS (binary or ascii). Decodes `exp(scale)`, `sigmoid(opacity)`, SH DC → RGB |
| `.splat` | antimatter15 32-byte packed format |
| `.json` | This app’s `web-gsplat-v1` scene |

Large files are **subsampled** (Max Gaussians) so the CPU raster stays interactive.

## Controls

| Control | Meaning |
|--|--|
| Live render | Keep redrawing (useful for FPS); off = only on camera/ε change |
| Load 3DGS file | Replace the demo scene |
| Max Gaussians | Cap / subsample for big PLYs |
| Early-out ε | Stop blending when remaining T &lt; ε |
| Resolution | Trade FPS vs fidelity |
| Bench ε sweep | Timing + eval counts for several ε |

## Next (real gsplat)

When you have CUDA: port the same `T < ε` stop into gsplat’s tile rasterize kernel and A/B quality vs throughput.
