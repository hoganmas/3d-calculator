# Polynomial density cloud viewer (Three.js)

1. Fit `f(x,y,z)` with a 3D Chebyshev polynomial, convert to a **world monomial tensor** \(c_{ijk}\).
2. **LOS modes** (raymarch / Path C): per-pixel nested Horner → univariate \(\gamma(u)\), then march / Chebyshev-\(T\).
3. **clip-grid**: per-view bake of fiber **density samples** on Chebyshev-root \(u\)-nodes (view-fixed \(t_\mathrm{mid},t_\mathrm{hw}\)). Prefer WebGPU **tile-parallel Babbage** (f32, tile≈128px) → resident atlas march; fall back to CPU f64 Babbage (tile≈256px) + upload / WebGL DataTexture. See `research/poly/notes/clip-space-babbage.md`.

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

Requires a browser with **WebGPU** for GPU bake + resident-atlas march (`navigator.gpu`). Without it, clip-grid falls back to CPU bake + WebGL DataTexture march automatically.

## Cost

| Stage | Cost |
|---|---|
| Fit (CPU, once) | Chebyshev + monomial convert |
| LOS per ray | \(\Theta(N^3)\) compose + \(\Theta(\mathrm{steps}\cdot 3N)\) |
| clip-grid bake (per view) | \(\sim O\!\big((W/\mathrm{tile})\,H\,N\cdot N^3 + WH\,N^2\big)\) coarse Babbage (not \(WH N^3\)) |
| clip-grid per sample | barycentric dens\((u)\) from atlas |

## Controls

- Drag to orbit · scroll to zoom · right-drag to pan
- **steps** — raymarch samples along the ray
- **Path C** — Chebyshev nodes for \(T\), quadrature for \(\int\sigma T\)
- **clip-grid** — NDC dens atlas; HUD shows `gpu-babbage` / `cpu-babbage` / `babbage+gpu` and time
