# Polynomial density cloud viewer (Three.js)

1. Fit `f(x,y,z)` with a 3D Chebyshev polynomial, convert to a **world monomial tensor** \(c_{ijk}\).
2. **LOS modes** (raymarch / Path C): per-pixel nested Horner → univariate \(\gamma(u)\), then march / Chebyshev-\(T\).
3. **clip-grid**: per-view bake of segment fibers \(\gamma(u)\) on an NDC atlas (WebGPU compute when available, else CPU), bilinear sample + Horner march. See `research/poly/notes/clip-space-babbage.md`.

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

Requires a browser with **WebGPU** for the fast clip-grid bake (`navigator.gpu`). Without it, clip-grid falls back to the CPU baker automatically.

## Cost

| Stage | Cost |
|---|---|
| Fit (CPU, once) | Chebyshev + monomial convert |
| LOS per ray | \(\Theta(N^3)\) compose + \(\Theta(\mathrm{steps}\cdot 3N)\) |
| clip-grid bake (per view) | \(\Theta(WHN^3)\) nested compose → **resident** GPU atlas (WebGPU), else CPU |
| clip-grid per sample | \(\Theta(3N)\) Horner from atlas |

## Controls

- Drag to orbit · scroll to zoom · right-drag to pan
- **steps** — raymarch samples along the ray
- **Path C** — Chebyshev nodes for \(T\), quadrature for \(\int\sigma T\)
- **clip-grid** — NDC γ atlas; HUD shows `webgpu` / `cpu` bake backend and time
