# Polynomial density cloud viewer (Three.js)

1. Fit `f(x,y,z)` with a 3D Chebyshev polynomial, convert to a **world monomial tensor** \(c_{ijk}\).
2. **LOS modes** (raymarch / Path C): per-pixel nested Horner → univariate \(\gamma(u)\), then march / Chebyshev-\(T\).
3. **clip-grid**: per-view bake of fiber coeffs \(\alpha_m(\mathrm{ndc})\) on the pixel grid (Babbage along rows); GPU Horner in ray parameter \(s\). See `research/poly/notes/clip-space-babbage.md`.

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

## Cost

| Stage | Cost |
|---|---|
| Fit (CPU, once) | Chebyshev + monomial convert |
| LOS per ray | \(\Theta(N^3)\) compose + \(\Theta(\mathrm{steps}\cdot 3N)\) |
| clip-grid bake (per view) | bivariate \(\alpha_m\) + row diffs → \(W\times H\times(3N+1)\) |
| clip-grid per sample | \(\Theta(3N)\) Horner from atlas |

## Controls

- Drag to orbit · scroll to zoom · right-drag to pan
- **steps** — raymarch samples along the ray
- **Path C** — Chebyshev nodes for \(T\), quadrature for \(\int\sigma T\)
- **clip-grid** — NDC fiber atlas; HUD shows bake time (camera motion rebakes)
