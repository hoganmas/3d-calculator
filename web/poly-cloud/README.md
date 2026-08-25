# Polynomial density cloud viewer (Three.js)

1. Fit `f(x,y,z)` with a 3D Chebyshev polynomial, convert to a **world monomial tensor** \(c_{ijk}\).
2. Shader evaluates density with **nested Horner** at world sample points (stable on long corner rays).
3. Raymarch / Path C both use clamped \(\sigma=\max(0,s\,f)\). Path C samples \(T=\exp(-\tau)\) on Chebyshev nodes, then Gauss–Chebyshev \(\int\sigma T\).

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
| Per sample | \(O((N+1)^3)\) nested Horner |

## Controls

- Drag to orbit · scroll to zoom · right-drag to pan
- **steps** — raymarch samples along the ray
- **Path C** — Chebyshev nodes for \(T\), quadrature for \(\int\sigma T\)
