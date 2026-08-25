# Path C — Chebyshev optical-depth transmittance (archived)

> **Status:** Removed from `web/poly-cloud`. Prefer **clip-grid** (GPU dens atlas + Beer–Lambert raymarch). This note keeps the design for reference.

Related golden-path dens bake: [`clip-space-babbage.md`](./clip-space-babbage.md).

---

## What it is

**Path C** approximates transmittance along a primary ray by fitting a low-degree Chebyshev model to optical depth \(\tau\), then setting \(T=\exp(-\hat\tau)\).

Pipeline (formerly in `web/poly-cloud/src/shaders.js`, `uMode == 1`):

1. **Per pixel:** nested Horner contraction of the world monomial tensor → univariate \(\gamma(u)\) on the box segment (degree \(\le 3N\)). Cost \(\Theta(N^3)\) per ray.
2. **Trapezoid-integrate** \(\sigma(u)=\max(0,s\cdot\gamma(u))\) on a fixed \(u\)-grid to accumulate \(\tau\) at Chebyshev nodes of the first kind.
3. **Discrete Chebyshev** fit \(\hat\tau(u)=\sum_{k=0}^{T_d} c_k T_k(u)\).
4. **Front-to-back:** sample \(\hat\tau\) via Clenshaw, \(T=\exp(-\hat\tau)\), emit with \(\sigma\) from Horner on \(\gamma\).

Profile stages in the old UI (0–4) peeled this pipeline for timing: after \(\gamma\), after \(\tau\) nodes, after exit \(T\), then full F2B.

---

## Why it existed

Closed-form \(\int g\,\exp(-\int f)\) is rare for polynomial \(f\). Path C keeps \(\gamma\) exact (polynomial) and approximates only the **cumulative** \(\tau\to T\) map with a cheap Chebyshev surrogate, aiming for fewer effective samples than naive Beer marching on \(\gamma\) alone.

See also the older analytic / CDF-style transmittance notes in [`approx-poly.md`](./approx-poly.md) (different approach; research scripts under `research/poly/`).

---

## Relation to clip-grid / Babbage

| | Path C (LOS) | clip-grid (golden) |
|--|--------------|--------------------|
| Dens / \(\gamma\) | Per-ray nested Horner | Per-view GPU middle-out Babbage dens atlas |
| Transmittance | Chebyshev \(\hat\tau\), \(T=e^{-\hat\tau}\) | Numerical Beer–Lambert along the ray |
| When | Optional UI mode | Default / preferred explorer path |

Path C does **not** consume the Babbage dens atlas. Fusing Chebyshev-\(T\) on top of atlas samples was never shipped.

If revisiting analytic \(\tau\) on a clip/homogeneous fiber (e.g. integrate \(F/(a\lambda+b)^n\)), keep that discussion on the dens/fiber algebra side — do not force an early Cartesian NDC dehomogenization of the volume tensor just to match Path C. Details of that clip algebra live in [`clip-space-babbage.md`](./clip-space-babbage.md) (research target sections).

---

## Viewer hooks (removed)

Formerly: mode `pathc`, `uMode` / `uTDeg` / `uProfileStage` in `shaders.js`, **T Cheb deg** and **profile stage** controls. All removed from the explorer; LOS Beer raymarch remains as the non–clip-grid volume mode.
