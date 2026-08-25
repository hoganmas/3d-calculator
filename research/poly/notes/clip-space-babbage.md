# Clip-grid dens bake (Chebyshev + Clenshaw) + Beer raymarch

**Golden path** for `web/poly-cloud`: fit a world polynomial density once, then each view bake an NDC dens atlas with **tile-parallel Chebyshev nodes + Clenshaw** (GPU f32; CPU f64 Babbage fallback) and **numerically raymarch** with Beer–Lambert. Transmittance is **not** closed-form and **not** Path C.

Legacy Path C (Chebyshev \(\hat\tau\)): [`path-c.md`](./path-c.md).

---

## Shipped pipeline

```
Fit (CPU) → world monomials c_ijk
                │
                ▼  camera change / LOD
     GPU compute: dens atlas
       (Chebyshev x-nodes → DCT coeffs → Clenshaw fill;
        narrow tiles → exact dens)
                │
                ▼
     Fullscreen march: bilinear dens(u) + Beer F2B
```

| Stage | What | Where |
|-------|------|--------|
| Fit | 3D Chebyshev → world monomials | `fit.js` |
| Bake | Dens at Chebyshev-root \(t_j\) on a view-fixed fiber; screen-\(x\) fill via Chebyshev+Clenshaw (or exact) | `clipBakeGpu.js` / `clipGrid.js` |
| March | Box intersect → sample atlas dens along ray → \(T\leftarrow T\,e^{-\sigma\,ds}\) | `clipBakeGpu.js` march / `clipShaders.js` |

Atlas layout: plane \(j\) is dens at \(t_j = t_\mathrm{mid} + t_\mathrm{hw}\,u_j\) with \(u_j\) Chebyshev roots on \((-1,1)\). Screen samples bilinear in \((x,y)\); along the ray, piecewise-linear in \(u\).

### Chebyshev + Clenshaw (bake)

Along each tile row, dens\((x)\) at fixed \(t_j\) is a univariate poly of degree \(\le D=3N\):

1. Exact `evalMonomial3D` at \(D{+}1\) **Chebyshev nodes** in \(x\) on the tile.
2. Discrete Chebyshev transform → coeffs \(c_k\).
3. **Clenshaw** recurrence at every integer pixel (stable in f32 at high \(D\)).
4. Zero / cap samples outside the fit box.

Equispaced forward-\(\Delta\) / Newton (classic Babbage) is **not** used on the GPU — it loses digits for \(D\gtrsim 18\). CPU fallback still uses f64 middle-out Babbage.

Narrow tiles (\(\mathrm{span}<D\)) use exact dens (not enough nodes for a degree-\(D\) interpolant).

### March (transmittance)

Ordinary front-to-back volume integration — **raymarch**, not analytic \(T\):

\[
T \leftarrow T\,e^{-\sigma\,ds},\qquad
\sigma = \max(0, s\cdot \mathrm{dens}(u)).
\]

Steps are controlled by the UI **steps** slider. Early-out on small \(T\).

---

## Cost (sketch)

| Stage | Cost |
|-------|------|
| Fit | once per expression / degree |
| Bake (Cheb+Clenshaw) | \(\sim O\big((W/\mathrm{tile})\,H\cdot(D{+}1)\cdot N^3 + WH\cdot D\big)\) — not \(WH\cdot N^3\) |
| Bake (exact, narrow tile) | \(O(WH\cdot n_\alpha\cdot N^3)\) on GPU |
| March | \(O(\mathrm{steps})\) dens lookups + Beer updates per pixel |

---

## Modes in the viewer

| Mode | Role |
|------|------|
| **clip-grid** | Golden path (this note) |
| raymarch (LOS \(\gamma\)) | Debug / reference: per-ray Horner + Beer |
| Path C | Legacy Chebyshev-\(\hat\tau\); see [`path-c.md`](./path-c.md) |

---

## Research target (not shipped): clip homogenization

Homogeneous pullback to clip and 2D/1D differences on fiber coeffs remains a **target architecture**. The explorer bakes **Cartesian dens samples** in ray-\(t\) with screen-space Chebyshev/Clenshaw.

---

## One-paragraph abstract (shipped)

*For realtime polynomial density volumes, the poly-cloud explorer fits a world monomial field once, then each view materializes a screen-space atlas of density samples at fixed Chebyshev ray-\(t\) nodes using tile-parallel Chebyshev interpolation in \(x\) with Clenshaw evaluation (GPU), and composites with standard Beer–Lambert raymarching. Path C Chebyshev transmittance is an optional legacy LOS path, not part of this dens-atlas pipeline.*
