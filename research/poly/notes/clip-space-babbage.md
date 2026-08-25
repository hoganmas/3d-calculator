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

**Normalized dens seeds:** GPU evaluates at box coords \(\xi=p/\mathrm{half}\in[-1,1]^3\) with
pre-scaled \(\hat c_{ijk}=c_{ijk}\,\mathrm{half}^{i+j+k}\). Seeds **outside** the fit box are
set to 0 *before* Chebyshev/Newton — exterior \(\|p\|^N\) (far off-axis at \(t\sim t_\mathrm{mid}\))
must not enter the DCT/Δ table. Screen indexing is NDC; fiber depth is Chebyshev \(u\in(-1,1)\).

**Far-camera ray points:** dens seeds avoid `p = ro + t·rd` in f32. CPU uploads
`anchor = ro + tMid·rdCenter` (f64), and the GPU uses
`p = anchor + tMid·(rd−rdC) + (t−tMid)·rd`.

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
| Bake (Cheb+Clenshaw) | \(\sim O\big((W/\mathrm{tile})\,H\cdot(D{+}1)\cdot N^3 + WH\cdot D\big)\) — not \(WH\cdot N^3\); **auto** tile shrinks with projected box size when far |
| Bake (exact, narrow tile) | \(O(WH\cdot n_\alpha\cdot N^3)\) on GPU |
| March | \(O(\mathrm{steps})\) dens lookups + Beer updates per pixel |

---

## Modes in the viewer

| Mode | Role |
|------|------|
| **clip-grid** | Golden path (this note) |
| raymarch (LOS \(\gamma\)) | Debug / reference: per-ray Horner + Beer |

Archived Chebyshev-\(\hat\tau\) Path C design notes: [`path-c.md`](./path-c.md) (removed from the explorer).

---

## Research target (not shipped): camera-clip homogenization

Seeds are now **fit-box normalized** (\(\xi=p/\mathrm{half}\)), not camera NDC/clip.
Homogeneous pullback through \(PV\) (dens as \(G(X,Y,Z,W)/W^{\cdot}\) or NDC rational)
and 2D/1D differences on fiber coeffs remain a **target architecture**.

---

## One-paragraph abstract (shipped)

*For realtime polynomial density volumes, the poly-cloud explorer fits a world monomial field once, then each view materializes a screen-space atlas of density samples at fixed Chebyshev ray-\(t\) nodes using tile-parallel Chebyshev interpolation in \(x\) with Clenshaw evaluation (GPU), and composites with standard Beer–Lambert raymarching.*
