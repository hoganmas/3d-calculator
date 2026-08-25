# Clip-grid dens bake (middle-out Babbage) + Beer raymarch

**Golden path** for `web/poly-cloud`: fit a world polynomial density once, then each view bake an NDC dens atlas with **tile-parallel middle-out Babbage** (GPU f32, CPU f64 fallback) and **numerically raymarch** with Beer–Lambert. Transmittance is **not** closed-form and **not** Path C.

Legacy Path C (Chebyshev \(\hat\tau\)): [`path-c.md`](./path-c.md).

---

## Shipped pipeline

```
Fit (CPU) → world monomials c_ijk
                │
                ▼  camera change / LOD
     GPU compute: middle-out Babbage dens atlas
       (seeds → forward Δ → Newton fill; high N → exact dens)
                │
                ▼
     Fullscreen march: bilinear dens(u) + Beer F2B
```

| Stage | What | Where |
|-------|------|--------|
| Fit | 3D Chebyshev → world monomials | `fit.js` |
| Bake | Dens at Chebyshev-root \(t_j\) on a view-fixed fiber window; middle-out coarse lattice + Newton (or exact at high deg) | `clipBakeGpu.js` / `clipGrid.js` |
| March | Box intersect → sample atlas dens along ray → \(T\leftarrow T\,e^{-\sigma\,ds}\) | `clipBakeGpu.js` march / `clipShaders.js` |

Atlas layout: plane \(j\) is dens at \(t_j = t_\mathrm{mid} + t_\mathrm{hw}\,u_j\) with \(u_j\) Chebyshev roots on \((-1,1)\). Screen samples bilinear in \((x,y)\); along the ray, piecewise-linear in \(u\).

### Middle-out Babbage (bake)

For each tile-row segment:

1. Choose coarse step \(h\) so \(D\cdot h\) covers the span (\(D=3N\)).
2. Center the \(D{+}1\) seed pixels in the tile (middle-out).
3. Exact world `evalMonomial3D` at seeds → forward differences → Newton series for all pixels in the tile.
4. Zero / cap samples whose world point lies outside the fit box (Runge / f32 safety).

**GPU f32:** high-order \(\Delta\) is fragile. Tile width shrinks with degree (`gpuBabbageTile`); at \(N\ge 6\) the bake uses **exact dens** per pixel (still parallel, no Newton). Clamp the coarse stencil into the atlas so out-of-frame seeds cannot poison \(\Delta\). Prefer arithmetic floor-halve for signed mid offsets (WGSL `/` truncates toward zero).

**CPU fallback:** f64 Babbage (tile≈256) + upload / WebGL `DataTexture` if WebGPU is missing.

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
| Bake (Babbage) | \(\sim O\big((W/\mathrm{tile})\,H\cdot(D{+}1)\cdot N^3 + WH\cdot D^2\big)\) — not \(WH\cdot N^3\) |
| Bake (exact, high \(N\)) | \(O(WH\cdot n_\alpha\cdot N^3)\) on GPU |
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

The original writeup aimed at **homogeneous** pullback to clip and 2D/1D differences on \(\beta_k(x,y)\) / depth fibers without per-ray \(\Omega(N^3)\) contraction. That remains a **target architecture**; the explorer instead bakes **Cartesian dens samples** in ray-\(t\) with screen-space Babbage.

Useful ideas still on the table:

- Homogenize \(F\), pull back \(G=F\circ P^{-1}\) once per view.
- 2D Babbage on bivariate fiber coeffs; optional 1D differences in depth.
- Dehomogenize \(F/W^n\) at sample time or fold into path weights — do **not** dehomogenize the tensor into a Cartesian NDC polynomial for physical \(f\) under perspective (Möbius / rational mess).

Float32 difference engines at high effective degree remain the main numerical caution (already observed in the shipped dens bake).

---

## One-paragraph abstract (shipped)

*For realtime polynomial density volumes, the poly-cloud explorer fits a world monomial field once, then each view materializes a screen-space atlas of density samples at fixed Chebyshev ray-\(t\) nodes using tile-parallel middle-out Babbage (GPU), and composites with standard Beer–Lambert raymarching. Path C Chebyshev transmittance is an optional legacy LOS path, not part of this dens-atlas pipeline.*
