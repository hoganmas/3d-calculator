# Polynomial density cloud viewer (Three.js / WebGPU)

**Golden path — clip-grid:** Fit → world monomials → per-view **GPU Chebyshev + Clenshaw** dens atlas → **Beer–Lambert raymarch**. Transmittance is numerical, not analytic.

Details: [`research/poly/notes/clip-space-babbage.md`](../../research/poly/notes/clip-space-babbage.md).

1. Fit `f(x,y,z)` (MathLive LaTeX → Compute Engine) with a 3D Chebyshev polynomial → **world monomial tensor** \(c_{ijk}\).
2. **clip-grid (preferred):** bake dens at Chebyshev-root \(t_j\) (view-fixed window); fill screen \(x\) with Chebyshev nodes + Clenshaw (WebGPU f32). March: bilinear \((x,y)\) → u-DCT + Clenshaw → Beer F2B. CPU f64 Babbage + WebGL atlas if no WebGPU.
3. **LOS raymarch (optional):** per-pixel nested Horner → \(\gamma(u)\) → Beer march (reference / debug).

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

Requires **WebGPU** (`navigator.gpu`) for GPU dens bake + Beer march. Without it, clip-grid falls back to CPU bake + WebGL `DataTexture` march.

## Cost

| Stage | Cost |
|---|---|
| Fit (CPU, once) | Chebyshev + monomial convert |
| clip-grid bake | \(\sim O\!\big((W/\mathrm{tile})\,H\cdot(D{+}1)\,N^3 + WH\,D\big)\) Cheb+Clenshaw (not \(WH N^3\)) |
| clip-grid march | \(O(M^2 + \mathrm{steps}\,D)\) u-DCT + Clenshaw + Beer per pixel |
| LOS per ray | \(\Theta(N^3)\) compose + Beer march |

## Controls

- Drag to orbit · scroll to zoom · right-drag to pan
- **mode** — prefer **clip-grid**; LOS raymarch is a secondary reference
- **Parameters** — free symbols besides `x,y,z` (e.g. `a` in `\exp(-(x^2+y^2+z^2)/a^2)`) get sliders; ▶ animates between min/max (cosine), **Hz** is cycles/sec. Changing a param refits the Chebyshev approximant.
- **Dens tile** — auto shrinks with projected box size; `exact` skips interpolation
- **steps** — raymarch samples along the ray (Beer in clip-grid and LOS)
