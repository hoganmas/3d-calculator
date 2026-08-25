# Polynomial density cloud viewer (Three.js / WebGPU)

**Golden path — clip-grid:** Fit → world monomials → per-view **GPU Chebyshev + Clenshaw** dens atlas → **Beer–Lambert raymarch**. Transmittance is numerical, not analytic.

Details: [`research/poly/notes/clip-space-babbage.md`](../../research/poly/notes/clip-space-babbage.md).  
Legacy Path C: [`research/poly/notes/path-c.md`](../../research/poly/notes/path-c.md).

1. Fit `f(x,y,z)` with a 3D Chebyshev polynomial → **world monomial tensor** \(c_{ijk}\).
2. **clip-grid (preferred):** bake dens at Chebyshev-root \(t_j\) (view-fixed window); fill screen \(x\) with Chebyshev nodes + Clenshaw (WebGPU f32). March with Beer F2B. CPU f64 Babbage + WebGL atlas if no WebGPU.
3. **LOS raymarch (optional):** per-pixel nested Horner → \(\gamma(u)\) → Beer march (reference / debug).
4. **Path C (legacy):** Horner → Chebyshev \(\hat\tau\) → \(T=e^{-\hat\tau}\); may be phased out.

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

Requires **WebGPU** (`navigator.gpu`) for GPU bake + resident-atlas march. Without it, clip-grid falls back to CPU bake + WebGL `DataTexture` march.

## Cost

| Stage | Cost |
|---|---|
| Fit (CPU, once) | Chebyshev + monomial convert |
| clip-grid bake | \(\sim O\!\big((W/\mathrm{tile})\,H\cdot(D{+}1)\,N^3 + WH\,D\big)\) Cheb+Clenshaw (not \(WH N^3\)) |
| clip-grid march | \(O(\mathrm{steps})\) dens lookup + Beer |
| LOS / Path C per ray | \(\Theta(N^3)\) compose + march / Cheb-\(T\) |

## Controls

- Drag to orbit · scroll to zoom · right-drag to pan
- **mode** — prefer **clip-grid**; LOS raymarch / Path C are secondary
- **Dens tile** — auto uses large tiles + Clenshaw; `exact` skips interpolation
- **steps** — raymarch samples along the ray (Beer in clip-grid and LOS)
- **T Cheb deg** / **profile stage** — Path C only (legacy)
