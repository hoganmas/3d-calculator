# Polynomial density cloud viewer (Three.js / WebGPU)

**Golden path — clip-grid:** Fit → Chebyshev coeffs → **IDCT dens volume** (box grid) → **Beer–Lambert raymarch**. Bake is view-independent; camera only affects march.

Details: [`research/poly/notes/cheb-idct-volume.md`](../../research/poly/notes/cheb-idct-volume.md).

1. Fit `f(x,y,z)` with a 3D Chebyshev polynomial → coeffs \(c_{ijk}\) (+ world monomials for LOS mode).
2. **clip-grid:** separable IDCT → dens on Chebyshev-root lattice \(M^3\) (\(M=N{+}1\)). March samples the volume (trilinear in Cheb-index space).
3. **LOS raymarch (optional):** per-pixel nested Horner on monomials → Beer (reference).

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

## Cost

| Stage | Cost |
|---|---|
| Fit | Chebyshev DCT once per expression |
| IDCT bake | \(O(N^3\log N)\) on coeff change only |
| March | \(O(\mathrm{steps})\) volume samples per pixel |

## Controls

- Drag to orbit · scroll to zoom · right-drag to pan
- **mode** — prefer **clip-grid**; LOS raymarch is a secondary reference
- **Expression kind** — bare expr or `f(…)=E` → Beer volume; constraint `A=B` → isosurface of `A−B=0` (auto)
- **Parameters** — free symbols besides spatial `x,y,z` / `r,θ,φ,ρ` (e.g. `a` in `\exp(-r^2/a^2)`) get sliders; ▶ animates between min/max (cosine), **Hz** is cycles/sec. Changing a param refits the Chebyshev approximant.
- **Dens tile** — auto shrinks with projected box size; `exact` skips interpolation
- **steps** / **march downscale** — Beer samples and internal march resolution
