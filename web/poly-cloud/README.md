# Polynomial density cloud viewer (Three.js / WebGPU)

**Golden path — clip-grid:** multi-expression Fit → IDCT volumes → **opaque manifolds**, then **density Beer** (shared free vars).

Details: [`research/poly/notes/cheb-idct-volume.md`](../../research/poly/notes/cheb-idct-volume.md).

1. Expression list (color, density vs manifold role).
2. Each expr → Chebyshev fit → IDCT dens grid.
3. Draw: constraints/isosurfaces first (opaque), then density clouds (transparent Beer).
4. **LOS raymarch (optional):** first density’s monomials (reference).

## Run

```bash
cd web/poly-cloud
npm install
npm run dev
```

## Cost

| Stage | Cost |
|---|---|
| Fit | Chebyshev DCT per expression |
| IDCT bake | per expression on coeff change |
| March | manifolds then density layers |

## Controls

- **Expressions** — Enter adds a row; badge cycles auto / density / manifold; color swatch per row
- **Parameters** — free symbols shared across all expressions
- **mode** — clip-grid (preferred) or LOS reference
- **steps** / **march downscale** — Beer samples and internal march resolution
