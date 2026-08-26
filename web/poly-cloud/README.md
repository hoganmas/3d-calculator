# Poly cloud

Multi-expression 3D polynomial calculator (Three.js + WebGPU).

**Pipeline:** Fit each expression with a Chebyshev polynomial → IDCT dens volume → draw manifolds (opaque isosurfaces) then density clouds (Beer–Lambert), with shared free parameters.

Details: [`research/poly/notes/cheb-idct-volume.md`](../../research/poly/notes/cheb-idct-volume.md).

## Run

```bash
cd web/poly-cloud
npm install && npm run dev
```

Requires a WebGPU-capable browser for the full multi-layer / manifold path. Without WebGPU, a single summed dens volume still marches in WebGL.

## Controls

- **Expressions** — Enter adds a row; badge cycles auto / density / manifold; color swatch per row. `A=B` → manifold; bare / `f=…` → density.
- **Parameters** — free symbols shared across expressions (animate with ▶).
- **poly deg / scale / steps / box size** — fit order, dens opacity, march samples, domain edge length.
- **march downscale** — internal march resolution (1×–16×); CSS upscales the result.

## Cost

| Stage | When |
|---|---|
| Fit + IDCT | Expression / deg / box / param change |
| March | Every frame (camera / hyperparams) |
