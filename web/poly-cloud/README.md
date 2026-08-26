# Laplacian

Multi-expression 3D polynomial calculator (Three.js + WebGPU). Package folder remains `poly-cloud`.

**Pipeline:** Fit each expression with a Chebyshev polynomial → IDCT dens volume → draw manifolds (opaque isosurfaces) then density clouds (Beer–Lambert), with shared free parameters.

Details: [`research/poly/notes/cheb-idct-volume.md`](../../research/poly/notes/cheb-idct-volume.md).

## Run

```bash
cd web/poly-cloud
npm install && npm run dev
```

Requires a WebGPU-capable browser for the full multi-layer / manifold path. Without WebGPU, a single summed dens volume still marches in WebGL.

## Controls

- **Expressions** — Enter splits / adds a row. Free symbols get a slider on that same row; write `a=…` for an explicit parameter equation. `A=B` → manifold; bare / `f=…` → density.
- **Settings** (gear) — preset, poly deg / scale / steps / box size, march downscale, diagnostics.
- **Reset view** — restores the default camera.

## Cost

| Stage | When |
|---|---|
| Fit + IDCT | Expression / deg / box / param change |
| March | Every frame (camera / hyperparams) |
