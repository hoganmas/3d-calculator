# Laplacian

Multi-expression 3D polynomial calculator (Three.js + WebGPU).

**Live demo:** [hoganmas.github.io/3d-calculator](https://hoganmas.github.io/3d-calculator/)

**Pipeline:** Fit each expression with a Chebyshev polynomial → IDCT dens volume → draw manifolds (opaque isosurfaces) then density clouds (Beer–Lambert), with shared free parameters.

Details: [`research/poly/notes/cheb-idct-volume.md`](../research/poly/notes/cheb-idct-volume.md).

## Run locally

```bash
npm install
npm run dev
```

## Build

Production build uses `base: /3d-calculator/` for GitHub Pages project hosting:

```bash
npm run build
npm run preview
```

GitHub Actions (`.github/workflows/pages.yml`) deploys `dist/` on push to `main`. Enable **Settings → Pages → Build and deployment → GitHub Actions** once if needed.

Requires a WebGPU-capable browser for the full multi-layer / manifold path. Without WebGPU, a single summed dens volume still marches in WebGL.

## Controls

- **Expressions** — Enter splits / adds a row. Free symbols get a slider on that same row; write `a=…` for an explicit parameter equation. `A=B` → manifold; bare / `f=…` → density.
- **Settings** (gear) — appearance (theme), preset, poly deg / scale / steps / box size, march downscale, diagnostics.
- **Reset view** (recenter icon) — restores the default camera.

## Cost

| Stage | When |
|---|---|
| Fit + IDCT | Expression / deg / box / param change |
| March | Every frame (camera / hyperparams) |
