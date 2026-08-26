# 3dgs-approx

Research sandbox for Gaussian-mixture transmittance / radiance approximations, plus a realtime web splat viewer and a polynomial **Laplacian** calculator (`web/poly-cloud`).

## Layout

```
research/
  gaussian/          # Appendix-A / blend / cluster / early-out Python
    notes/ results/
  poly/              # Polynomial density / transmittance track
    notes/           # cheb-idct-volume.md (poly-cloud), path-c.md, …
    curves/ results/
web/
  gsplat/            # Realtime Kerbl-style software raster (Vite + Worker)
  poly-cloud/        # Laplacian — Multi-expression Chebyshev → IDCT → iso + Beer
```

## Quick start — Laplacian (poly-cloud)

```bash
cd web/poly-cloud && npm install && npm run dev
```

Fit → IDCT volumes → manifolds + density Beer. Notes: `research/poly/notes/cheb-idct-volume.md`.

## Quick start — realtime splat viewer

```bash
.venv/bin/python web/gsplat/gen_scene.py
cd web/gsplat && npm install && npm run dev
```

Drag to orbit, scroll to zoom. Toggle ε / resolution in the panel.

## Quick start — research scripts

```bash
.venv/bin/python research/gaussian/validate_approx.py
.venv/bin/python research/gaussian/blend_benchmark.py
.venv/bin/python research/gaussian/cluster_blend.py
.venv/bin/python research/poly/validate_poly.py
```

Results write under each package’s `results/` folder.
