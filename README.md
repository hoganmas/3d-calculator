# 3dgs-approx

Research sandbox for Gaussian-mixture transmittance / radiance approximations and a small realtime web splat viewer with `T < ε` early-out.

## Layout

```
research/
  gaussian/          # Appendix-A / blend / cluster / early-out Python
    notes/           # approx-3dgs.md
    results/         # JSON dumps from the scripts
  poly/              # Polynomial transmittance track
    notes/           # approx-poly.md
    curves/ results/
web/
  gsplat/            # Realtime Kerbl-style software raster (Vite + Worker)
  poly-cloud/        # Polynomial cloud viewer
```

## Quick start — realtime viewer

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
