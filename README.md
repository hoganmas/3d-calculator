# Laplacian

Multi-expression **3D polynomial calculator** in the browser. Enter math with MathLive, fit each expression to a Chebyshev polynomial, IDCT to a density volume, then ray-march manifolds (isosurfaces) and density clouds (Beer–Lambert) with shared free parameters.

**Live demo:** [hoganmas.github.io/3d-calculator](https://hoganmas.github.io/3d-calculator/)

## Quick start

```bash
cd web
npm install
npm run dev
```

Build for production (same output GitHub Pages deploys):

```bash
cd web
npm run build
npm run preview
```

Requires a WebGPU-capable browser for the full multi-layer / manifold path. Without WebGPU, a single summed density volume still marches in WebGL.

**MCP setup:** click **Setup MCP** in the app header — see [web/README.md — MCP setup](web/README.md#mcp-setup).

## Architecture

```
MathLive expressions → Chebyshev fit → IDCT dens volume → WebGPU march (WebGL fallback)
```

Pipeline notes: [`research/poly/notes/cheb-idct-volume.md`](research/poly/notes/cheb-idct-volume.md).

## Layout

```
web/                Laplacian app — see [web/STRUCTURE.md](web/STRUCTURE.md)
research/
  gaussian/         Gaussian-mixture transmittance experiments (Python)
  poly/             Polynomial density / transmittance track (Python + notes)
```

## Research scripts

Optional Python experiments (not required to run the site):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python research/gaussian/validate_approx.py
.venv/bin/python research/gaussian/blend_benchmark.py
.venv/bin/python research/poly/validate_poly.py
```

Results write under each package’s `results/` folder.

## License

MIT — see [LICENSE](LICENSE).
