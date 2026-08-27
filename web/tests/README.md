# Simulation tests

Node-only correctness checks for the fit → IDCT → volume pipeline. No browser or WebGPU required.

```bash
npm test              # all suites
npm test vector       # filter by path segment
npm test scalar
npm test iso
```

## Layout

| Folder | What it checks |
|---|---|
| `scalar/` | Cloud density: Chebyshev fit + IDCT vs analytic |
| `iso/` | Isosurface normals: `idctChebGrad3D` vs ∇f |
| `vector/` | Flow fields: parse/compile + Chebyshev vector fit |
| `integration/` | Browser smoke tests (Playwright; needs `npm run dev`) |
| `helpers/` | Shared assert, grid sampling, suite runner |

Each `*.test.ts` exports `run(): Promise<number>` (failure count). `run.mjs` discovers and runs them via `tsx`.

Add new suites as `tests/<area>/<name>.test.ts` following the existing pattern.
