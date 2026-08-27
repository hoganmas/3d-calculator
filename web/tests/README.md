# Simulation tests

Node-only correctness checks for the fit → IDCT → volume pipeline. No browser or WebGPU required.

```bash
npm test              # all suites
npm test flow          # filter by path segment
npm test cloud
npm test isosurface
```

## Layout

| Folder | Primitive | What it checks |
|---|---|---|
| `cloud/` | cloud | Chebyshev fit + IDCT vs analytic scalar field |
| `isosurface/` | isosurface | `idctChebGrad3D` vs ∇f |
| `flow/` | flow | parse/compile + Chebyshev vector fit |
| `integration/` | — | Browser smoke tests (Playwright; needs `npm run dev`) |
| `helpers/` | — | Shared assert, grid sampling, suite runner |

Each `*.test.ts` exports `run(): Promise<number>` (failure count). `run.mjs` discovers and runs them via `tsx`.

Add new suites as `tests/<primitive>/<name>.test.ts` following the existing pattern.
