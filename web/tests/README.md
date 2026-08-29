# Simulation tests

Node-only correctness checks for the fit → IDCT → volume pipeline. No browser or WebGPU required.

```bash
npm test              # all suites
npm test flow          # filter by path segment
npm test cloud
npm test isosurface
npm run test:coverage  # c8 line coverage (HTML → coverage/index.html)
```

## Layout

| Folder | Primitive | What it checks |
|---|---|---|
| `app/` | compile | `compileAllExprs` orchestration (registry, warnings, params) |
| `cloud/` | cloud | Chebyshev fit + IDCT vs analytic scalar field |
| `isosurface/` | isosurface | `idctChebGrad3D` vs ∇f |
| `flow/` | flow | parse/compile, calcOps, Chebyshev vector fit |
| `math/` | — | Direct IDCT spectral operator tests |
| `symbols/` | — | Classification, params, registry |
| `expr/` | — | Expression error reporting |
| `persistence/` | — | Document schema + IndexedDB storage |
| `integration/` | — | Browser smoke tests (Playwright; needs `npm run dev`) |
| `helpers/` | — | Shared assert, grid sampling, DOM stub, suite runner |

Each `*.test.ts` exports `run(): Promise<number>` (failure count). `run.mjs` discovers and runs them via `tsx`.

Add new suites as `tests/<area>/<name>.test.ts` following the existing pattern.

## Coverage baseline

Scoped to `src/math/**`, `src/model/**`, `src/app/compile.ts`, `src/app/exprErrors.ts` (see `.c8rc.json`). Run `npm run test:coverage` to refresh.

| File | Lines | Notes |
|------|------:|-------|
| **Overall** | **76%** | 19 suites, ~170 cases |
| `math/calcOps.ts` | 87% | Direct unit tests in `flow/calcOps.test.ts` |
| `math/idct.ts` | 96% | Spectral operators in `math/idct-operators.test.ts` |
| `math/fit.ts` | 91% | Scalar compile/fit/integrals |
| `math/fitVector.ts` | 87% | Vector compile/fit/flow sim |
| `app/compile.ts` | 75% | Integration via `app/compile.test.ts` |
| `app/exprErrors.ts` | 84% | Format + collect |
| `model/symbols.ts` | 97% | Registry |
| `model/expressions.ts` | 74% | Used indirectly by compile tests |
| `model/params.ts` | 61% | Param compile chain in `symbols/params.test.ts` |
| `model/keyframes.ts` | 0% | Animation bake — not Node-testable without pipeline mock |

### Out of scope (expected low/zero coverage)

- `src/render/**` — WebGPU/WebGL (needs browser + GPU)
- `src/ui/**` — Svelte components
- `src/app/pipeline.ts`, `loop.ts`, `splash.ts`, etc. — app lifecycle / GPU upload
- WebMCP beyond `integration/webmcp.test.mjs` smoke
