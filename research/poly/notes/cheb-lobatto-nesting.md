# Chebyshev–Lobatto nested fit (experiment)

Branch: `feat/lobatto-nesting`  
Worktree: `../3dgs-approx-lobatto`

## Motivation

The shipping fit uses **Gauss–Chebyshev roots** (`cos(π(2i+1)/(2n))`). Those nodes do **not** nest when degree increases — every refinement requires a full `(n+1)³` resample.

**Chebyshev–Lobatto** nodes (`cos(πj/N)`, j = 0..N) **do nest**: degree N nodes are even-index slots in the degree 2N grid. Refining N → 2N reuses `(N+1)³` samples and only evaluates the odd-index complement.

## Implementation

| Module | Role |
|--------|------|
| `web/src/math/chebLobatto.ts` | Lobatto DCT-I, IDCT-I, full fit, `refineLobatto3D`, progressive ladder |
| `web/tests/cloud/cheb-lobatto.test.ts` | Round-trip, nesting reuse, progressive ladder, vs Gauss roots |

### API sketch

```ts
fitChebyshevLobatto3D(fn, half, deg)        // full fit
refineLobatto3D(prevState, fn, newDeg)      // nested reuse when newDeg = 2 * prev.deg
fitChebyshevLobattoProgressive(fn, half, targetDeg, onStep)
```

## Sample reuse when doubling 4 → 8

- Reused: 5³ = 125 (even/even/even indices)
- New: 9³ − 125 = 604 evaluations (~59% of full grid)

Each further doubling saves more; asymptotically ~87.5% of nodes are nested at each step.

## Open questions before shipping

1. **IDCT grid mismatch** — march pipeline expects Chebyshev-root grid; Lobatto endpoints at ±half need grid/index mapping changes in `idct.ts` and shaders.
2. **Spectral operators** — grad/laplacian/curl on Lobatto coeffs should work (same T_k basis) but need validation.
3. **Endpoint inclusion** — Lobatto includes box faces; may help boundary-heavy fields, may hurt Runge near corners for non-smooth data.
4. **Async integration** — progressive ladder + `onStep` maps naturally to `scheduleUploadFit` with generation tokens (same pattern as keyframes).

## Next steps

- [ ] Compare L2 error vs Gauss roots across preset expressions
- [ ] Wire progressive ladder into pipeline behind a flag
- [ ] Benchmark sample savings at deg 16/32/64
- [ ] Decide whether to switch IDCT grid or evaluate Lobatto coeffs on root grid for rendering
