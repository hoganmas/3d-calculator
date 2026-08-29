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

## Benchmark results (2026-08-29)

Run: `npm run bench:lobatto` (or `bench:lobatto:quick`)  
Output: `research/poly/results/lobatto_accuracy_benchmark.json`

Degrees 8–64, half=1, six expressions, 12³ probe grid.

| Finding | Detail |
|---------|--------|
| Probe L2 ratio (Lobatto/Gauss) | ~0.24–1.31; ~1.0 by deg 16+ on smooth fields |
| Grid max error | Both ~1e-7 on native grids |
| DCT-I convention | Off-grid probes need `evalLobattoChebTensor3D` (endpoint halving) |

## Pipeline integration (done)

| Module | Role |
|--------|------|
| `web/src/app/progressiveFit.ts` | Ladder scheduler, cancellation, per-layer Lobatto cache |
| `pipeline.ts` | Cloud layers → Lobatto progressive; iso/flow/operators → Gauss |

Ladder: 4 → 8 → 16 → … → target. Toggle via `USE_LOBATTO_PROGRESSIVE`.

## Animation keyframes (done)

Progressive degree ladder is integrated into [`keyframes.ts`](../3dgs-approx-lobatto/web/src/model/keyframes.ts):

- Sync-bakes current blend pair `(i0,i1)` at **deg min(4, target)** only
- Async pump uses **degree-first** scheduling: outer loop = ladder rungs (4→8→16→…), inner loop = K slots (blend pair first, one slot per frame)
- Pump runs **after each render frame** via `tickKeyframePump()` (one work unit per frame); anim ticks pass `deferSyncBake` to skip sync blend bakes
- All slots reach deg 4 before any slot advances to deg 8, etc.
- **Cloud / isosurface**: Lobatto via `bakeScalarKeyframeFrame` + per-k `lobattoByK` cache
- **Flow**: `fitVectorField` at each ladder step (no nested reuse)
- `readyCount` / `allKeyframesComplete()` count slots at **targetDeg** only

## Remaining open questions

1. **March grid** — Lobatto volume uses endpoint-inclusive grid; shaders sample in Cheb-index space (may need validation).
2. **Isosurface grad** — still Gauss-grid `idctChebGrad3D`; dens/grad grid mismatch if iso uses Lobatto.
3. **Spectral operators** on Lobatto coeffs — untested.
