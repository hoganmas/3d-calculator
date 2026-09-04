---
name: memory-check
description: Run and interpret the laplaci JS heap / DOM / listener leak-check (web/scripts/measure-memory.mjs). Use when investigating reported performance degradation over long sessions, before/after a perf optimization pass, or to verify a leak fix.
---

# Memory / leak check (laplaci web app)

This resumes the memory-leak investigation workflow from the mobile
performance pass — see `web/docs/known-issues.md` for the confirmed,
not-yet-fixed leak this was built to track.

## Running it

```bash
cd web
npm run test:memory:quick   # ~1 min sanity check the harness + a fast read
npm run test:memory         # ~10 min, the statistically meaningful run
```

Both are fully self-contained: they spawn their own Vite dev server on a
free port (5190+) and a headless Chromium via Playwright, so no manual
`npm run dev` step is needed first. They clean up the server/browser on
exit even on failure.

What it does each run: loads the app at a 390×844 (mobile) viewport,
confirms the default animating parameter is actually playing, then every
`--interval-sec` (default 10s) forces two GC passes and records JS heap
used, DOM node count, and event-listener count via CDP
`Performance.getMetrics`. Every ~30s it also nudges the viewport width and
cycles the sidebar panel collapsed/expanded once, to exercise the
resize/mount-unmount paths a long real session hits repeatedly on top of
continuous animation. At the end it prints a summary with a linear-regression
estimate of MB/min heap growth.

Useful flags: `--duration-sec N`, `--interval-sec N` (see the script header).

## Reading the result — do NOT use `performance.memory` for this

The first attempt at this test used `performance.memory.usedJSHeapSize`
directly and got the *exact same byte count* (`68000000`) on every one of
50 samples across ~20 minutes — that's Chrome's anti-fingerprinting fuzzing
of that API, not evidence of no leak. This script reads `JSHeapUsedSize`
from CDP `Performance.getMetrics` instead, which is not fuzzed and gave
real, distinct values every sample. If you ever add a new metric or write a
one-off variant of this test, use CDP metrics or `HeapProfiler`, never the
JS-exposed `performance.memory`, for anything you intend to trend over time.

Interpretation, per field:
- **Heap climbing steadily and not plateauing after forced GC** → a real
  leak (retained JS/GPU-resource objects, not just uncollected garbage).
  This is what we found: ~1.7MB/min sustained over a 10-minute run.
- **DOM node count trending up** (not just fluctuating in a stable band) →
  a DOM leak (e.g. detached nodes still referenced). Not observed so far.
- **Event listener count trending up** → classic listener-accumulation bug
  (e.g. a resize/visibilitychange listener re-added without removing the
  old one). Not observed so far — stayed exactly flat across every sample.

## If a leak shows up: narrowing it down

`measure-memory.mjs` tells you *whether* and *how fast*, not *where*. The
leading suspects identified so far (not yet confirmed) are the per-tick
render/fit pipeline that runs continuously while an animation plays:
`tickSimulation()` in `web/src/app/loop.ts` → `tickGpuKeyframeBlends` /
`uploadFit` / `scheduleMarchPipelines` in `web/src/app/pipeline.ts`, driving
Chebyshev/Lobatto fitting in `web/src/math/fit.ts` and
`web/src/model/keyframes.ts`. Three.js needs explicit `.dispose()` on
geometries/materials/textures/GPU buffers — plain JS GC won't free that
class of resource, so a missing dispose call on a per-tick refit is the
prime suspect for exactly this "steady, GC-resistant growth" signature.

To actually pinpoint the source rather than guess from code reading, the
next step is a CDP heap-snapshot diff: take two full
`HeapProfiler.takeHeapSnapshot` snapshots several minutes apart during a
run like this one, and compare retained size by constructor. This hasn't
been built yet — do it if/when someone commits to root-causing this rather
than just tracking the trend.

## Recording findings

Update `web/docs/known-issues.md` with any new measurement, especially
before/after numbers from an optimization pass or fix attempt — that file
is the running record, not this skill. `gh` was not authenticated in this
repo when this was set up, so findings go there as a note rather than a
GitHub issue; check `gh auth status` and file a real issue instead if it's
available now.

## On CI

There's no PR-gating CI in this repo currently — only a `Deploy GitHub
Pages` workflow (`.github/workflows/*.yml`) triggered on push to `main`,
running the fast `npm test` suite before build/deploy. Don't wire
`test:memory` into that as a blocking step while the leak is unfixed: it
has no pass/fail threshold (any threshold would either fail every deploy or
be meaningless), and even `--quick` adds real runtime to every deploy for a
result nobody's positioned to act on yet. Once the leak is actually fixed,
converting `test:memory:quick` into a real regression gate (assert heap
growth stays under some MB/min bound) becomes worthwhile. Until then, if
visibility without gating is wanted, a `workflow_dispatch`-only or scheduled
job that runs `test:memory:quick` and uploads its output as a log/artifact
is the lower-risk option — ask before adding it, since it's a shared CI
change.
