# Known issues

## JS heap grows continuously while an animation plays

**Status:** confirmed, not yet root-caused. Reported as gradual performance
degradation on long mobile sessions.

**Evidence:** `npm run test:memory` drives a real page for a sustained
period with an animation playing (plus periodic resize/panel-collapse
cycling) and samples JS heap, DOM node count, and event-listener count via
CDP `Performance.getMetrics` (not the fuzzed `performance.memory` API — see
below). A 10-minute run measured:

- JS heap: **27.57MB → 44.31MB** (~1.7MB/min, roughly linear, no plateau),
  measured *after* forcing two GC passes before every sample — so this is
  memory still reachable/retained, not just uncollected garbage.
- DOM node count: fluctuated in a stable range with no trend — not a DOM leak.
- Event listener count: flat at 343 for the entire run — not a listener leak.

**Leading suspects** (not yet confirmed): the per-tick render/fit pipeline
that runs continuously while an animation plays — `tickSimulation()` in
`src/app/loop.ts` calls `tickGpuKeyframeBlends` and (throttled) `uploadFit` /
`scheduleMarchPipelines` in `src/app/pipeline.ts` on every animating frame,
which drive the Chebyshev/Lobatto fitting in `src/math/fit.ts` and
`src/model/keyframes.ts`.
Three.js requires explicit `.dispose()` on geometries/materials/textures/GPU
buffers — plain JS GC does not free that class of resource, so a per-tick
refit that replaces geometry/texture objects without disposing the old ones
would show up as exactly this kind of steady, GC-resistant growth.

**Next step to actually find it:** a CDP heap-snapshot diff (two full
snapshots minutes apart, compared by retained size per constructor) would
give a definitive answer instead of a guess from code reading.

**Measurement caveat:** `test:memory` runs headless Chromium via the WebGL
fallback path — Playwright's bundled Chromium has no WebGPU support, so this
doesn't exercise the actual WebGPU renderer real phones use. It's useful for
catching a general JS/GPU-resource leak, but can't confirm or rule out
anything specific to the WebGPU code path.

**Why not `performance.memory`:** Chrome deliberately quantizes/fuzzes that
API's return value (anti-fingerprinting mitigation) — an earlier attempt
using it produced *exactly* the same rounded byte count on every single
sample across 50 samples over ~20 minutes, which was a measurement artifact,
not evidence of no leak. `test:memory` reads `JSHeapUsedSize` from CDP
`Performance.getMetrics` instead, which is not fuzzed.
