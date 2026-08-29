/**
 * Chebyshev–Lobatto nested-node fit: round-trip, nesting reuse, vs Gauss roots.
 */
import { compileExpr, fitChebyshev3D } from "../../src/math/fit.ts";
import {
  fitChebyshevLobatto3D,
  fitChebyshevLobattoProgressive,
  idctLobatto3D,
  lobattoDCT1D,
  lobattoIDCT1D,
  lobattoNodes,
  lobattoWorld,
  lobattoLadderDegrees,
  refineLobatto3D,
  ensureLobattoDegree,
  beginLobattoBuild,
  stepLobattoBuild,
  finishLobattoBuild,
} from "../../src/math/chebLobatto.ts";
import { idctCheb3D } from "../../src/math/idct.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function densIndex(ix: number, iy: number, iz: number, M: number): number {
  return ix + iy * M + iz * M * M;
}

export async function run() {
  return runSuite("cloud / cheb-lobatto", [
    {
      name: "1D Lobatto DCT round-trip",
      fn: () => {
        const n = 9;
        const vals = new Float64Array(n);
        for (let j = 0; j < n; j++) vals[j] = Math.sin(j * 0.7) + 0.3 * j;
        const coeff = lobattoDCT1D(vals);
        const back = lobattoIDCT1D(coeff, n);
        let maxErr = 0;
        for (let j = 0; j < n; j++) maxErr = Math.max(maxErr, Math.abs(back[j]! - vals[j]!));
        assert(maxErr < 1e-10, `1D round-trip max err ${maxErr}`);
      },
    },
    {
      name: "Lobatto nodes nest when doubling degree",
      fn: () => {
        const deg4 = lobattoNodes(4);
        const deg8 = lobattoNodes(8);
        for (let j = 0; j <= 4; j++) {
          const nested = deg8[j * 2]!;
          assert(Math.abs(nested - deg4[j]!) < 1e-15, `node ${j}: ${nested} vs ${deg4[j]}`);
        }
      },
    },
    {
      name: "3D Lobatto IDCT matches samples on grid",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const deg = 8;
        const fit = fitChebyshevLobatto3D(fn, half, deg, { skipL2: true });
        const { dens, M } = idctLobatto3D(fit.cheb, deg, deg + 1);
        let maxErr = 0;
        for (let ix = 0; ix < M; ix++) {
          const x = lobattoWorld(ix, deg, half);
          for (let iy = 0; iy < M; iy++) {
            const y = lobattoWorld(iy, deg, half);
            for (let iz = 0; iz < M; iz++) {
              const z = lobattoWorld(iz, deg, half);
              const truth = fn(x, y, z);
              const approx = dens[densIndex(ix, iy, iz, M)]!;
              maxErr = Math.max(maxErr, Math.abs(approx - truth));
            }
          }
        }
        assert(maxErr < 0.05, `max grid error ${maxErr}`);
      },
    },
    {
      name: "refineLobatto3D reuses samples and matches full fit",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const coarse = fitChebyshevLobatto3D(fn, half, 4, { skipL2: true }).lobatto;
        const refined = refineLobatto3D(coarse, fn, 8);
        const full = fitChebyshevLobatto3D(fn, half, 8, { skipL2: true });

        assert(refined.reusedSamples === 5 * 5 * 5, `reused ${refined.reusedSamples}, expected 125`);
        assert(refined.newSamples === 9 * 9 * 9 - 125, `new ${refined.newSamples}, expected 604`);

        let maxCoeffDiff = 0;
        for (let i = 0; i < refined.cheb.length; i++) {
          maxCoeffDiff = Math.max(maxCoeffDiff, Math.abs(refined.cheb[i]! - full.cheb[i]!));
        }
        assert(maxCoeffDiff < 1e-10, `coeff diff after refine ${maxCoeffDiff}`);
      },
    },
    {
      name: "progressive ladder reaches target degree",
      fn: () => {
        const fn = compileExpr(String.raw`\exp(-(x^2+y^2+z^2))`).bind({});
        const steps: number[] = [];
        const state = fitChebyshevLobattoProgressive(fn, 1, 16, (s) => steps.push(s.deg));
        assert(state.deg === 16, `final deg ${state.deg}`);
        assert(steps.join(",") === "4,8,16", `steps ${steps.join(",")}`);
      },
    },
    {
      name: "ensureLobattoDegree matches full fit",
      fn: () => {
        const fn = compileExpr(String.raw`\exp(-(x^2+y^2+z^2))`).bind({});
        const half = 1;
        let cache = null;
        cache = ensureLobattoDegree(cache, fn, half, 4);
        cache = ensureLobattoDegree(cache, fn, half, 8);
        const full = fitChebyshevLobatto3D(fn, half, 8, { skipL2: true }).lobatto;
        let maxDiff = 0;
        for (let i = 0; i < cache.cheb.length; i++) {
          maxDiff = Math.max(maxDiff, Math.abs(cache.cheb[i]! - full.cheb[i]!));
        }
        assert(maxDiff < 1e-10, `cached refine diff ${maxDiff}`);
      },
    },
    {
      name: "lobattoLadderDegrees",
      fn: () => {
        assert(lobattoLadderDegrees(16).join(",") === "4,8,16", "pow2");
        assert(lobattoLadderDegrees(20).join(",") === "4,8,16,20", "non-pow2");
        assert(lobattoLadderDegrees(3).join(",") === "3", "small");
      },
    },
    {
      name: "chunked Lobatto build matches full refine",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const coarse = fitChebyshevLobatto3D(fn, half, 16, { skipL2: true }).lobatto;
        const begun = beginLobattoBuild(coarse, half, 32);
        assert(!!begun.job, "expected refine job");
        assert(begun.job!.mode === "refine", "refine mode");
        const chunked = finishLobattoBuild(begun.job!, fn);
        const full = refineLobatto3D(coarse, fn, 32);
        let maxDiff = 0;
        for (let i = 0; i < chunked.cheb.length; i++) {
          maxDiff = Math.max(maxDiff, Math.abs(chunked.cheb[i]! - full.cheb[i]!));
        }
        assert(maxDiff < 1e-10, `chunked refine diff ${maxDiff}`);
      },
    },
    {
      name: "stepLobattoBuild respects time budget",
      fn: () => {
        const fn = compileExpr(String.raw`\exp(-(x^2+y^2+z^2))`).bind({});
        const half = 1;
        const coarse = fitChebyshevLobatto3D(fn, half, 16, { skipL2: true }).lobatto;
        let job = beginLobattoBuild(coarse, half, 32).job!;
        let steps = 0;
        while (job) {
          const step = stepLobattoBuild(job, fn, { budgetMs: 1 });
          job = step.job!;
          steps++;
          if (step.done) break;
          assert(steps < 5000, "too many chunk steps");
        }
        assert(steps > 1, "expected multiple chunks for deg 16→32");
      },
    },
    {
      name: "Lobatto accuracy comparable to Gauss roots on smooth field",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const deg = 8;
        const gauss = fitChebyshev3D(fn, half, deg, { skipMono: true, skipL2: true });
        const lobatto = fitChebyshevLobatto3D(fn, half, deg, { skipL2: true });
        const gVol = idctCheb3D(gauss.cheb, deg, deg + 1);
        const lVol = idctLobatto3D(lobatto.cheb, deg, deg + 1);

        let gaussMax = 0;
        let lobattoMax = 0;
        const M = deg + 1;
        for (let ix = 0; ix < M; ix++) {
          const x = lobattoWorld(ix, deg, half);
          for (let iy = 0; iy < M; iy++) {
            const y = lobattoWorld(iy, deg, half);
            for (let iz = 0; iz < M; iz++) {
              const z = lobattoWorld(iz, deg, half);
              const truth = fn(x, y, z);
              lobattoMax = Math.max(
                lobattoMax,
                Math.abs(lVol.dens[densIndex(ix, iy, iz, M)]! - truth),
              );
            }
          }
        }
        // Gauss grid uses roots, not Lobatto — compare on same Lobatto nodes via volume interpolation
        for (let ix = 0; ix < M; ix++) {
          const x = lobattoWorld(ix, deg, half);
          for (let iy = 0; iy < M; iy++) {
            const y = lobattoWorld(iy, deg, half);
            for (let iz = 0; iz < M; iz++) {
              const z = lobattoWorld(iz, deg, half);
              const truth = fn(x, y, z);
              // Nearest Gauss-root grid sample (rough)
              const u = x / half;
              const gi = Math.min(gVol.M - 1, Math.round(((u + 1) / 2) * (gVol.M - 1)));
              gaussMax = Math.max(
                gaussMax,
                Math.abs(gVol.dens[densIndex(gi, gi, gi, gVol.M)]! - truth),
              );
            }
          }
        }
        assert(lobattoMax < 0.05, `Lobatto grid max err ${lobattoMax}`);
        assert(lobattoMax < gaussMax * 2 + 0.01, `Lobatto ${lobattoMax} vs Gauss ${gaussMax}`);
      },
    },
  ]);
}
