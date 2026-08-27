/**
 * Isosurface pipeline: Chebyshev grad IDCT vs analytic ∇f.
 */
import { compileExpr, fitChebyshev3D } from "../../src/math/fit.ts";
import { idctChebGrad3D } from "../../src/math/idct.ts";
import { assert } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("iso / gradient", [
    {
      name: "grad of x^2+y^2+z^2 on Cheb grid",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const deg = 8;
        const fit = fitChebyshev3D(fn, half, deg, { skipMono: true });
        const { gx, gy, gz, M } = idctChebGrad3D(fit.cheb, deg, deg + 1);
        const scale = 1 / half;
        let maxErr = 0;
        for (let ix = 0; ix < M; ix++) {
          const x = chebWorld(ix, M, half);
          for (let iy = 0; iy < M; iy++) {
            const y = chebWorld(iy, M, half);
            for (let iz = 0; iz < M; iz++) {
              const z = chebWorld(iz, M, half);
              const idx = densIndex(ix, iy, iz, M);
              maxErr = Math.max(maxErr, Math.abs(gx[idx]! * scale - 2 * x));
              maxErr = Math.max(maxErr, Math.abs(gy[idx]! * scale - 2 * y));
              maxErr = Math.max(maxErr, Math.abs(gz[idx]! * scale - 2 * z));
            }
          }
        }
        assert(maxErr < 0.15, `max grad error ${maxErr}`);
      },
    },
    {
      name: "grad of x^2+y^2 (2D-like) on Cheb grid",
      fn: () => {
        const fn = compileExpr("x^2+y^2").bind({});
        const half = 1;
        const deg = 8;
        const fit = fitChebyshev3D(fn, half, deg, { skipMono: true });
        const { gx, gy, gz, M } = idctChebGrad3D(fit.cheb, deg, deg + 1);
        const scale = 1 / half;
        let maxErr = 0;
        for (let ix = 0; ix < M; ix++) {
          const x = chebWorld(ix, M, half);
          for (let iy = 0; iy < M; iy++) {
            const y = chebWorld(iy, M, half);
            for (let iz = 0; iz < M; iz++) {
              const z = chebWorld(iz, M, half);
              const idx = densIndex(ix, iy, iz, M);
              maxErr = Math.max(maxErr, Math.abs(gx[idx]! * scale - 2 * x));
              maxErr = Math.max(maxErr, Math.abs(gy[idx]! * scale - 2 * y));
              maxErr = Math.max(maxErr, Math.abs(gz[idx]! * scale));
            }
          }
        }
        assert(maxErr < 0.15, `max grad error ${maxErr}`);
      },
    },
  ]);
}
