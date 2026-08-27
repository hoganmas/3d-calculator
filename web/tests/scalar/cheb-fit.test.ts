/**
 * Cloud (density) pipeline: Chebyshev fit → IDCT volume vs analytic field.
 */
import { compileExpr, fitChebyshev3D } from "../../src/math/fit.ts";
import { idctCheb3D } from "../../src/math/idct.ts";
import { assert } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("scalar / cheb-fit (cloud)", [
    {
      name: "quadratic sphere relative L2",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const fit = fitChebyshev3D(fn, 1, 8, { skipMono: true });
        assert(fit.fitRelL2 < 0.02, `fitRelL2 too high: ${fit.fitRelL2}`);
      },
    },
    {
      name: "IDCT volume matches analytic on Cheb grid",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const deg = 8;
        const fit = fitChebyshev3D(fn, half, deg, { skipMono: true });
        const { dens, M } = idctCheb3D(fit.cheb, deg, deg + 1);
        let maxErr = 0;
        for (let ix = 0; ix < M; ix++) {
          const x = chebWorld(ix, M, half);
          for (let iy = 0; iy < M; iy++) {
            const y = chebWorld(iy, M, half);
            for (let iz = 0; iz < M; iz++) {
              const z = chebWorld(iz, M, half);
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
      name: "Gaussian blob relative L2",
      fn: () => {
        const fn = compileExpr(String.raw`\exp(-(x^2+y^2+z^2))`).bind({});
        const fit = fitChebyshev3D(fn, 1, 10, { skipMono: true });
        assert(fit.fitRelL2 < 0.05, `fitRelL2 too high: ${fit.fitRelL2}`);
      },
    },
  ]);
}
