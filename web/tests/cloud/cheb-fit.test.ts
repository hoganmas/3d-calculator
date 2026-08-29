/**
 * Cloud (density) pipeline: Chebyshev fit → IDCT volume vs analytic field.
 */
import { compileExpr, fitChebyshev3D, fitScalarField } from "../../src/math/fit.ts";
import { idctCheb3D } from "../../src/math/idct.ts";
import { assert } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("cloud / cheb-fit", [
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
    {
      name: "fitChebyshev3D builds monomial basis when skipMono false",
      fn: () => {
        const fn = compileExpr("x").bind({});
        const fit = fitChebyshev3D(fn, 1, 4, { skipMono: false, skipL2: true });
        assert(!!fit.mono && fit.mono.length > 0, "mono coeffs");
      },
    },
    {
      name: "fitChebyshev3D computes relative L2 with monomial basis",
      fn: () => {
        const fn = compileExpr("x^2").bind({});
        const fit = fitChebyshev3D(fn, 1, 6, { skipMono: false, skipL2: false });
        assert(Number.isFinite(fit.fitRelL2), "fitRelL2");
        assert(!!fit.mono, "mono");
      },
    },
    {
      name: "fitScalarField default path IDCTs plain scalar",
      fn: () => {
        const compiled = compileExpr("x^2+y^2+z^2");
        const fit = fitScalarField(compiled, compiled.bind({}), 1, 6, { skipMono: true });
        assert(fit.dens.length > 0 && fit.M > 0, "volume baked");
      },
    },
  ]);
}
