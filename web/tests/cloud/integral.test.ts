import { compileExpr, fitScalarField } from "../../src/math/fit.ts";
import { chebDefiniteInt1D, evalCheb1D } from "../../src/math/idct.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("cloud / integral", [
    {
      name: "chebDefiniteInt1D integrates x on [0,1] in ξ",
      fn: () => {
        // f(ξ) = ξ → coeffs for T_1: c_1 = 1, rest 0 (approx on [-1,1])
        const coeff = new Float64Array(4);
        coeff[1] = 1;
        const val = chebDefiniteInt1D(coeff, 0, 1);
        assertNear(val, 0.5, 0.02, "∫_0^1 ξ dξ");
      },
    },
    {
      name: "\\int_{0}^{1} 2x\\,dx evaluates to 1",
      fn: () => {
        const compiled = compileExpr(String.raw`\int_{0}^{1} 2x\,dx`);
        assert(compiled.operator === "definite_integral", "expected definite_integral");
        const fn = compiled.bind({});
        assertNear(fn(0.2, 0.3, 0.4), 1, 0.08, "∫_0^1 2x dx");
      },
    },
    {
      name: "\\int_{-1}^{1} x^2\\,dx evaluates to 2/3",
      fn: () => {
        const compiled = compileExpr(String.raw`\int_{-1}^{1} x^2\,dx`);
        const fn = compiled.bind({});
        assertNear(fn(0, 0, 0), 2 / 3, 0.08, "∫_{-1}^1 x^2 dx");
      },
    },
    {
      name: "spectral \\int_{-1}^{1} 1\\,dx is constant 2 on Cheb grid",
      fn: () => {
        const compiled = compileExpr(String.raw`\int_{-1}^{1} 1\,dx`);
        const half = 1;
        const deg = 8;
        const fit = fitScalarField(compiled, compiled.bind({}), half, deg, { skipMono: true });
        let maxErr = 0;
        for (let ix = 0; ix < fit.M; ix++) {
          for (let iy = 0; iy < fit.M; iy++) {
            for (let iz = 0; iz < fit.M; iz++) {
              const idx = densIndex(ix, iy, iz, fit.M);
              maxErr = Math.max(maxErr, Math.abs(fit.dens[idx]! - 2));
            }
          }
        }
        assert(maxErr < 0.3, `max integral fit error ${maxErr}`);
      },
    },
    {
      name: "\\int_{-1}^{1} y\\,dx yields field y",
      fn: () => {
        const compiled = compileExpr(String.raw`\int_{-1}^{1} y\,dx`);
        const half = 1;
        const deg = 8;
        const fit = fitScalarField(compiled, compiled.bind({}), half, deg, { skipMono: true });
        let maxErr = 0;
        for (let ix = 0; ix < fit.M; ix++) {
          for (let iy = 0; iy < fit.M; iy++) {
            const y = chebWorld(iy, fit.M, half);
            for (let iz = 0; iz < fit.M; iz++) {
              const idx = densIndex(ix, iy, iz, fit.M);
              maxErr = Math.max(maxErr, Math.abs(fit.dens[idx]! - 2 * y));
            }
          }
        }
        assert(maxErr < 0.35, `max ∫ y dx fit error ${maxErr}`);
      },
    },
    {
      name: "evalCheb1D matches polynomial value",
      fn: () => {
        const coeff = new Float64Array(3);
        coeff[0] = 1;
        coeff[1] = 0.5;
        assertNear(evalCheb1D(coeff, 0.5), 1 + 0.5 * 0.5, 0.01, "cheb eval");
      },
    },
  ]);
}
