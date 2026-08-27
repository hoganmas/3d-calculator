import { compileVectorExpr, fitVectorField } from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("vector / fit", [
    {
      name: "grad field matches analytic",
      fn: () => {
        const compiled = compileVectorExpr(String.raw`\grad(x^2+y^2+z^2)`);
        const vectorFn = compiled.bind({});
        const half = 1;
        const deg = 8;
        const result = fitVectorField(compiled, vectorFn, half, deg);
        const M = result.M;
        let maxErr = 0;
        for (let ix = 0; ix < M; ix++) {
          const x = chebWorld(ix, M, half);
          for (let iy = 0; iy < M; iy++) {
            const y = chebWorld(iy, M, half);
            for (let iz = 0; iz < M; iz++) {
              const z = chebWorld(iz, M, half);
              const idx = densIndex(ix, iy, iz, M);
              const [ex, ey, ez] = vectorFn(x, y, z);
              maxErr = Math.max(maxErr, Math.abs(result.fx[idx]! - ex));
              maxErr = Math.max(maxErr, Math.abs(result.fy[idx]! - ey));
              maxErr = Math.max(maxErr, Math.abs(result.fz[idx]! - ez));
            }
          }
        }
        assert(maxErr < 0.2, `max vector grad fit error ${maxErr}`);
      },
    },
    {
      name: "tuple field (-y, x, 0) matches analytic",
      fn: () => {
        const latex = String.raw`(-y, x, 0)`;
        const compiled = compileVectorExpr(latex);
        const vectorFn = compiled.bind({});
        const half = 1;
        const deg = 8;
        const result = fitVectorField(compiled, vectorFn, half, deg);
        const M = result.M;
        let maxErr = 0;
        for (let ix = 0; ix < M; ix++) {
          const x = chebWorld(ix, M, half);
          for (let iy = 0; iy < M; iy++) {
            const y = chebWorld(iy, M, half);
            for (let iz = 0; iz < M; iz++) {
              const z = chebWorld(iz, M, half);
              const idx = densIndex(ix, iy, iz, M);
              const [ex, ey, ez] = vectorFn(x, y, z);
              maxErr = Math.max(maxErr, Math.abs(result.fx[idx]! - ex));
              maxErr = Math.max(maxErr, Math.abs(result.fy[idx]! - ey));
              maxErr = Math.max(maxErr, Math.abs(result.fz[idx]! - ez));
            }
          }
        }
        assert(maxErr < 0.25, `max tuple fit error ${maxErr}`);
      },
    },
  ]);
}
