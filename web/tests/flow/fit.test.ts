import { compileVectorExpr, fitVectorField } from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("flow / fit", [
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
    {
      name: "grad field fit with L2 metric enabled",
      fn: () => {
        const compiled = compileVectorExpr(String.raw`\grad(x^2+y^2+z^2)`);
        const vectorFn = compiled.bind({});
        const result = fitVectorField(compiled, vectorFn, 1, 6, { skipL2: false });
        assert(result.source === "gradient", "gradient source");
        assert(Number.isFinite(result.fitRel), "fitRel computed");
      },
    },
    {
      name: "grad field with param a scales with slider value",
      fn: () => {
        const compiled = compileVectorExpr(String.raw`\nabla\left(ar\right)`);
        const half = 2.5;
        const deg = 10;
        const fitLo = fitVectorField(
          compiled,
          compiled.bind({ a: 1 }),
          half,
          deg,
          { params: { a: 1 } },
        );
        const fitHi = fitVectorField(
          compiled,
          compiled.bind({ a: 5 }),
          half,
          deg,
          { params: { a: 5 } },
        );
        const maxLo = Math.max(...fitLo.fx);
        const maxHi = Math.max(...fitHi.fx);
        assert(maxHi > maxLo * 3, `expected scale with a: maxLo=${maxLo} maxHi=${maxHi}`);
        const ratio = maxHi / maxLo;
        assert(Math.abs(ratio - 5) < 0.6, `expected ~5× scale, got ${ratio}`);
      },
    },
    {
      name: "tuple field fit with L2 metric enabled",
      fn: () => {
        const compiled = compileVectorExpr(String.raw`(-y,x,0)`);
        const vectorFn = compiled.bind({});
        const result = fitVectorField(compiled, vectorFn, 1, 6, { skipL2: false });
        assert(result.source === "tuple", "tuple source");
        assert(Number.isFinite(result.fitRel), "fitRel");
      },
    },
  ]);
}
