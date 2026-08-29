import { compileExpr, fitScalarField } from "../../src/math/fit.ts";
import {
  classifyVectorExpr,
  compileVectorExpr,
  fitVectorField,
} from "../../src/math/fitVector.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("flow / operators", [
    {
      name: "\\laplacian(x^2+y^2+z^2) evaluates to 6",
      fn: () => {
        const compiled = compileExpr(String.raw`\laplacian(x^2+y^2+z^2)`);
        assert(compiled.operator === "laplacian", "expected laplacian operator");
        const fn = compiled.bind({});
        assertNear(fn(0.1, 0.2, 0.3), 6, 0.05, "laplacian r^2");
      },
    },
    {
      name: "\\nabla\\cdot(x,y,z) evaluates to 3",
      fn: () => {
        const compiled = compileExpr(String.raw`\nabla\cdot(x,y,z)`);
        assert(compiled.operator === "divergence", "expected divergence operator");
        const fn = compiled.bind({});
        assertNear(fn(0.1, 0.2, 0.3), 3, 0.05, "nabla dot identity field");
      },
    },
    {
      name: "\\nabla^2(x^2+y^2+z^2) evaluates to 6",
      fn: () => {
        const compiled = compileExpr(String.raw`\nabla^2(x^2+y^2+z^2)`);
        assert(compiled.operator === "laplacian", "expected laplacian operator");
        const fn = compiled.bind({});
        assertNear(fn(0.1, 0.2, 0.3), 6, 0.05, "nabla^2 r^2");
      },
    },
    {
      name: "\\del\\cdot(x,y,z) evaluates to 3",
      fn: () => {
        const compiled = compileExpr(String.raw`\del\cdot(x,y,z)`);
        assert(compiled.operator === "divergence", "expected divergence operator");
        const fn = compiled.bind({});
        assertNear(fn(0.1, 0.2, 0.3), 3, 0.05, "del dot identity field");
      },
    },
    {
      name: "\\div(x,y,z) evaluates to 3",
      fn: () => {
        const compiled = compileExpr(String.raw`\div(x,y,z)`);
        assert(compiled.operator === "divergence", "expected divergence operator");
        const fn = compiled.bind({});
        assertNear(fn(0.1, 0.2, 0.3), 3, 0.05, "div identity field");
      },
    },
    {
      name: "\\curl(-y,x,0) evaluates to (0,0,2)",
      fn: () => {
        const latex = String.raw`\curl(-y,x,0)`;
        assert(classifyVectorExpr(latex).kind === "curl", "expected curl");
        const compiled = compileVectorExpr(latex);
        const [fx, fy, fz] = compiled.bind({})(0.1, 0.2, 0.3);
        assertNear(fx, 0, 0.05, "curl fx");
        assertNear(fy, 0, 0.05, "curl fy");
        assertNear(fz, 2, 0.05, "curl fz");
      },
    },
    {
      name: "\\nabla\\times(-y,x,0) evaluates to (0,0,2)",
      fn: () => {
        const latex = String.raw`\nabla\times(-y,x,0)`;
        assert(classifyVectorExpr(latex).kind === "curl", "expected curl");
        const compiled = compileVectorExpr(latex);
        const [fx, fy, fz] = compiled.bind({})(0.1, 0.2, 0.3);
        assertNear(fx, 0, 0.05, "nabla cross fx");
        assertNear(fy, 0, 0.05, "nabla cross fy");
        assertNear(fz, 2, 0.05, "nabla cross fz");
      },
    },
    {
      name: "\\del(x^2+y^2) classifies as gradient",
      fn: () => {
        const latex = String.raw`\del(x^2+y^2)`;
        assert(classifyVectorExpr(latex).kind === "gradient", "expected gradient");
        const fn = compileVectorExpr(latex).bind({});
        assert(fn(0.1, 0.2, 0.3).every(Number.isFinite), "non-finite eval");
      },
    },
    {
      name: "spectral laplacian of r^2 on Cheb grid",
      fn: () => {
        const compiled = compileExpr(String.raw`\laplacian(x^2+y^2+z^2)`);
        const half = 1;
        const deg = 8;
        const fit = fitScalarField(compiled, compiled.bind({}), half, deg, { skipMono: true });
        let maxErr = 0;
        for (let ix = 0; ix < fit.M; ix++) {
          const x = chebWorld(ix, fit.M, half);
          for (let iy = 0; iy < fit.M; iy++) {
            const y = chebWorld(iy, fit.M, half);
            for (let iz = 0; iz < fit.M; iz++) {
              const idx = densIndex(ix, iy, iz, fit.M);
              maxErr = Math.max(maxErr, Math.abs(fit.dens[idx]! - 6));
            }
          }
        }
        assert(maxErr < 0.25, `max laplacian fit error ${maxErr}`);
      },
    },
    {
      name: "spectral curl of swirl field",
      fn: () => {
        const compiled = compileVectorExpr(String.raw`\curl(-y,x,0)`);
        const vectorFn = compiled.bind({});
        const half = 2.5;
        const deg = 8;
        const result = fitVectorField(compiled, vectorFn, half, deg);
        assert(result.source === "curl", "expected curl source");
        let maxErr = 0;
        for (let ix = 0; ix < result.M; ix++) {
          for (let iy = 0; iy < result.M; iy++) {
            for (let iz = 0; iz < result.M; iz++) {
              const idx = densIndex(ix, iy, iz, result.M);
              maxErr = Math.max(maxErr, Math.abs(result.fz[idx]! - 2));
            }
          }
        }
        assert(maxErr < 0.3, `max curl fit error ${maxErr}`);
      },
    },
    {
      name: "\\frac{\\partial}{\\partial x}(x^2+y^2) evaluates to 2x",
      fn: () => {
        const compiled = compileExpr(String.raw`\frac{\partial}{\partial x}(x^2+y^2)`);
        assert(compiled.operator === "partial", "expected partial operator");
        assert(compiled.partialAxis === 0, "expected x axis");
        const fn = compiled.bind({});
        assertNear(fn(0.3, 0.4, 0.5), 0.6, 0.05, "partial d/dx");
      },
    },
    {
      name: "\\partial_y (x^2+y^2+z^2) evaluates to 2y",
      fn: () => {
        const compiled = compileExpr(String.raw`\partial_y (x^2+y^2+z^2)`);
        assert(compiled.operator === "partial", "expected partial operator");
        assert(compiled.partialAxis === 1, "expected y axis");
        const fn = compiled.bind({});
        assertNear(fn(0.1, 0.25, 0.3), 0.5, 0.05, "partial d/dy");
      },
    },
    {
      name: "\\grad_z (x^2+y^2+z^2) evaluates to 2z",
      fn: () => {
        const compiled = compileExpr(String.raw`\grad_z (x^2+y^2+z^2)`);
        assert(compiled.operator === "partial", "expected partial operator");
        assert(compiled.partialAxis === 2, "expected z axis");
        const fn = compiled.bind({});
        assertNear(fn(0.1, 0.2, 0.35), 0.7, 0.05, "partial d/dz via grad_z");
      },
    },
    {
      name: "spectral partial d/dx of x^2+y^2 on Cheb grid",
      fn: () => {
        const compiled = compileExpr(String.raw`\frac{\partial}{\partial x}(x^2+y^2)`);
        const half = 1;
        const deg = 8;
        const fit = fitScalarField(compiled, compiled.bind({}), half, deg, { skipMono: true });
        let maxErr = 0;
        for (let ix = 0; ix < fit.M; ix++) {
          const x = chebWorld(ix, fit.M, half);
          for (let iy = 0; iy < fit.M; iy++) {
            const y = chebWorld(iy, fit.M, half);
            for (let iz = 0; iz < fit.M; iz++) {
              const idx = densIndex(ix, iy, iz, fit.M);
              maxErr = Math.max(maxErr, Math.abs(fit.dens[idx]! - 2 * x));
            }
          }
        }
        assert(maxErr < 0.25, `max partial fit error ${maxErr}`);
      },
    },
  ]);
}
