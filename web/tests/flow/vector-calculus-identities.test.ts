import { classifyExpr, compileExpr, fitScalarField } from "../../src/math/fit.ts";
import { compileVectorExpr } from "../../src/math/fitVector.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

const XYZ = { x: 0.2, y: 0.3, z: 0.4 };

function evalScalar(latex: string, pt = XYZ) {
  const compiled = compileExpr(latex);
  const v = compiled.bind({})(pt.x, pt.y, pt.z);
  return { compiled, v };
}

function evalVector(latex: string, pt = XYZ) {
  const compiled = compileVectorExpr(latex);
  const [fx, fy, fz] = compiled.bind({})(pt.x, pt.y, pt.z);
  return { compiled, fx, fy, fz };
}

export async function run() {
  return runSuite("flow / vector-calculus-identities", [
    {
      name: "\\nabla(r)\\cdot\\nabla(r) = 1 (scalar, not flow)",
      fn: () => {
        const latex = String.raw`\nabla\left(r\right)\cdot\nabla\left(r\right)`;
        const { compiled, v } = evalScalar(latex);
        assert(compiled.operator === "grad_dot", "grad dot operator");
        assertNear(v, 1, 0.05, "|grad r|^2");
        try {
          compileVectorExpr(latex);
          assert(false, "should not compile as vector");
        } catch {
          /* expected */
        }
      },
    },
    {
      name: "\\nabla\\cdot(x,y,z) = 3",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\nabla\cdot(x,y,z)`);
        assert(compiled.operator === "divergence", "operator");
        assertNear(v, 3, 0.05, "div identity");
      },
    },
    {
      name: "\\nabla\\cdot\\left(r\\right) = 3 (position vector)",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\nabla\cdot\left(r\right)`);
        assert(compiled.operator === "divergence", "operator");
        assertNear(v, 3, 0.05, "div r");
      },
    },
    {
      name: "\\div(r) = 3",
      fn: () => {
        const { v } = evalScalar(String.raw`\div(r)`);
        assertNear(v, 3, 0.05, "div r");
      },
    },
    {
      name: "\\nabla\\cdot(x^2+y^2+z^2) = 6 (scalar → Laplacian)",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\nabla\cdot(x^2+y^2+z^2)`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "div scalar as laplacian");
      },
    },
    {
      name: "\\nabla\\cdot(\\grad(x^2+y^2+z^2)) = 6",
      fn: () => {
        const { compiled, v } = evalScalar(
          String.raw`\nabla\cdot(\grad(x^2+y^2+z^2))`,
        );
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "div grad");
      },
    },
    {
      name: "\\laplacian(x^2+y^2+z^2) = 6",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\laplacian(x^2+y^2+z^2)`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "laplacian");
      },
    },
    {
      name: "\\nabla^2(x^2+y^2+z^2) = 6",
      fn: () => {
        const { v } = evalScalar(String.raw`\nabla^2(x^2+y^2+z^2)`);
        assertNear(v, 6, 0.08, "nabla^2");
      },
    },
    {
      name: "\\grad(x^2+y^2+z^2) = (2x,2y,2z)",
      fn: () => {
        const { fx, fy, fz } = evalVector(String.raw`\grad(x^2+y^2+z^2)`);
        assertNear(fx, 2 * XYZ.x, 0.05, "grad x");
        assertNear(fy, 2 * XYZ.y, 0.05, "grad y");
        assertNear(fz, 2 * XYZ.z, 0.05, "grad z");
      },
    },
    {
      name: "\\curl(-y,x,0) = (0,0,2)",
      fn: () => {
        const { fx, fy, fz } = evalVector(String.raw`\curl(-y,x,0)`);
        assertNear(fx, 0, 0.05, "curl fx");
        assertNear(fy, 0, 0.05, "curl fy");
        assertNear(fz, 2, 0.05, "curl fz");
      },
    },
    {
      name: "\\curl(\\grad(x^2+y^2+z^2)) = 0",
      fn: () => {
        const { fx, fy, fz } = evalVector(String.raw`\curl(\grad(x^2+y^2+z^2))`);
        assertNear(fx, 0, 0.05, "curl grad fx");
        assertNear(fy, 0, 0.05, "curl grad fy");
        assertNear(fz, 0, 0.05, "curl grad fz");
      },
    },
    {
      name: "\\nabla\\times(\\grad(x^2+y^2)) = 0",
      fn: () => {
        const { fx, fy, fz } = evalVector(String.raw`\nabla\times(\grad(x^2+y^2))`);
        assertNear(fx, 0, 0.05, "cross grad fx");
        assertNear(fy, 0, 0.05, "cross grad fy");
        assertNear(fz, 0, 0.05, "cross grad fz");
      },
    },
    {
      name: "\\frac{\\partial}{\\partial x}(x^2+y^2) = 2x",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\frac{\partial}{\partial x}(x^2+y^2)`);
        assert(compiled.operator === "partial", "operator");
        assertNear(v, 2 * XYZ.x, 0.05, "partial x");
      },
    },
    {
      name: "\\nabla\\cdot\\grad f matches \\laplacian f for f=x^2+y^2",
      fn: () => {
        const f = String.raw`x^2+y^2`;
        const divGrad = evalScalar(String.raw`\nabla\cdot(\grad(${f}))`).v;
        const lap = evalScalar(String.raw`\laplacian(${f})`).v;
        assertNear(divGrad, lap, 0.08, "div grad vs laplacian");
        assertNear(lap, 4, 0.08, "laplacian 2D slice");
      },
    },
    {
      name: "\\div r^2 = 6 (juxtaposition → Laplacian of r^2)",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\div r^2`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "div r^2");
      },
    },
    {
      name: "\\div r^{2} = 6 (braced exponent)",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\div r^{2}`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "div r^{2}");
      },
    },
    {
      name: "(\\div r)^2 = 9 (div before power)",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`(\div r)^2`);
        assert(!compiled.usesSpace, "constant");
        assertNear(v, 9, 1e-6, "(div r)^2");
      },
    },
    {
      name: "\\nabla\\cdot\\left(x^2+y^2+z^2\\right) with \\left/\\right",
      fn: () => {
        const { compiled, v } = evalScalar(
          String.raw`\nabla\cdot\left(x^2+y^2+z^2\right)`,
        );
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "div scalar with left/right");
      },
    },
    {
      name: "\\mathrm{div}(x^2+y^2+z^2) CE round-trip → Laplacian",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\mathrm{div}(x^2+y^2+z^2)`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "mathrm div");
      },
    },
    {
      name: "constraint \\nabla\\cdot(...)=0 bakes without NaN",
      fn: () => {
        const latex = String.raw`\nabla\cdot\left(x^2+y^2+z^2\right)=0`;
        const classified = classifyExpr(latex);
        const compiled = compileExpr(classified.compileLatex);
        assert(compiled.operator === "laplacian", "operator");
        const fit = fitScalarField(compiled, compiled.bind({}), 3, 8, {
          skipMono: true,
        });
        assert(fit.dens.length > 0, "dens");
      },
    },
  ]);
}
