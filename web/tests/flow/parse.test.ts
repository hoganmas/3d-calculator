import { compileExpr } from "../../src/math/fit.ts";
import {
  classifyVectorExpr,
  compileVectorExpr,
  isVectorFieldLatex,
} from "../../src/math/fitVector.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("flow / parse", [
    {
      name: "tuple (-y, x, 0) classifies and evaluates",
      fn: () => {
        const latex = String.raw`(-y, x, 0)`;
        assert(classifyVectorExpr(latex).kind === "tuple", "expected tuple");
        const fn = compileVectorExpr(latex).bind({});
        const [fx, fy, fz] = fn(0.1, 0.2, 0.3);
        assert([fx, fy, fz].every(Number.isFinite), "non-finite eval");
        assertNear(fx, -0.2, 1e-9, "fx");
        assertNear(fy, 0.1, 1e-9, "fy");
        assertNear(fz, 0, 1e-9, "fz");
      },
    },
    {
      name: "\\grad classifies as gradient",
      fn: () => {
        const latex = String.raw`\grad(x^2+y^2+z^2)`;
        assert(classifyVectorExpr(latex).kind === "gradient", "expected gradient");
        const fn = compileVectorExpr(latex).bind({});
        const [fx, fy, fz] = fn(0.1, 0.2, 0.3);
        assert([fx, fy, fz].every(Number.isFinite), "non-finite eval");
      },
    },
    {
      name: "\\nabla classifies as gradient",
      fn: () => {
        const latex = String.raw`\nabla(x^2+y^2)`;
        assert(classifyVectorExpr(latex).kind === "gradient", "expected gradient");
        const fn = compileVectorExpr(latex).bind({});
        assert(fn(0.1, 0.2, 0.3).every(Number.isFinite), "non-finite eval");
      },
    },
    {
      name: "\\grad with \\left(\\right) parens (MathLive)",
      fn: () => {
        const latex = String.raw`\grad\left(x^2+y^2+z^2\right)`;
        assert(classifyVectorExpr(latex).kind === "gradient");
        const fn = compileVectorExpr(latex).bind({});
        assert(fn(0.1, 0.2, 0.3).every(Number.isFinite), "non-finite eval");
      },
    },
    {
      name: "\\nabla with \\left(\\right) parens (MathLive)",
      fn: () => {
        const latex = String.raw`\nabla\left(x^2+y^2+z^2\right)`;
        assert(classifyVectorExpr(latex).kind === "gradient");
        compileVectorExpr(latex);
      },
    },
    {
      name: "\\nabla\\cdot with \\left(\\right) parens (MathLive)",
      fn: () => {
        const latex = String.raw`\nabla\cdot\left(x,y,z\right)`;
        const compiled = compileExpr(latex);
        assert(compiled.operator === "divergence", "expected divergence");
        assertNear(compiled.bind({})(0.1, 0.2, 0.3), 3, 0.05, "nabla dot with left/right");
      },
    },
    {
      name: "scalar sin(x) is not a vector field",
      fn: () => {
        assert(!isVectorFieldLatex("sin(x)"), "sin(x) wrongly detected as vector");
      },
    },
  ]);
}
