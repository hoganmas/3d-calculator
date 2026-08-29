import { classifyExpr, compileExpr } from "../../src/math/fit.ts";
import {
  classifyVectorExpr,
  compileVectorExpr,
  isVectorFieldLatex,
} from "../../src/math/fitVector.ts";
import { resolveExprRole } from "../../src/model/expressions.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

const PT = { x: 0.25, y: -0.35, z: 0.55 };
const { x, y, z } = PT;
const r2 = x * x + y * y + z * z;
const r = Math.sqrt(r2);

function evalScalar(latex: string) {
  const compiled = compileExpr(latex);
  const v = compiled.bind({})(x, y, z);
  assert(!Array.isArray(v), `expected scalar, got vector for ${latex}`);
  return { compiled, v: v as number };
}

function evalVector(latex: string) {
  const compiled = compileVectorExpr(latex);
  const out = compiled.bind({})(x, y, z);
  assert(Array.isArray(out) && out.length === 3, `expected vector for ${latex}`);
  return { compiled, fx: out[0]!, fy: out[1]!, fz: out[2]! };
}

function assertScalarRoute(latex: string) {
  const classified = classifyExpr(latex);
  assert(!isVectorFieldLatex(latex), `should not infer flow: ${latex}`);
  assert(
    resolveExprRole("auto", classified.kind, classified.compileLatex) !== "flow",
    `role should not be flow: ${latex}`,
  );
  try {
    compileVectorExpr(latex);
    assert(false, `should not compile as vector: ${latex}`);
  } catch {
    /* expected */
  }
}

function assertVectorRoute(latex: string) {
  assert(isVectorFieldLatex(latex), `should infer flow: ${latex}`);
  classifyVectorExpr(latex);
}

export async function run() {
  return runSuite("flow / eval-stress", [
    {
      name: "grad dot: \\nabla(r)\\cdot\\nabla(r) = 1 (scalar)",
      fn: () => {
        const latex = String.raw`\nabla\left(r\right)\cdot\nabla\left(r\right)`;
        assertScalarRoute(latex);
        const { compiled, v } = evalScalar(latex);
        assert(compiled.operator === "grad_dot", "operator");
        assertNear(v, 1, 0.05, "|grad r|^2");
      },
    },
    {
      name: "grad dot: \\grad(f)\\cdot\\grad(g) orthogonal → 0",
      fn: () => {
        const { v } = evalScalar(String.raw`\grad(x^2)\cdot\grad(y^2)`);
        assertNear(v, 0, 0.05, "orthogonal grad dot");
      },
    },
    {
      name: "grad dot: \\grad(r^2)\\cdot\\grad(r^2) = 4r^2",
      fn: () => {
        const { v } = evalScalar(String.raw`\grad(x^2+y^2+z^2)\cdot\grad(x^2+y^2+z^2)`);
        assertNear(v, 4 * r2, 0.15, "|grad r^2|^2");
      },
    },
    {
      name: "tuple dot: (x,y,z)\\cdot(x,y,z) = r^2",
      fn: () => {
        const latex = String.raw`\left(x,y,z\right)\cdot\left(x,y,z\right)`;
        assertScalarRoute(latex);
        const { compiled, v } = evalScalar(latex);
        assert(compiled.operator === "dot_product", "operator");
        assertNear(v, r2, 1e-6, "tuple dot");
      },
    },
    {
      name: "tuple dot with \\grad components = 3",
      fn: () => {
        const latex = String.raw`\left(\grad x,\grad y,\grad z\right)\cdot\left(\grad x,\grad y,\grad z\right)`;
        assertScalarRoute(latex);
        const { compiled, v } = evalScalar(latex);
        assert(compiled.operator === "dot_product", "operator");
        assertNear(v, 3, 0.05, "basis grad tuple dot");
      },
    },
    {
      name: "cross product: (x,y,z)\\times(z,y,x)",
      fn: () => {
        const latex = String.raw`\left(x,y,z\right)\times\left(z,y,x\right)`;
        assertVectorRoute(latex);
        const { compiled, fx, fy, fz } = evalVector(latex);
        assert(compiled.kind === "cross", "kind");
        assertNear(fx, y * x - z * y, 1e-6, "cross fx");
        assertNear(fy, z * z - x * x, 1e-6, "cross fy");
        assertNear(fz, x * y - y * z, 1e-6, "cross fz");
      },
    },
    {
      name: "scalar × tuple",
      fn: () => {
        const latex = String.raw`0.5\left(x,y,z\right)`;
        assertVectorRoute(latex);
        const { fx, fy, fz } = evalVector(latex);
        assertNear(fx, 0.5 * x, 1e-6, "scale fx");
        assertNear(fy, 0.5 * y, 1e-6, "scale fy");
        assertNear(fz, 0.5 * z, 1e-6, "scale fz");
      },
    },
    {
      name: "gradient \\grad(r^2)",
      fn: () => {
        const { compiled, fx, fy, fz } = evalVector(String.raw`\grad(x^2+y^2+z^2)`);
        assert(compiled.kind === "gradient", "kind");
        assertNear(fx, 2 * x, 0.05, "grad x");
        assertNear(fy, 2 * y, 0.05, "grad y");
        assertNear(fz, 2 * z, 0.05, "grad z");
      },
    },
    {
      name: "gradient \\nabla(r)",
      fn: () => {
        const { fx, fy, fz } = evalVector(String.raw`\nabla\left(r\right)`);
        assertNear(fx, x / r, 0.05, "nabla r fx");
        assertNear(fy, y / r, 0.05, "nabla r fy");
        assertNear(fz, z / r, 0.05, "nabla r fz");
      },
    },
    {
      name: "divergence \\nabla\\cdot(x,y,z) = 3",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\nabla\cdot(x,y,z)`);
        assert(compiled.operator === "divergence", "operator");
        assertNear(v, 3, 0.05, "div");
      },
    },
    {
      name: "divergence scalar shorthand \\nabla\\cdot(r^2) = laplacian",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\nabla\cdot\left(r^2\right)`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.1, "laplacian r^2");
      },
    },
    {
      name: "laplacian \\nabla^2(r^2)",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\nabla^2\left(x^2+y^2+z^2\right)`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 6, 0.08, "laplacian");
      },
    },
    {
      name: "partial \\partial_x(x^2+y^2)",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`\partial_x\left(x^2+y^2\right)`);
        assert(compiled.operator === "partial", "operator");
        assertNear(v, 2 * x, 0.05, "partial x");
      },
    },
    {
      name: "curl \\curl(-y,x,0)",
      fn: () => {
        const { compiled, fx, fy, fz } = evalVector(String.raw`\curl(-y,x,0)`);
        assert(compiled.kind === "curl", "kind");
        assertNear(fx, 0, 0.05, "curl fx");
        assertNear(fy, 0, 0.05, "curl fy");
        assertNear(fz, 2, 0.05, "curl fz");
      },
    },
    {
      name: "curl of gradient is zero",
      fn: () => {
        const { fx, fy, fz } = evalVector(String.raw`\curl(\grad(x^2+y^2+z^2))`);
        assertNear(fx, 0, 0.05, "curl grad fx");
        assertNear(fy, 0, 0.05, "curl grad fy");
        assertNear(fz, 0, 0.05, "curl grad fz");
      },
    },
    {
      name: "plain scalar stays scalar route",
      fn: () => {
        const latex = String.raw`x^2+y^2+z^2`;
        assertScalarRoute(latex);
        const { compiled, v } = evalScalar(latex);
        assert(compiled.operator == null || compiled.operator === "none", "no vector op");
        assertNear(v, r2, 1e-6, "value");
      },
    },
    {
      name: "vector tuple stays vector route",
      fn: () => {
        const latex = String.raw`(-y,x,0)`;
        assertVectorRoute(latex);
        const { fx, fy, fz } = evalVector(latex);
        assertNear(fx, -y, 1e-6, "tuple fx");
        assertNear(fy, x, 1e-6, "tuple fy");
        assertNear(fz, 0, 1e-6, "tuple fz");
      },
    },
    {
      name: "scaled divergence preserves laplacian scale",
      fn: () => {
        const { compiled, v } = evalScalar(String.raw`2\left(\nabla\cdot\frac{1}{r}\right)`);
        assert(compiled.operator === "laplacian", "operator");
        assertNear(v, 0, 0.15, "scaled laplacian of 1/r");
      },
    },
  ]);
}
