import { compileExpr } from "../../src/math/fit.ts";
import { compileVectorExpr } from "../../src/math/fitVector.ts";
import {
  parseDivergenceMatch,
  parseTupleCrossMatch,
  parseTupleDotMatch,
  normalizeCalcLatex,
} from "../../src/math/calcOps.ts";
import { ComputeEngine } from "@cortex-js/compute-engine";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

const ce = new ComputeEngine();
const XYZ = { x: 0.2, y: 0.3, z: 0.4 };

function normalizedJson(latex: string) {
  const src = normalizeCalcLatex(latex);
  const box = ce.parse(src);
  const json = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);
  return { src, json };
}

export async function run() {
  return runSuite("flow / vector-ops", [
    {
      name: "parseTupleCrossMatch",
      fn: () => {
        const src = normalizeCalcLatex(String.raw`\left(x,y,z\right)\times\left(z,y,x\right)`);
        const match = parseTupleCrossMatch(src);
        assert(match != null, "cross match");
        assert(match!.left.join(",") === "x,y,z", "left");
        assert(match!.right.join(",") === "z,y,x", "right");
      },
    },
    {
      name: "parseTupleDotMatch",
      fn: () => {
        const src = normalizeCalcLatex(String.raw`\left(x,y,z\right)\cdot\left(z,y,x\right)`);
        const match = parseTupleDotMatch(src);
        assert(match != null, "dot match");
      },
    },
    {
      name: "scaled divergence → laplacian with scale",
      fn: () => {
        const latex = String.raw`0.1\left(\nabla\cdot\frac{1}{r}\right)`;
        const { src, json } = normalizedJson(latex);
        const match = parseDivergenceMatch(src, json);
        assert(match?.mode === "laplacian", `mode ${match?.mode}`);
        if (match?.mode === "laplacian") {
          assertNear(match.scale ?? 1, 0.1, 1e-9, "scale");
          assert(!match.inner.includes(")"), `inner should not have stray paren: ${match.inner}`);
        }
      },
    },
    {
      name: "cross product compiles as flow",
      fn: () => {
        const compiled = compileVectorExpr(String.raw`\left(x,y,z\right)\times\left(z,y,x\right)`);
        assert(compiled.kind === "cross", "kind");
        const [fx, fy, fz] = compiled.bind({})(XYZ.x, XYZ.y, XYZ.z);
        assertNear(fx, -0.06, 1e-6, "fx");
        assertNear(fy, 0.12, 1e-6, "fy");
        assertNear(fz, -0.06, 1e-6, "fz");
      },
    },
    {
      name: "dot product compiles as scalar",
      fn: () => {
        const compiled = compileExpr(String.raw`\left(x,y,z\right)\cdot\left(z,y,x\right)`);
        assert(compiled.operator === "dot_product", "operator");
        const v = compiled.bind({})(XYZ.x, XYZ.y, XYZ.z);
        assertNear(v, 0.25, 1e-6, "dot");
      },
    },
    {
      name: "scalar × tuple compiles as scaled flow",
      fn: () => {
        const compiled = compileVectorExpr(String.raw`0.1\left(x,y,z\right)`);
        assert(compiled.kind === "tuple", "kind");
        const [fx, fy, fz] = compiled.bind({})(XYZ.x, XYZ.y, XYZ.z);
        assertNear(fx, 0.02, 1e-6, "fx");
        assertNear(fy, 0.03, 1e-6, "fy");
        assertNear(fz, 0.04, 1e-6, "fz");
      },
    },
    {
      name: "scaled laplacian from div shorthand compiles",
      fn: () => {
        const compiled = compileExpr(String.raw`0.1\left(\nabla\cdot\frac{1}{r}\right)`);
        assert(compiled.operator === "laplacian", "operator");
        const v = compiled.bind({})(XYZ.x, XYZ.y, XYZ.z);
        assertNear(v, 0, 0.15, "scaled laplacian of 1/r away from origin");
      },
    },
  ]);
}
