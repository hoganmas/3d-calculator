import { ComputeEngine } from "@cortex-js/compute-engine";
import {
  looksLikePartial,
  normalizeCalcLatex,
  normalizePartialForms,
  peelDefiniteIntegrals,
} from "../../src/math/calcOps.ts";
import { compileExpr, normalizeForCe } from "../../src/math/fit.ts";
import { isVectorFieldLatex } from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

const ce = new ComputeEngine();

function partialFromLatex(latex: string) {
  const normalized = normalizeForCe(latex);
  const box = ce.parse(normalized);
  const j = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);
  return looksLikePartial(normalized, j);
}

export async function run() {
  return runSuite("flow / calculus-parse", [
    {
      name: "normalizePartialForms: \\partial_x subscript",
      fn: () => {
        const out = normalizePartialForms(String.raw`\partial_x f`);
        assert(out === String.raw`\operatorname{partial_x}{f}`, `got ${out}`);
      },
    },
    {
      name: "normalizePartialForms: \\grad_y with parens",
      fn: () => {
        const out = normalizePartialForms(String.raw`\grad_y (x^2)`);
        assert(out === String.raw`\operatorname{partial_y}{x^2}`, `got ${out}`);
      },
    },
    {
      name: "normalizePartialForms: \\frac{\\partial f}{\\partial x}",
      fn: () => {
        const out = normalizePartialForms(String.raw`\frac{\partial f}{\partial x}`);
        assert(out === String.raw`\operatorname{partial_x}{f}`, `got ${out}`);
      },
    },
    {
      name: "normalizePartialForms: \\frac{\\partial}{\\partial z} prefix",
      fn: () => {
        const out = normalizePartialForms(String.raw`\frac{\partial}{\partial z} e^{-(x^2)}`);
        assert(
          out === String.raw`\operatorname{partial_z}{e^{-(x^2)}}`,
          `got ${out}`,
        );
      },
    },
    {
      name: "normalizePartialForms: \\partial_x e^{...} keeps exponent braces",
      fn: () => {
        const latex = String.raw`\partial_x e^{-(x^2+y^2+z^2)}`;
        const out = normalizePartialForms(latex);
        assert(
          out === String.raw`\operatorname{partial_x}{e^{-(x^2+y^2+z^2)}}`,
          `got ${out}`,
        );
      },
    },
    {
      name: "normalizePartialForms: brace-wrapped argument",
      fn: () => {
        const out = normalizePartialForms(String.raw`\partial_x {x^2+y^2}`);
        assert(out === String.raw`\operatorname{partial_x}{x^2+y^2}`, `got ${out}`);
      },
    },
    {
      name: "normalizePartialForms: already normalized passthrough",
      fn: () => {
        const latex = String.raw`\operatorname{partial_x}{f}`;
        assert(normalizePartialForms(latex) === latex, "should not rewrite");
      },
    },
    {
      name: "normalizeCalcLatex: \\grad_z becomes partial_z not vector grad",
      fn: () => {
        const out = normalizeCalcLatex(String.raw`\grad_z (x^2+y^2+z^2)`);
        assert(out.includes(String.raw`\operatorname{partial_z}{`), `got ${out}`);
        assert(!out.includes(String.raw`\operatorname{grad}`), "grad_z must not become grad");
      },
    },
    {
      name: "peelDefiniteIntegrals: single bound integral",
      fn: () => {
        const peeled = peelDefiniteIntegrals(String.raw`\int_{0}^{1} 2x\,dx`);
        assert(peeled != null, "expected peel");
        assert(peeled.inner === "2x", `inner ${peeled.inner}`);
        assert(peeled.axes.length === 1, "one axis");
        assert(peeled.axes[0]!.axis === 0, "x axis");
        assert(peeled.axes[0]!.aLatex === "0", "lower bound");
        assert(peeled.axes[0]!.bLatex === "1", "upper bound");
      },
    },
    {
      name: "peelDefiniteIntegrals: symbolic bounds preserved",
      fn: () => {
        const peeled = peelDefiniteIntegrals(String.raw`\int_{a}^{b} x^2\,dx`);
        assert(peeled != null, "expected peel");
        assert(peeled.axes[0]!.aLatex === "a", "aLatex");
        assert(peeled.axes[0]!.bLatex === "b", "bLatex");
      },
    },
    {
      name: "peelDefiniteIntegrals: missing bounds",
      fn: () => {
        const peeled = peelDefiniteIntegrals(String.raw`\int x^2\,dx`);
        assert(peeled != null, "expected peel");
        assert(peeled.axes[0]!.aLatex === "", "empty lower");
        assert(peeled.axes[0]!.bLatex === "", "empty upper");
      },
    },
    {
      name: "peelDefiniteIntegrals: chained \\int … dy … dx",
      fn: () => {
        const peeled = peelDefiniteIntegrals(String.raw`\int_{0}^{1}\int_{0}^{1} xy\,dy\,dx`);
        assert(peeled != null, "expected peel");
        assert(peeled.inner === "xy", `inner ${peeled.inner}`);
        assert(peeled.axes.length === 2, "two axes");
        assert(peeled.axes[0]!.axis === 1, "inner y axis first");
        assert(peeled.axes[1]!.axis === 0, "outer x axis second");
      },
    },
    {
      name: "peelDefiniteIntegrals: rejects missing differential",
      fn: () => {
        assert(peelDefiniteIntegrals(String.raw`\int_{0}^{1} x^2`) === null, "no dx");
      },
    },
    {
      name: "peelDefiniteIntegrals: rejects plain expression",
      fn: () => {
        assert(peelDefiniteIntegrals("x^2") === null, "not an integral");
      },
    },
    {
      name: "peelDefiniteIntegrals: rejects empty integrand",
      fn: () => {
        assert(peelDefiniteIntegrals(String.raw`\int_{0}^{1}\,dx`) === null, "empty inner");
      },
    },
    {
      name: "looksLikePartial: detects each axis from normalized LaTeX",
      fn: () => {
        const cases: Array<{ latex: string; axis: 0 | 1 | 2; inner: string }> = [
          { latex: String.raw`\partial_x f`, axis: 0, inner: "f" },
          { latex: String.raw`\partial_y (x^2+y^2)`, axis: 1, inner: "x^2+y^2" },
          { latex: String.raw`\grad_z z^2`, axis: 2, inner: "z^2" },
          {
            latex: String.raw`\frac{\partial}{\partial x}(x^2+y^2)`,
            axis: 0,
            inner: "x^2+y^2",
          },
          {
            latex: String.raw`\frac{\partial (x^2)}{\partial x}`,
            axis: 0,
            inner: "x^2",
          },
        ];
        for (const { latex, axis, inner } of cases) {
          const match = partialFromLatex(latex);
          assert(match != null, `no match for ${latex}`);
          assert(match.axis === axis, `${latex}: axis ${match.axis} !== ${axis}`);
          assert(match.inner === inner, `${latex}: inner "${match.inner}" !== "${inner}"`);
        }
      },
    },
    {
      name: "looksLikePartial: plain scalar is not partial",
      fn: () => {
        assert(partialFromLatex("x^2+y^2") === null, "quadratic");
        assert(partialFromLatex(String.raw`\laplacian(x^2)`) === null, "laplacian");
      },
    },
    {
      name: "compileExpr routes \\frac{\\partial f}{\\partial x} to partial operator",
      fn: () => {
        const compiled = compileExpr(String.raw`\frac{\partial (x^2+y^2)}{\partial x}`);
        assert(compiled.operator === "partial", "operator");
        assert(compiled.partialAxis === 0, "x axis");
        assert(compiled.scalarCompileLatex === "x^2+y^2", `inner ${compiled.scalarCompileLatex}`);
      },
    },
    {
      name: "compileExpr routes definite integral without evaluating as plain scalar",
      fn: () => {
        const compiled = compileExpr(String.raw`\int_{0}^{1} 2x\,dx`);
        assert(compiled.operator === "definite_integral", "operator");
        assert(compiled.scalarCompileLatex === "2x", `inner ${compiled.scalarCompileLatex}`);
        assert(compiled.integralAxes?.length === 1, "one axis spec");
        assert(compiled.integralAxes![0]!.axis === 0, "x axis");
      },
    },
    {
      name: "compileExpr: integral without dx is not definite_integral",
      fn: () => {
        const compiled = compileExpr(String.raw`\int_{0}^{1} x^2`);
        assert(compiled.operator !== "definite_integral", "should not peel");
      },
    },
    {
      name: "partial LaTeX is not classified as vector field syntax",
      fn: () => {
        const cases = [
          String.raw`\partial_x (x^2+y^2)`,
          String.raw`\frac{\partial}{\partial y}(z^2)`,
          String.raw`\grad_z (x^2+y^2+z^2)`,
        ];
        for (const latex of cases) {
          assert(!isVectorFieldLatex(latex), `${latex} must not be vector syntax`);
        }
      },
    },
    {
      name: "compileExpr: double integral metadata",
      fn: () => {
        const compiled = compileExpr(String.raw`\int_{0}^{1}\int_{0}^{1} xy\,dy\,dx`);
        assert(compiled.operator === "definite_integral", "operator");
        assert(compiled.scalarCompileLatex === "xy", `inner ${compiled.scalarCompileLatex}`);
        assert(compiled.integralAxes?.length === 2, "two axes");
        assert(compiled.integralAxes![0]!.axis === 1, "inner y");
        assert(compiled.integralAxes![1]!.axis === 0, "outer x");
      },
    },
  ]);
}
