import { ComputeEngine } from "@cortex-js/compute-engine";
import {
  extractOperatornameArg,
  extractTriple,
  isPositionVectorLatex,
  mathJsonHasError,
  normalizeCalcLatex,
  normalizeLatexAliases,
  normalizeDegreeLatex,
  parseDivergenceMatch,
  scalarFromGradJson,
  tripleFromUnaryOpJson,
  unwrapLatexSymbolTokens,
} from "../../src/math/calcOps.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

const ce = new ComputeEngine();

function normalizedJson(latex: string) {
  const src = normalizeCalcLatex(latex);
  const box = ce.parse(src);
  const json = box?.json ?? (typeof box?.toJSON === "function" ? box.toJSON() : null);
  return { src, json };
}

export async function run() {
  return runSuite("flow / calcOps", [
    {
      name: "normalizeLatexAliases: \\del → \\nabla",
      fn: () => {
        assert(
          normalizeLatexAliases(String.raw`\del\cdot(x,y,z)`).includes("\\nabla"),
          "del alias",
        );
      },
    },
    {
      name: "normalizeDegreeLatex: numeric \\deg → \\circ",
      fn: () => {
        const out = normalizeDegreeLatex("45\\deg");
        assert(out.includes("\\circ"), `got ${out}`);
      },
    },
    {
      name: "mathJsonHasError: Error node",
      fn: () => {
        assert(mathJsonHasError(["Error", "syntax"]), "Error node");
        assert(!mathJsonHasError(["Add", 1, 2]), "valid Add");
        assert(mathJsonHasError("Degree"), "Degree symbol");
      },
    },
    {
      name: "unwrapLatexSymbolTokens strips mathrm",
      fn: () => {
        assert(
          unwrapLatexSymbolTokens(String.raw`\mathrm{div}(x,y,z)`) === "div(x,y,z)",
          "unwrap mathrm",
        );
      },
    },
    {
      name: "extractOperatornameArg: div",
      fn: () => {
        const arg = extractOperatornameArg(String.raw`\operatorname{div}(x,y,z)`, "div");
        assert(arg === "(x,y,z)" || arg === "x,y,z", `got ${arg}`);
      },
    },
    {
      name: "isPositionVectorLatex: r and \\mathbf{r}",
      fn: () => {
        assert(isPositionVectorLatex("r"), "bare r");
        assert(isPositionVectorLatex(String.raw`\mathbf{r}`), "mathbf r");
        assert(!isPositionVectorLatex("(x,y)"), "not position");
      },
    },
    {
      name: "parseDivergenceMatch: \\nabla\\cdot(x,y,z)",
      fn: () => {
        const latex = String.raw`\nabla\cdot(x,y,z)`;
        const { src, json } = normalizedJson(latex);
        const match = parseDivergenceMatch(src, json);
        assert(match?.mode === "triple", `mode ${match?.mode}`);
        if (match?.mode === "triple") {
          assert(match.parts.join(",") === "x,y,z", "triple parts");
        }
      },
    },
    {
      name: "parseDivergenceMatch: \\div(r) position vector",
      fn: () => {
        const latex = String.raw`\div(r)`;
        const { src, json } = normalizedJson(latex);
        const match = parseDivergenceMatch(src, json);
        assert(match?.mode === "triple", `mode ${match?.mode}`);
      },
    },
    {
      name: "parseDivergenceMatch: scalar arg → laplacian",
      fn: () => {
        const latex = String.raw`\nabla\cdot(x^2+y^2+z^2)`;
        const { src, json } = normalizedJson(latex);
        const match = parseDivergenceMatch(src, json);
        assert(match?.mode === "laplacian", `mode ${match?.mode}`);
      },
    },
    {
      name: "extractTriple from tuple JSON",
      fn: () => {
        const triple = extractTriple(["Tuple", "x", "y", "z"]);
        assert(triple?.length === 3 && triple[0] === "x", "tuple");
      },
    },
    {
      name: "tripleFromUnaryOpJson: curl tuple arg",
      fn: () => {
        const triple = tripleFromUnaryOpJson(
          ["Multiply", "curl", ["Tuple", "-y", "x", "0"]],
          "curl",
        );
        assert(triple?.length === 3, "curl triple");
      },
    },
    {
      name: "scalarFromGradJson",
      fn: () => {
        const inner = "x^2+y^2";
        const { json } = normalizedJson(String.raw`\grad(${inner})`);
        const got = scalarFromGradJson(json);
        assert(got === inner || got?.includes("x"), `got ${got}`);
      },
    },
  ]);
}
