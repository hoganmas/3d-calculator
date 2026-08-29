import "../helpers/setup-dom.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import { compileParamLatex, formatParamLatexValue } from "../../src/math/fit.ts";
import { clearExpressions, setExpressions } from "../../src/model/expressions.ts";
import { getParamValues } from "../../src/model/params.ts";
import { state } from "../../src/app/state.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function resetScene() {
  state.pendingParamSeed = {};
  state.exprListApi = null;
  clearExpressions();
}

export async function run() {
  return runSuite("symbols / params", [
    {
      name: "compileParamLatex extracts free params from RHS",
      fn: () => {
        const compiled = compileParamLatex("a=2b", "a");
        assert(compiled.freeParams.includes("b"), "depends on b");
        assert(!compiled.isConstant, "not constant");
      },
    },
    {
      name: "compileParamLatex constant value",
      fn: () => {
        const compiled = compileParamLatex("a=3", "a");
        assert(compiled.isConstant, "constant");
        assertNear(compiled.eval({}), 3, 1e-9, "value");
      },
    },
    {
      name: "formatParamLatexValue finite and scientific",
      fn: () => {
        assert(formatParamLatexValue(1.5) === "1.5", "simple");
        assert(formatParamLatexValue(1e7).includes("e"), "scientific large");
        assert(formatParamLatexValue(NaN) === "0", "non-finite");
      },
    },
    {
      name: "compileAllExprs: param equation a=2b resolves through field",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "b=2", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
          { id: "e3", latex: "a x", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        const params = getParamValues();
        assertNear(params.b ?? NaN, 2, 1e-9, "b");
        assertNear(params.a ?? NaN, 4, 1e-9, "a=2b");
      },
    },
  ]);
}
