import "../helpers/setup-dom.ts";
import {
  formatParamDefLatex,
  paramCeIdToLatexSymbol,
  paramCeIdToPlainSymbol,
  isGreekParamCeId,
} from "../../src/math/paramSymbols.ts";
import { classifyExpr } from "../../src/math/fit.ts";
import { createParamRows, collectPendingParamsForExpr } from "../../src/app/pendingParams.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import { clearExpressions, listExpressions, setExpressions } from "../../src/model/expressions.ts";
import { state } from "../../src/app/state.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function resetScene() {
  state.pendingParamSeed = {};
  state.exprListApi = null;
  clearExpressions();
}

export async function run() {
  return runSuite("math / param-symbols", [
    {
      name: "paramCeIdToLatexSymbol maps Greek ids",
      fn: () => {
        assert(paramCeIdToLatexSymbol("alpha") === "\\alpha", "alpha");
        assert(paramCeIdToLatexSymbol("beta") === "\\beta", "beta");
        assert(paramCeIdToPlainSymbol("alpha") === "α", "alpha plain");
        assert(paramCeIdToLatexSymbol("a") === "a", "latin passthrough");
        assert(isGreekParamCeId("mu"), "mu is greek");
        assert(!isGreekParamCeId("mass"), "mass not greek");
      },
    },
    {
      name: "formatParamDefLatex emits Greek declaration rows",
      fn: () => {
        assert(formatParamDefLatex("alpha", 1) === "\\alpha=1", "alpha row");
        assert(formatParamDefLatex("b", 0.5) === "b=0.5", "latin row");
      },
    },
    {
      name: "bare alpha=1 is not a parameter declaration",
      fn: () => {
        const c = classifyExpr("alpha=1");
        assert(c.kind !== "parameter", "juxtaposition is not param");
      },
    },
    {
      name: "createParamRows uses Greek LaTeX for auto-generated rows",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: String.raw`\alpha x`, enabled: true }]);
        const item = listExpressions()[0]!;
        assert(collectPendingParamsForExpr(item).includes("alpha"), "pending alpha");
        createParamRows(["alpha"]);
        const row = listExpressions().find((e) => e.latex.startsWith(String.raw`\alpha=`));
        assert(!!row, "greek param row");
        assert(classifyExpr(row!.latex).kind === "parameter", "valid parameter row");
        compileAllExprs({ rebuildUi: false });
        assert(collectPendingParamsForExpr(item).length === 0, "alpha defined");
      },
    },
    {
      name: "createParamRows handles beta and omega",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: String.raw`\beta x + \omega y`, enabled: true }]);
        createParamRows(["beta", "omega"]);
        const rows = listExpressions().filter((e) => /^(\\beta=|\\omega=)/.test(e.latex));
        assert(rows.length === 2, "two greek rows");
      },
    },
  ]);
}
