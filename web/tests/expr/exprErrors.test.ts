import { collectExpressionErrors, formatExpressionErrors, type ExpressionErrorReport } from "../../src/app/exprErrors.js";
import "../helpers/setup-dom.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import { clearExpressions, getExprWarning, setExpressions } from "../../src/model/expressions.ts";
import { state } from "../../src/app/state.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("expr / exprErrors", [
    {
      name: "formats multi-line banner with row numbers",
      fn() {
        const report: ExpressionErrorReport = {
          compileOk: false,
          globalError: "Fit failed",
          expressionCount: 2,
          errorCount: 3,
          errors: [
            { kind: "global", message: "Fit failed" },
            { kind: "expression", id: "a", row: 1, latex: "x^", message: "Unexpected token" },
            {
              kind: "parameter",
              id: "b",
              row: 2,
              name: "a",
              latex: "a=1/0",
              message: "Division by zero",
            },
          ],
        };
        assert(
          formatExpressionErrors(report) ===
            "Fit failed\nrow 1: Unexpected token\nrow 2 · a: Division by zero",
          "multi-line banner",
        );
      },
    },
    {
      name: "returns empty string when no errors",
      fn() {
        assert(
          formatExpressionErrors({
            compileOk: true,
            globalError: null,
            expressionCount: 1,
            errorCount: 0,
            errors: [],
          }) === "",
          "empty when no errors",
        );
      },
    },
    {
      name: "collectExpressionErrors gathers warnings after compile",
      fn() {
        state.pendingParamSeed = {};
        state.exprListApi = null;
        clearExpressions();
        setExpressions([
          { id: "e1", latex: "a=1", enabled: true },
          { id: "e2", latex: "a=2", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        const report = collectExpressionErrors(true, null);
        assert(report.errorCount >= 1, "has errors");
        assert(report.errors.some((e) => e.kind === "expression"), "expression error");
        assert(getExprWarning("e2") != null, "warning on duplicate");
      },
    },
    {
      name: "collectExpressionErrors includes global error",
      fn() {
        const report = collectExpressionErrors(false, "Fit failed");
        assert(report.globalError === "Fit failed", "global");
        assert(report.errors[0]?.kind === "global", "global kind");
      },
    },
  ]);
}
