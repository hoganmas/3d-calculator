import { formatExpressionErrors, type ExpressionErrorReport } from "../../src/app/exprErrors.js";
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
  ]);
}
