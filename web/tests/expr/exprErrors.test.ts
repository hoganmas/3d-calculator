import { collectExpressionErrors, formatExpressionErrors, logExpressionErrors, type ExpressionErrorReport } from "../../src/app/exprErrors.js";
import "../helpers/setup-dom.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import { clearExpressions, getExprWarning, setExpressions } from "../../src/model/expressions.ts";
import { recompileParam, syncParamsFromDefinitions } from "../../src/model/params.ts";
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
    {
      name: "collectExpressionErrors includes parameter compile errors",
      fn() {
        state.pendingParamSeed = {};
        state.exprListApi = null;
        clearExpressions();
        setExpressions([{ id: "e1", latex: "a=1", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        syncParamsFromDefinitions([{ name: "a", latex: "a=((", exprId: "e1" }]);
        recompileParam("a");
        const report = collectExpressionErrors(false, null);
        assert(report.errors.some((e) => e.kind === "parameter" && e.name === "a"), "param error");
        const formatted = formatExpressionErrors(report);
        assert(formatted.includes("a:"), "param in banner");
      },
    },
    {
      name: "formatExpressionErrors param without row uses param name",
      fn() {
        const text = formatExpressionErrors({
          compileOk: false,
          globalError: null,
          expressionCount: 0,
          errorCount: 1,
          errors: [{ kind: "parameter", name: "t", message: "bad rhs" }],
        });
        assert(text === "param t · t: bad rhs", text);
      },
    },
    {
      name: "logExpressionErrors no-op and dump paths",
      fn() {
        const logs: string[] = [];
        const origInfo = console.info;
        const origGroup = console.group;
        const origWarn = console.warn;
        const origGroupEnd = console.groupEnd;
        console.info = (msg: string) => logs.push(String(msg));
        console.group = (msg: string) => logs.push(`group:${msg}`);
        console.warn = (msg: string) => logs.push(`warn:${msg}`);
        console.groupEnd = () => logs.push("groupEnd");
        try {
          logExpressionErrors({
            compileOk: true,
            globalError: null,
            expressionCount: 0,
            errorCount: 0,
            errors: [],
          });
          assert(logs.some((l) => l.includes("no errors")), "info when clean");
          logExpressionErrors({
            compileOk: false,
            globalError: null,
            expressionCount: 1,
            errorCount: 2,
            errors: [
              { kind: "global", message: "boom" },
              { kind: "expression", id: "e1", row: 1, latex: "x^", message: "syntax" },
              { kind: "parameter", name: "a", message: "bad", latex: "a=((" },
            ],
          });
          assert(logs.some((l) => l.startsWith("group:")), "group when errors");
          assert(logs.some((l) => l.startsWith("warn:global")), "warn global");
        } finally {
          console.info = origInfo;
          console.group = origGroup;
          console.warn = origWarn;
          console.groupEnd = origGroupEnd;
        }
      },
    },
  ]);
}
