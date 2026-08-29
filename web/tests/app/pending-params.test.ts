import "../helpers/setup-dom.ts";
import {
  collectDefinedParamNames,
  collectPendingParamsForExpr,
  createParamRows,
  formatPendingParamLabel,
  pendingParamErrorMessage,
} from "../../src/app/pendingParams.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import {
  clearExpressions,
  getExprWarning,
  listExpressions,
  setExpressions,
  updateExprSilent,
} from "../../src/model/expressions.ts";
import { state } from "../../src/app/state.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function resetScene() {
  state.pendingParamSeed = {};
  state.exprListApi = null;
  clearExpressions();
}

function row(id: string, latex: string, extra: Record<string, unknown> = {}) {
  return { id, latex, enabled: true, ...extra };
}

export async function run() {
  return runSuite("app / pending-params", [
    {
      name: "collectPendingParamsForExpr finds missing field symbols",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "a x + b y")]);
        const pending = collectPendingParamsForExpr(listExpressions()[0]!);
        assert(pending.includes("a"), "missing a");
        assert(pending.includes("b"), "missing b");
      },
    },
    {
      name: "collectPendingParamsForExpr ignores defined params",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "a=1"), row("e2", "a x")]);
        const pending = collectPendingParamsForExpr(listExpressions()[1]!);
        assert(pending.length === 0, "a already defined");
      },
    },
    {
      name: "collectPendingParamsForExpr finds deps on parameter rows",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "a=2 b")]);
        const pending = collectPendingParamsForExpr(listExpressions()[0]!);
        assert(pending.includes("b"), "missing b on param row");
      },
    },
    {
      name: "createParamRows inserts declaration rows",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "b x")]);
        assert(createParamRows(["b"]), "created");
        assert(listExpressions().some((e) => e.latex.startsWith("b=")), "b row exists");
        assert(listExpressions().find((e) => e.latex.startsWith("b="))?.autoParam === true, "ephemeral");
      },
    },
    {
      name: "compileAllExprs no longer auto-creates param rows",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "b x")]);
        compileAllExprs({ rebuildUi: false });
        assert(!listExpressions().some((e) => e.latex.startsWith("b=")), "no implicit row");
        assert(collectPendingParamsForExpr(listExpressions()[0]!).includes("b"), "still pending");
      },
    },
    {
      name: "formatPendingParamLabel compresses long lists",
      fn: () => {
        assert(formatPendingParamLabel(["a"]) === "a", "single");
        assert(formatPendingParamLabel(["a", "b", "c"]) === "a, b, c", "triple");
        assert(formatPendingParamLabel(["a", "b", "c", "d"]) === "a, b +2", "many");
      },
    },
    {
      name: "collectDefinedParamNames tracks owning rows only",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "a=1"), row("e2", "a=2")]);
        const defined = collectDefinedParamNames();
        assert(defined.size === 1, "one owner");
        assert(defined.has("a"), "a defined");
      },
    },
    {
      name: "createParamRows skips already-defined names",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "a=1"), row("e2", "a x")]);
        const before = listExpressions().length;
        assert(!createParamRows(["a"]), "no duplicate row");
        assert(listExpressions().length === before, "unchanged count");
      },
    },
    {
      name: "pendingParamErrorMessage describes undefined symbols",
      fn: () => {
        assert(
          pendingParamErrorMessage(["b"]) ===
            "Undefined parameter: b. Press Tab or click below to create.",
          "single",
        );
        assert(
          pendingParamErrorMessage(["a", "b"])?.includes("Undefined parameters: a, b"),
          "multiple",
        );
      },
    },
    {
      name: "compile warns and skips layers when params are pending",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "b x")]);
        const result = compileAllExprs({ rebuildUi: false });
        assert(result.layers.length === 0, "no layer until params exist");
        const warn = getExprWarning("e1");
        assert(!!warn && warn.includes("Undefined parameter: b"), warn ?? "missing warn");
      },
    },
    {
      name: "pending clears after createParamRows and recompile",
      fn: () => {
        resetScene();
        setExpressions([row("e1", "b x")]);
        const item = listExpressions()[0]!;
        assert(collectPendingParamsForExpr(item).includes("b"), "pending before");
        createParamRows(["b"]);
        compileAllExprs({ rebuildUi: false });
        updateExprSilent("e1", { latex: "b x" });
        const after = collectPendingParamsForExpr(listExpressions()[0]!);
        assert(after.length === 0, "pending cleared");
      },
    },
  ]);
}
