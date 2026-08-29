import "../helpers/setup-dom.ts";
import {
  compileAllExprs,
  collectParamReferences,
} from "../../src/app/compile.ts";
import {
  clearExpressions,
  commitAutoParams,
  listExpressions,
  mergeExprIntoPrevious,
  moveExpr,
  setExpressions,
  splitExprAt,
  updateExprSilent,
} from "../../src/model/expressions.ts";
import { state } from "../../src/app/state.ts";
import { getParamValues } from "../../src/model/params.ts";
import { clearMockFocusedMathField } from "../helpers/setup-dom.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function resetScene() {
  state.pendingParamSeed = {};
  state.exprListApi = null;
  clearMockFocusedMathField();
  clearExpressions();
}

function paramRow(name: string) {
  return listExpressions().find((e) => String(e.latex || "").startsWith(`${name}=`));
}

function autoParamNames() {
  return listExpressions()
    .filter((e) => e.autoParam)
    .map((e) => String(e.latex || "").split("=")[0]);
}

export async function run() {
  return runSuite("app / auto-param", [
    {
      name: "compile creates multiple auto-param rows for a x + b y",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "a x + b y", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        assert(!!paramRow("a"), "auto row a");
        assert(!!paramRow("b"), "auto row b");
        assert(paramRow("a")?.autoParam === true, "a ephemeral");
        assert(paramRow("b")?.autoParam === true, "b ephemeral");
      },
    },
    {
      name: "split field row preserves auto-param reference",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "b x", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        assert(!!paramRow("b"), "b auto row");
        splitExprAt("e1", "b ", "x");
        compileAllExprs({ rebuildUi: false });
        assert(!!paramRow("b"), "b still referenced after split");
        assert(collectParamReferences().has("b"), "b in refs");
      },
    },
    {
      name: "merge field rows then prune drops unused auto-param",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "b ", enabled: true },
          { id: "e2", latex: "x", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        assert(!!paramRow("b"), "b auto row");
        updateExprSilent("e1", { latex: "b " });
        const merged = mergeExprIntoPrevious("e2");
        assert(!!merged, "merged");
        updateExprSilent("e1", { latex: "x^2" });
        compileAllExprs({ rebuildUi: false });
        assert(!paramRow("b"), "b row removed by compile/prune");
      },
    },
    {
      name: "commitAutoParams before reorder keeps param rows",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "b x", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        const bId = paramRow("b")?.id;
        assert(!!bId, "b exists");
        commitAutoParams();
        assert(paramRow("b")?.autoParam === false, "b permanent");
        moveExpr("e1", bId!);
        compileAllExprs({ rebuildUi: false });
        assert(!!paramRow("b"), "b survives reorder");
      },
    },
    {
      name: "disabled field row does not reference free params",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "b x", enabled: false }]);
        compileAllExprs({ rebuildUi: false });
        assert(!paramRow("b"), "no auto row for disabled field");
        assert(!collectParamReferences().has("b"), "b not referenced");
      },
    },
    {
      name: "flow field auto-creates missing param",
      fn: () => {
        resetScene();
        setExpressions([
          {
            id: "e1",
            latex: String.raw`\curl(b x, b y, 0)`,
            enabled: true,
            role: "flow",
          },
        ]);
        compileAllExprs({ rebuildUi: false });
        assert(!!paramRow("b"), "auto param for flow curl");
        assert(collectParamReferences().has("b"), "b referenced");
      },
    },
    {
      name: "param equation chain auto-creates transitive deps",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "a x", enabled: true },
          { id: "e2", latex: "a=2 b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        assert(!!paramRow("b"), "auto b for a=2b");
        const vals = getParamValues();
        assert(Number.isFinite(vals.a ?? NaN), "a resolves");
      },
    },
    {
      name: "rename param row orphan prunes after reference removed",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "b x", enabled: true },
          { id: "e2", latex: "b=1", enabled: true, autoParam: false },
        ]);
        compileAllExprs({ rebuildUi: false });
        updateExprSilent("e2", { latex: "c=1" });
        updateExprSilent("e1", { latex: "c x" });
        compileAllExprs({ rebuildUi: false });
        assert(!paramRow("b"), "old b row unused");
        assert(!!paramRow("c"), "c row present");
      },
    },
  ]);
}
