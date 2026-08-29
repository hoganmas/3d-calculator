import "../helpers/setup-dom.ts";
import {
  applyPreset,
  clearAllExprs,
  compileAllExprs,
  ensureParamExprRows,
  fmtParamNum,
  initCompile,
  layerRgbFromItem,
  pruneUnusedAutoParams,
  shouldDeferAutoParamRows,
  collectParamReferences,
} from "../../src/app/compile.ts";
import {
  clearExpressions,
  getExprWarning,
  listExpressions,
  setExpressions,
  updateExprSilent,
} from "../../src/model/expressions.ts";
import { getParam, getParamValues } from "../../src/model/params.ts";
import { state } from "../../src/app/state.ts";
import { clearMockFocusedMathField, setMockFocusedMathField } from "../helpers/setup-dom.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function resetScene() {
  state.pendingParamSeed = {};
  state.exprListApi = null;
  clearMockFocusedMathField();
  clearExpressions();
}

export async function run() {
  return runSuite("app / compile", [
    {
      name: "compileAllExprs: mixed param, alias, funcdef, cloud, flow",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "a=1", enabled: true },
          { id: "e2", latex: String.raw`T=x^2+y^2`, enabled: true },
          { id: "e3", latex: String.raw`f(x)=\sin(x)`, enabled: true },
          { id: "e4", latex: "T", enabled: true },
          { id: "e5", latex: String.raw`(-y,x,0)`, enabled: true, role: "flow" },
        ]);
        const result = compileAllExprs({ rebuildUi: false });
        assert(result.layers.length >= 2, `expected layers, got ${result.layers.length}`);
        const roles = new Set(result.layers.map((l) => l.role));
        assert(roles.has("cloud") || roles.has("isosurface"), "scalar layer");
        assert(roles.has("flow"), "flow layer");
        const params = getParamValues();
        assertNear(params.a ?? NaN, 1, 1e-9, "param a");
      },
    },
    {
      name: "compileAllExprs: duplicate parameter warns",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "a=1", enabled: true },
          { id: "e2", latex: "a=2", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        const warn = getExprWarning("e2");
        assert(!!warn, "expected duplicate warning on second row");
        assert(warn!.toLowerCase().includes("duplicate") || warn!.toLowerCase().includes("already"), warn!);
      },
    },
    {
      name: "compileAllExprs: invalid LaTeX warns",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "x^", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        const warn = getExprWarning("e1");
        assert(!!warn, "expected parse warning");
      },
    },
    {
      name: "compileAllExprs: invalid flow field warns",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "x^2", enabled: true, role: "flow" }]);
        compileAllExprs({ rebuildUi: false });
        const warn = getExprWarning("e1");
        assert(!!warn, "expected flow compile warning");
        assert(warn!.toLowerCase().includes("flow") || warn!.toLowerCase().includes("tuple"), warn!);
      },
    },
    {
      name: "hidden funcdef stays in scope for curl reference",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: String.raw`f(x,y,z)=(-y,x,0)`, enabled: false },
          { id: "e2", latex: String.raw`\del \times f`, enabled: true, role: "flow" },
        ]);
        const result = compileAllExprs({ rebuildUi: false });
        assert(result.layers.length >= 1, "curl layer compiles");
        const flow = result.layers.find((l) => l.role === "flow");
        assert(!!flow?.vectorFn, "flow fn bound");
        const [, , cz] = flow!.vectorFn!(0.2, 0.3, 0.4);
        assertNear(cz, 2, 0.15, "curl z");
      },
    },
    {
      name: "compileAllExprs: field references param value",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "a=2", enabled: true },
          { id: "e2", latex: "a x", enabled: true },
        ]);
        const result = compileAllExprs({ rebuildUi: false });
        const layer = result.layers[0];
        assert(!!layer?.fn, "layer fn");
        assertNear(layer!.fn!(0.5, 0, 0), 1, 1e-6, "a*x at x=0.5");
      },
    },
    {
      name: "compileAllExprs: auto-creates missing param rows",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "b x", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        const rows = listExpressions().filter((e) => String(e.latex || "").trim());
        assert(rows.some((r) => r.latex.startsWith("b=")), "auto param row for b");
      },
    },
    {
      name: "compileAllExprs: defers auto-param rows while math field focused",
      fn: () => {
        resetScene();
        setMockFocusedMathField();
        setExpressions([{ id: "e1", latex: "c x", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        const before = listExpressions().filter((e) => e.latex.startsWith("c="));
        assert(before.length === 0, "no auto row while typing");
        clearMockFocusedMathField();
        compileAllExprs({ rebuildUi: false });
        const after = listExpressions().filter((e) => e.latex.startsWith("c="));
        assert(after.length >= 1, "auto row after blur");
      },
    },
    {
      name: "shouldDeferAutoParamRows: false on param-def row",
      fn: () => {
        resetScene();
        setMockFocusedMathField({ paramDefRow: true });
        assert(!shouldDeferAutoParamRows(), "param-def row does not defer");
        clearMockFocusedMathField();
      },
    },
    {
      name: "compileAllExprs: applies pendingParamSeed to param rows",
      fn: () => {
        resetScene();
        state.pendingParamSeed = { a: { value: 5, min: 0, max: 10, speed: 0.4 } };
        setExpressions([{ id: "e1", latex: "a=1", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        assert(Object.keys(state.pendingParamSeed).length === 0, "seed consumed");
        const row = listExpressions().find((e) => e.id === "e1");
        assert(row?.latex === "a=5", `expected a=5, got ${row?.latex}`);
        assertNear(getParamValues().a ?? NaN, 5, 1e-9, "param value");
        const p = getParam("a");
        assertNear(p?.min ?? NaN, 0, 1e-9, "seed min");
        assertNear(p?.max ?? NaN, 10, 1e-9, "seed max");
      },
    },
    {
      name: "compileAllExprs: rebuildUi calls exprListApi.render",
      fn: () => {
        resetScene();
        let rendered = false;
        state.exprListApi = {
          render: () => {
            rendered = true;
          },
          syncParamChrome: () => false,
        };
        setExpressions([{ id: "e1", latex: "x^2", enabled: true }]);
        compileAllExprs({ rebuildUi: true });
        assert(rendered, "render called");
      },
    },
    {
      name: "pruneUnusedAutoParams removes unreferenced auto rows",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "b x", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        const autoBefore = listExpressions().filter((e) => e.autoParam);
        assert(autoBefore.length >= 1, "auto row created");
        updateExprSilent("e1", { latex: "x^2" });
        const removed = pruneUnusedAutoParams();
        assert(removed, "pruned unused auto param");
        assert(listExpressions().every((e) => !e.autoParam || e.latex.startsWith("b=") === false), "auto b gone");
      },
    },
    {
      name: "collectParamReferences gathers field and param deps",
      fn: () => {
        resetScene();
        setExpressions([
          { id: "e1", latex: "a=2b", enabled: true },
          { id: "e2", latex: "a x", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        const refs = collectParamReferences();
        assert(refs.has("a"), "field refs a");
        assert(refs.has("b"), "param refs b");
      },
    },
    {
      name: "fmtParamNum formats edge cases",
      fn: () => {
        assert(fmtParamNum(1.2345) === "1.235", "rounds");
        assert(fmtParamNum(999) === "999", "large int below 1e3");
        assert(fmtParamNum(0.001) === "0.00100", "small scientific");
        assert(fmtParamNum(0.05) === "0.05", "small decimal");
        assert(fmtParamNum(Number.NaN) === "—", "nan");
        assert(fmtParamNum(Number.POSITIVE_INFINITY) === "—", "inf");
      },
    },
    {
      name: "layerRgbFromItem resolves gradient colors",
      fn: () => {
        const rgb = layerRgbFromItem({
          id: "t",
          latex: "x",
          enabled: true,
          color: "#ff0000",
          color2: "#0000ff",
        });
        assert(rgb.color.length === 3, "rgb triplet");
        assertNear(rgb.color[0]!, 1, 1e-6, "red channel");
        assertNear(rgb.color2[2]!, 1, 1e-6, "blue channel");
        assert(rgb.colors.length >= 2, "multi-stop");
      },
    },
    {
      name: "applyPreset loads sincos expressions",
      fn: () => {
        resetScene();
        applyPreset("sincos");
        const rows = listExpressions().filter((e) => String(e.latex || "").trim());
        assert(rows.length >= 2, "preset rows");
        assert(rows.some((r) => r.latex.includes("sin")), "field row");
        assert(rows.some((r) => r.latex.startsWith("t=")), "param row");
      },
    },
    {
      name: "clearAllExprs leaves one blank row",
      fn: () => {
        resetScene();
        setExpressions([{ id: "e1", latex: "x^2", enabled: true }]);
        clearAllExprs();
        const rows = listExpressions();
        assert(rows.length >= 1, "blank row");
        assert(rows.every((r) => !String(r.latex || "").trim() || rows.length === 1), "cleared");
      },
    },
    {
      name: "initCompile seeds default preset when empty",
      fn: () => {
        resetScene();
        clearAllExprs();
        initCompile();
        const rows = listExpressions().filter((e) => String(e.latex || "").trim());
        assert(rows.length >= 2, "preset loaded");
      },
    },
    {
      name: "compileAllExprs: grad(r)·grad(r) is scalar cloud not flow",
      fn: () => {
        resetScene();
        const latex = String.raw`\nabla\left(r\right)\cdot\nabla\left(r\right)`;
        setExpressions([{ id: "e1", latex, enabled: true, role: "auto" }]);
        const result = compileAllExprs({ rebuildUi: false });
        assert(result.layers.length === 1, "one layer");
        assert(result.layers[0]?.role === "cloud", "cloud role");
        assert(result.layers[0]?.compiled?.operator === "grad_dot", "grad dot scalar");
        assert(result.warnings.length === 0, "no warnings");
        const v = result.layers[0]?.fn?.(0.5, 0.2, 0.1);
        assertNear(v ?? NaN, 1, 0.05, "|grad r|^2");
      },
    },
  ]);
}
