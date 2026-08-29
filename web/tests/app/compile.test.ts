import "../helpers/setup-dom.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import {
  clearExpressions,
  getExprWarning,
  listExpressions,
  setExpressions,
} from "../../src/model/expressions.ts";
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
  ]);
}
