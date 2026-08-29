import { classifyExpr } from "../../src/math/fit.ts";
import {
  clearExpressions,
  commitAutoParams,
  cssGradientFromColors,
  getSelectedId,
  getExprWarning,
  hasExprWarnings,
  hexToRgb01,
  insertExprAt,
  listExpressions,
  listExprWarnings,
  mergeExprIntoPrevious,
  moveExpr,
  nextExprGradient,
  normalizeGradColors,
  normalizeExprRole,
  removeExpr,
  removeExprSilent,
  replaceExprWarnings,
  resolveExprGradient,
  resolveExprRole,
  selectExpr,
  setExpressions,
  setExpressionsOnChange,
  splitExprAt,
  updateExpr,
  updateExprSilent,
} from "../../src/model/expressions.ts";
import { SymbolRegistry } from "../../src/model/symbols.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("model / expressions", [
    {
      name: "normalizeExprRole maps legacy density/constraint",
      fn: () => {
        assert(normalizeExprRole("density") === "cloud", "density→cloud");
        assert(normalizeExprRole("constraint") === "isosurface", "constraint→iso");
        assert(normalizeExprRole("flow") === "flow", "flow passthrough");
      },
    },
    {
      name: "normalizeGradColors clamps and fills endpoints",
      fn: () => {
        const one = normalizeGradColors(["#ff0000"]);
        assert(one.length === 2, "single becomes pair");
        const many = normalizeGradColors(["#f00", "#0f0", "#00f", "#111", "#222", "#333", "#444"]);
        assert(many.length === 6, "max stops");
      },
    },
    {
      name: "resolveExprGradient and cssGradientFromColors",
      fn: () => {
        const grad = resolveExprGradient({ color: "#ff0000", color2: "#0000ff" });
        assert(grad.colors.length === 2, "two stops");
        const css = cssGradientFromColors(grad.colors);
        assert(css.includes("linear-gradient"), "css gradient");
        assert(css.includes("#ff0000"), "start color");
      },
    },
    {
      name: "hexToRgb01 converts hex",
      fn: () => {
        const rgb = hexToRgb01("#ff8000");
        assertNear(rgb[0]!, 1, 1e-6, "r");
        assertNear(rgb[1]!, 0.5, 0.01, "g");
        assertNear(rgb[2]!, 0, 1e-6, "b");
      },
    },
    {
      name: "insertExprAt and removeExprSilent",
      fn: () => {
        clearExpressions();
        setExpressions([{ latex: "x" }]);
        const row = insertExprAt(0, { latex: "y" });
        assert(listExpressions().length >= 2, "inserted row");
        const removed = removeExprSilent(row.id);
        assert(removed === true, "removed");
      },
    },
    {
      name: "mergeExprIntoPrevious combines latex",
      fn: () => {
        clearExpressions();
        setExpressions([
          { id: "e1", latex: "x" },
          { id: "e2", latex: "+y" },
        ]);
        const merged = mergeExprIntoPrevious("e2");
        assert(!!merged, "merged");
        assert(
          listExpressions().some((e) => e.latex.includes("x") && e.latex.includes("y")),
          "combined",
        );
      },
    },
    {
      name: "splitExprAt creates new row",
      fn: () => {
        clearExpressions();
        setExpressions([{ id: "e1", latex: "x+y" }]);
        const split = splitExprAt("e1", "x", "+y");
        assert(!!split, "split created");
        assert(listExpressions().length >= 2, "two rows");
      },
    },
    {
      name: "moveExpr reorders rows",
      fn: () => {
        clearExpressions();
        setExpressions([
          { id: "e1", latex: "a" },
          { id: "e2", latex: "b" },
        ]);
        assert(moveExpr("e2", "e1"), "moved");
        assert(listExpressions()[0]?.id === "e2", "e2 first");
        assert(!moveExpr("e1", "e1"), "no-op self move");
      },
    },
    {
      name: "updateExpr and updateExprSilent patch colors",
      fn: () => {
        clearExpressions();
        setExpressions([{ id: "e1", latex: "x" }]);
        updateExpr("e1", { colors: ["#ff0000", "#0000ff"] });
        const row = listExpressions().find((e) => e.id === "e1");
        assert(row?.color === "#ff0000", "color set");
        updateExprSilent("e1", { colors: ["#111111", "#222222"] });
        const silent = listExpressions().find((e) => e.id === "e1");
        assert(silent?.colors?.length === 2, "silent colors");
        assert(silent?.color === "#111111", "silent primary");
      },
    },
    {
      name: "selectExpr and commitAutoParams",
      fn: () => {
        clearExpressions();
        setExpressions([{ id: "e1", latex: "x", autoParam: true }]);
        selectExpr("e1");
        assert(getSelectedId() === "e1", "selected");
        commitAutoParams();
        assert(!listExpressions()[0]?.autoParam, "auto cleared");
      },
    },
    {
      name: "replaceExprWarnings and listExprWarnings",
      fn: () => {
        clearExpressions();
        setExpressions([{ id: "e1", latex: "x" }]);
        replaceExprWarnings([["e1", "warn"]]);
        const listed = listExprWarnings();
        assert(listed.length === 1 && listed[0]?.message === "warn", "listed");
        assert(hasExprWarnings(), "has warnings");
      },
    },
    {
      name: "removeExpr notifies via emit trailing blank",
      fn: () => {
        clearExpressions();
        setExpressions([
          { id: "e1", latex: "x" },
          { id: "e2", latex: "y" },
        ]);
        removeExpr("e2");
        assert(listExpressions().length >= 2, "trailing blank kept");
      },
    },
    {
      name: "resolveExprRole infers flow from vector latex",
      fn: () => {
        const reg = new SymbolRegistry();
        assert(resolveExprRole("auto", "field", "(-y,x,0)", reg) === "flow", "vector→flow");
        assert(resolveExprRole("auto", "constraint", "x^2", reg) === "isosurface", "constraint");
        const param = classifyExpr("a=1");
        assert(resolveExprRole("auto", param.kind) === "parameter", "param row");
      },
    },
    {
      name: "hexToRgb01 invalid hex uses fallback",
      fn: () => {
        const rgb = hexToRgb01("not-a-color");
        assert(rgb.length === 3 && rgb.every((v) => v > 0), "fallback rgb");
      },
    },
    {
      name: "nextExprGradient cycles palette",
      fn: () => {
        clearExpressions();
        setExpressions([{ id: "e1", latex: "x" }]);
        const g = nextExprGradient();
        assert(g.colors.length === 2, "gradient pair");
      },
    },
    {
      name: "setExpressionsOnChange fires on updateExpr",
      fn: () => {
        clearExpressions();
        setExpressions([{ id: "e1", latex: "x" }]);
        let calls = 0;
        setExpressionsOnChange(() => {
          calls++;
        });
        updateExpr("e1", { enabled: false });
        setExpressionsOnChange(null);
        assert(calls >= 1, "onChange called");
      },
    },
    {
      name: "getExprWarning reads stored warning",
      fn: () => {
        clearExpressions();
        setExpressions([{ id: "e1", latex: "x" }]);
        replaceExprWarnings([["e1", "dup"]]);
        assert(getExprWarning("e1") === "dup", "warning");
      },
    },
  ]);
}
