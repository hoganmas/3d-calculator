import {
  clearExpressions,
  cssGradientFromColors,
  hexToRgb01,
  insertExprAt,
  listExpressions,
  mergeExprIntoPrevious,
  normalizeGradColors,
  normalizeExprRole,
  removeExprSilent,
  resolveExprGradient,
  setExpressions,
  splitExprAt,
} from "../../src/model/expressions.ts";
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
  ]);
}
