import "../helpers/setup-dom.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import { clearExpressions, setExpressions, updateExprSilent } from "../../src/model/expressions.ts";
import {
  anyParamAnimating,
  collectAnimDirtyParams,
  evalParamEquations,
  applyParamSeed,
  ensureParamAnimationFromExprs,
  getParam,
  getParamValues,
  normalizeAnimMode,
  phaseForValue,
  recompileParam,
  setParamValue,
  stopParamAnimation,
  syncParamsFromDefinitions,
  tickParamAnimation,
  toggleParamAnimate,
  updateParam,
  collectValueDirtyParams,
} from "../../src/model/params.ts";
import { state } from "../../src/app/state.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function resetParams() {
  state.pendingParamSeed = {};
  state.exprListApi = null;
  clearExpressions();
  syncParamsFromDefinitions([]);
}

export async function run() {
  return runSuite("model / params-animation", [
    {
      name: "normalizeAnimMode defaults to pingpong",
      fn: () => {
        assert(normalizeAnimMode(undefined) === "pingpong", "default");
        assert(normalizeAnimMode("loop") === "loop", "loop");
      },
    },
    {
      name: "phaseForValue pingpong and loop modes",
      fn: () => {
        const p = { value: 0.5, min: 0, max: 1, speed: 0.5, animMode: "pingpong" as const };
        const phase = phaseForValue(p, 0);
        assert(Number.isFinite(phase), "pingpong phase");
        const loop = phaseForValue({ ...p, animMode: "loop" }, 1);
        assert(Number.isFinite(loop), "loop phase");
      },
    },
    {
      name: "toggleParamAnimate and stopParamAnimation",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.5", exprId: "e1", min: 0, max: 1, animating: false },
        ]);
        const on = toggleParamAnimate("a", 0);
        assert(on?.animating === true, "started");
        assert(anyParamAnimating(), "any animating");
        const off = stopParamAnimation("a");
        assert(off?.animating === false, "stopped");
      },
    },
    {
      name: "tickParamAnimation updates slider value",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          {
            name: "a",
            latex: "a=0",
            exprId: "e1",
            min: 0,
            max: 10,
            value: 0,
            animating: true,
            speed: 0.35,
            phase: 0,
            animMode: "pingpong",
          },
        ]);
        const changed = tickParamAnimation(0);
        assert(changed, "value changed");
        assertNear(getParamValues().a ?? NaN, 5, 0.01, "pingpong at phase 0");
      },
    },
    {
      name: "evalParamEquations resolves driven param",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        updateParam("b", { value: 3 });
        assertNear(getParamValues().a ?? NaN, 2, 1e-9, "a unchanged until eval");
        const changed = evalParamEquations();
        assert(changed, "eval changed values");
        assertNear(getParamValues().a ?? NaN, 6, 1e-9, "a=2b");
      },
    },
    {
      name: "collectAnimDirtyParams includes driven dependents",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        updateParam("b", { animating: true });
        const dirty = collectAnimDirtyParams();
        assert(dirty.has("b"), "anim param dirty");
        assert(dirty.has("a"), "driven dependent dirty");
      },
    },
    {
      name: "driven params cannot animate",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        toggleParamAnimate("a", 0);
        assert(getParam("a")?.animating !== true, "driven stays off");
        assert(getParam("a")?.driven === true, "marked driven");
      },
    },
    {
      name: "tickParamAnimation loop mode wraps phase",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          {
            name: "a",
            latex: "a=0",
            exprId: "e1",
            min: 0,
            max: 10,
            value: 0,
            animating: true,
            speed: 1,
            phase: 0,
            animMode: "loop",
          },
        ]);
        tickParamAnimation(0.5);
        assertNear(getParamValues().a ?? NaN, 5, 0.01, "loop midpoint");
        tickParamAnimation(1.25);
        assert(getParam("a")?.animating === true, "still animating");
      },
    },
    {
      name: "ensureParamAnimationFromExprs honors sliderAnimating",
      fn: () => {
        resetParams();
        setExpressions([{ id: "e1", latex: "a=0.5", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        updateExprSilent("e1", { sliderAnimating: true });
        const changed = ensureParamAnimationFromExprs(0);
        assert(changed, "started from expr flag");
        assert(getParam("a")?.animating === true, "param animating");
      },
    },
    {
      name: "syncParamsFromDefinitions preserves in-flight animation",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          {
            name: "a",
            latex: "a=0.5",
            exprId: "e1",
            min: 0,
            max: 1,
            animating: true,
          },
        ]);
        syncParamsFromDefinitions([
          {
            name: "a",
            latex: "a=0.5",
            exprId: "e1",
            min: 0,
            max: 1,
            animating: false,
          },
        ]);
        assert(getParam("a")?.animating === true, "live animating survives stale defs");
      },
    },
    {
      name: "recompileParam records invalid latex error",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([{ name: "a", latex: "a=((", exprId: "e1" }]);
        assert(recompileParam("a") === false, "failed compile");
        assert(!!getParam("a")?.error, "error stored");
      },
    },
    {
      name: "static value may sit outside animation bounds",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          {
            name: "a",
            latex: "a=0.01",
            exprId: "e1",
            min: 0.5,
            max: 1,
            value: 0.01,
            animating: false,
          },
        ]);
        assertNear(getParamValues().a ?? NaN, 0.01, 1e-12, "value kept outside [min,max]");
        assertNear(getParam("a")!.min, 0.5, 1e-12, "anim min unchanged");
        assertNear(getParam("a")!.max, 1, 1e-12, "anim max unchanged");

        updateParam("a", { min: 0.6, max: 0.9 });
        assertNear(getParamValues().a ?? NaN, 0.01, 1e-12, "bounds change does not clamp value");

        setParamValue("a", 0.02, { stopAnim: true, rewriteLatex: true });
        assertNear(getParamValues().a ?? NaN, 0.02, 1e-12, "setParamValue allows out-of-range");
        assert(getParam("a")!.latex.includes("0.02"), "latex rewritten");
      },
    },
    {
      name: "recompile keeps constant value outside animation bounds",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.01", exprId: "e1", min: 0.5, max: 1, value: 0.01 },
        ]);
        assert(recompileParam("a"), "recompile ok");
        assertNear(getParamValues().a ?? NaN, 0.01, 1e-12, "constant outside bounds");
      },
    },
    {
      name: "driven eval may produce values outside animation bounds",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        updateParam("a", { min: 0.5, max: 1 });
        setParamValue("b", 0.01, { stopAnim: true, rewriteLatex: false });
        evalParamEquations();
        assertNear(getParamValues().a ?? NaN, 0.02, 1e-9, "driven a=2b outside bounds");
      },
    },
    {
      name: "applyParamSeed updates value and latex",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([{ name: "a", latex: "a=1", exprId: "e1" }]);
        applyParamSeed({ a: { value: 3.5 } });
        assertNear(getParamValues().a ?? NaN, 3.5, 1e-9, "seed value");
        assert(getParam("a")?.latex.includes("3.5"), "seed latex");
      },
    },
    {
      name: "setParamValue without rewriting latex keeps driven flag",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        setParamValue("b", 4, { stopAnim: true, rewriteLatex: false });
        evalParamEquations();
        assertNear(getParamValues().a ?? NaN, 8, 1e-9, "driven updated");
      },
    },
    {
      name: "setParamValue immediately evals driven dependents",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        setParamValue("b", 5, { stopAnim: true, rewriteLatex: true });
        assertNear(getParamValues().a ?? NaN, 10, 1e-9, "a follows b without extra eval");
      },
    },
    {
      name: "collectValueDirtyParams detects static slider edits",
      fn: () => {
        const dirty = collectValueDirtyParams({ a: 1, b: 2 }, { a: 1.5, b: 2 });
        assert(dirty.has("a"), "changed a");
        assert(!dirty.has("b"), "unchanged b");
        assert(collectValueDirtyParams({ a: 1 }, { a: 1 }).size === 0, "equal");
        assert(collectValueDirtyParams({}, { a: 1 }).has("a"), "new param");
      },
    },
  ]);
}
