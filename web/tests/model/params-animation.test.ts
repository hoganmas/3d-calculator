import "../helpers/setup-dom.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import { clearExpressions, setExpressions } from "../../src/model/expressions.ts";
import {
  anyParamAnimating,
  collectAnimDirtyParams,
  evalParamEquations,
  getParam,
  getParamValues,
  normalizeAnimMode,
  phaseForValue,
  setParamValue,
  stopParamAnimation,
  syncParamsFromDefinitions,
  tickParamAnimation,
  toggleParamAnimate,
  updateParam,
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
        setParamValue("b", 3, { stopAnim: true, rewriteLatex: false });
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
  ]);
}
