import "../helpers/setup-dom.ts";
import { readParamPlayChrome } from "../../src/ui/expr-sidebar/paramChrome.ts";
import { neededParamForItem } from "../../src/ui/expr-sidebar/helpers.ts";
import { clearExpressions, setExpressions, listExpressions } from "../../src/model/expressions.ts";
import { compileAllExprs } from "../../src/app/compile.ts";
import {
  getParam,
  setParamValue,
  stopParamAnimation,
  syncParamsFromDefinitions,
  toggleParamAnimate,
  updateParam,
} from "../../src/model/params.ts";
import { state } from "../../src/app/state.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function resetParams() {
  state.pendingParamSeed = {};
  state.exprListApi = null;
  clearExpressions();
  syncParamsFromDefinitions([]);
}

function chrome(name: string, tick: number) {
  return readParamPlayChrome(name, tick);
}

function rail(item: { id: string; latex: string }, tick: number) {
  return neededParamForItem(item as import("../../src/types/models.ts").ExprItem, tick);
}

export async function run() {
  return runSuite("ui / param-chrome", [
    {
      name: "idle param shows play icon",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.5", exprId: "e1", min: 0, max: 1, animating: false },
        ]);
        const c0 = chrome("a", 0);
        assert(c0.visible, "visible");
        assert(c0.icon === "▶", "play icon");
        assert(!c0.animating, "not animating");
        assert(!c0.disabled, "enabled");
      },
    },
    {
      name: "toggle animation flips icon on next tick",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.5", exprId: "e1", min: 0, max: 1, animating: false },
        ]);
        toggleParamAnimate("a", 0);
        const c1 = chrome("a", 1);
        assert(c1.icon === "⏸", "pause icon after start");
        assert(c1.animating, "animating");
        assert(c1.title.includes("Pause"), "pause title");

        stopParamAnimation("a");
        const c2 = chrome("a", 2);
        assert(c2.icon === "▶", "play icon after stop");
        assert(!c2.animating, "stopped");
      },
    },
    {
      name: "driven param keeps play icon and stays disabled",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        toggleParamAnimate("a", 0);
        const c = chrome("a", 0);
        assert(c.driven, "driven");
        assert(c.disabled, "disabled");
        assert(c.icon === "▶", "no pause for driven");
        assert(!c.animating, "not animating");
        assert(c.title.includes("Driven"), "driven title");
        assert(getParam("a")?.animating !== true, "model stays off");
      },
    },
    {
      name: "slider drag stops animation chrome",
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
          },
        ]);
        assert(chrome("a", 0).icon === "⏸", "starts paused icon");
        setParamValue("a", 3, { stopAnim: true, rewriteLatex: true });
        const c = chrome("a", 1);
        assert(c.icon === "▶", "play after drag");
        assert(!c.animating, "not animating");
      },
    },
    {
      name: "animating flag on driven param is ignored for chrome",
      fn: () => {
        resetParams();
        setExpressions([
          { id: "e1", latex: "b=1", enabled: true },
          { id: "e2", latex: "a=2b", enabled: true },
        ]);
        compileAllExprs({ rebuildUi: false });
        updateParam("a", { animating: true });
        const c = chrome("a", 0);
        assert(c.driven, "driven");
        assert(c.icon === "▶", "play despite stale animating flag");
        assert(!c.animating, "chrome not animating");
      },
    },
    {
      name: "missing param is hidden",
      fn: () => {
        resetParams();
        const c = chrome("missing", 0);
        assert(!c.visible, "hidden");
        assert(c.icon === "▶", "default icon");
      },
    },
    {
      name: "double toggle returns to play icon",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.5", exprId: "e1", min: 0, max: 1, animating: false },
        ]);
        toggleParamAnimate("a", 0);
        assert(chrome("a", 1).icon === "⏸", "pause after first toggle");
        toggleParamAnimate("a", 0);
        const c = chrome("a", 2);
        assert(c.icon === "▶", "play after second toggle");
        assert(!c.animating, "not animating");
      },
    },
    {
      name: "param rail appears once param map is synced",
      fn: () => {
        resetParams();
        const item = { id: "e1", latex: "a=0.5" };
        assert(rail(item, 0) === null, "no rail before param exists");
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.5", exprId: "e1", min: 0, max: 1 },
        ]);
        assert(rail(item, 0) === "a", "rail visible after param sync");
      },
    },
    {
      name: "param rail hidden when exprId does not own param",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.5", exprId: "other", min: 0, max: 1 },
        ]);
        const item = { id: "e1", latex: "a=0.5" };
        assert(rail(item, 1) === null, "wrong owner hides rail");
      },
    },
    {
      name: "param rail hidden for non-parameter rows",
      fn: () => {
        resetParams();
        syncParamsFromDefinitions([
          { name: "a", latex: "a=0.5", exprId: "e1", min: 0, max: 1 },
        ]);
        const item = { id: "e1", latex: "a x^2" };
        assert(rail(item, 1) === null, "field row has no rail");
      },
    },
    {
      name: "auto param row gets rail after deferred compile",
      fn: () => {
        resetParams();
        setExpressions([{ id: "e1", latex: "b x", enabled: true }]);
        compileAllExprs({ rebuildUi: false });
        const auto = listExpressions().find((e) => String(e.latex || "").startsWith("b="));
        assert(!!auto, "auto row created");
        assert(rail({ id: auto!.id, latex: auto!.latex }, 1) === "b", "auto row shows rail");
      },
    },
  ]);
}
