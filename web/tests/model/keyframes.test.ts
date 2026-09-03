import "../helpers/setup-dom.ts";
import { compileExpr } from "../../src/math/fit.ts";
import { compileVectorExpr } from "../../src/math/fitVector.ts";
import {
  allKeyframesComplete,
  beginKeyframePass,
  clearKeyframeCaches,
  DEFAULT_KEYFRAME_K,
  diagnoseKeyframeCaches,
  ensureLayerKeyframes,
  getIsoBlendSceneM,
  getKeyframeLoadSummary,
  getKeyframeProgress,
  getKeyframeMetrics,
  hasActiveKeyframeCaches,
  hasLayerKeyframeCache,
  keyframeAnimParam,
  keyframeAnimParams,
  keyframeFixedParamsFingerprint,
  resolveKeyframeParamNames,
  keyframesSplashReady,
  logKeyframeBake,
  noteKeyframeLayer,
  peekKeyframeBlend,
  sampleFlowLayerKeyframes,
  sampleLayerKeyframes,
  setKeyframeProgressHandler,
  syncKeyframeCachesWithExpressions,
  tickKeyframePump,
} from "../../src/model/keyframes.ts";
import { syncParamsFromDefinitions, updateParam, recompileParam, getParamValues, collectAnimDirtyParams } from "../../src/model/params.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function setupAnimParam(name: string, value = 0.5) {
  syncParamsFromDefinitions([
    {
      name,
      latex: `${name}=${value}`,
      exprId: "e1",
      min: 0,
      max: 1,
      speed: 0.5,
      animating: true,
      animMode: "pingpong",
    },
  ]);
}

function keyframeOpts(layerId: string, paramName: string, role: "cloud" | "isosurface" = "cloud") {
  const compiled = compileExpr(String.raw`\sin(x+${paramName})`);
  return {
    layerId,
    latex: String.raw`\sin(x+t)`,
    role,
    isoLevel: 0,
    paramName,
    compiled,
    baseParams: {},
    half: 0.75,
    deg: 4,
    K: 3,
  };
}

function flowKeyframeOpts(layerId: string, paramName: string) {
  const vectorCompiled = compileVectorExpr(String.raw`(${paramName}\cdot x,0,0)`);
  return {
    layerId,
    latex: String.raw`(t\cdot x,0,0)`,
    role: "flow" as const,
    paramName,
    vectorCompiled,
    baseParams: {},
    half: 0.75,
    deg: 4,
    K: 3,
  };
}

async function waitForComplete(timeoutMs = 8000) {
  const t0 = Date.now();
  while (!allKeyframesComplete()) {
    tickKeyframePump(32);
    if (Date.now() - t0 > timeoutMs) {
      throw new Error("keyframe async fill timed out");
    }
    await new Promise((r) => setTimeout(r, 0));
  }
}

async function waitForProgress(
  layerId: string,
  predicate: (p: NonNullable<ReturnType<typeof getKeyframeProgress>>) => boolean,
  timeoutMs = 8000,
) {
  const t0 = Date.now();
  while (true) {
    tickKeyframePump(32);
    const p = getKeyframeProgress(layerId);
    if (p && predicate(p)) return p;
    if (Date.now() - t0 > timeoutMs) throw new Error("waitForProgress timed out");
    await new Promise((r) => setTimeout(r, 0));
  }
}

export async function run() {
  return runSuite("model / keyframes", [
    {
      name: "keyframeAnimParam: single dirty anim param",
      fn: () => {
        setupAnimParam("t");
        updateParam("t", { animating: true });
        const hit = keyframeAnimParam(["t", "a"], new Set(["t"]));
        assert(hit === "t", "picks anim param");
      },
    },
    {
      name: "keyframeAnimParams: multiple dirty anim params",
      fn: () => {
        setupAnimParam("t");
        syncParamsFromDefinitions([
          {
            name: "t",
            latex: "t=0.5",
            exprId: "e1",
            min: 0,
            max: 1,
            speed: 0.5,
            animating: true,
            animMode: "pingpong",
          },
          {
            name: "a",
            latex: "a=0.5",
            exprId: "e2",
            min: 0,
            max: 1,
            speed: 0.4,
            animating: true,
            animMode: "pingpong",
          },
        ]);
        updateParam("a", { animating: true });
        const hit = keyframeAnimParams(["t", "a"], new Set(["t", "a"]));
        if (!hit || hit.length !== 2 || hit[0] !== "a" || hit[1] !== "t") {
          throw new Error(`expected [a,t] got ${JSON.stringify(hit)}`);
        }
      },
    },
    {
      name: "multi-param cloud uses keyframe path",
      fn: async () => {
        clearKeyframeCaches();
        syncParamsFromDefinitions([
          {
            name: "t",
            latex: "t=0.5",
            exprId: "e1",
            min: 0,
            max: 1,
            speed: 0.5,
            animating: true,
            animMode: "pingpong",
          },
          {
            name: "a",
            latex: "a=0.5",
            exprId: "e2",
            min: 0,
            max: 1,
            speed: 0.4,
            animating: true,
            animMode: "pingpong",
          },
        ]);
        const compiled = compileExpr(String.raw`\sin(x+t)\cos(y+a)`);
        const sample = sampleLayerKeyframes({
          layerId: "layer-multi",
          latex: String.raw`\sin(x+t)\cos(y+a)`,
          role: "cloud",
          isoLevel: 0,
          paramNames: ["a", "t"],
          compiled,
          baseParams: {},
          half: 0.75,
          deg: 4,
          K: 3,
        });
        assert(sample.dens.length > 0, "dens sampled");
        assert(sample.M > 0, "grid M");
        const prog = getKeyframeProgress("layer-multi");
        assert(prog != null, "cache exists");
        assert(prog!.totalFrames === 9, "3^2 grid");
      },
    },
    {
      name: "pausing one of two anim params reuses N-D cache",
      fn: async () => {
        clearKeyframeCaches();
        syncParamsFromDefinitions([
          {
            name: "t",
            latex: "t=0.5",
            exprId: "e1",
            min: 0,
            max: 1,
            speed: 0.5,
            animating: true,
            animMode: "pingpong",
          },
          {
            name: "a",
            latex: "a=0.5",
            exprId: "e2",
            min: 0,
            max: 1,
            speed: 0.4,
            animating: true,
            animMode: "pingpong",
          },
        ]);
        const latex = String.raw`\sin(x+t)\cos(y+a)`;
        const compiled = compileExpr(latex);
        const layerId = "layer-pause-one";
        const baseOpts = {
          layerId,
          latex,
          role: "cloud" as const,
          isoLevel: 0,
          paramNames: ["a", "t"],
          compiled,
          baseParams: {},
          half: 0.75,
          deg: 4,
          K: 3,
        };
        const first = sampleLayerKeyframes(baseOpts);
        assert(first.baked, "initial bake");
        const prog0 = getKeyframeProgress(layerId);
        assert(prog0?.totalFrames === 9, "2D grid");
        const ready0 = prog0!.readyCount;

        updateParam("a", { animating: false });
        const dirty = collectAnimDirtyParams();
        const kf = keyframeAnimParams(compiled.freeParams, dirty);
        assert(kf?.length === 1 && kf[0] === "t", "only t animating");
        const axes = resolveKeyframeParamNames(layerId, kf);
        assert(axes.length === 2 && axes[0] === "a" && axes[1] === "t", "reuse 2D axes");

        const second = sampleLayerKeyframes({ ...baseOpts, paramNames: axes, deferSyncBake: true });
        assert(!second.baked, "no rebake on pause");
        const prog1 = getKeyframeProgress(layerId);
        assert(prog1?.totalFrames === 9, "grid unchanged");
        assert(prog1!.readyCount === ready0, "ready count preserved");

        updateParam("a", { animating: true });
        const kf2 = keyframeAnimParams(compiled.freeParams, collectAnimDirtyParams());
        assert(kf2?.length === 2, "both animating again");
        const axes2 = resolveKeyframeParamNames(layerId, kf2!);
        const third = sampleLayerKeyframes({ ...baseOpts, paramNames: axes2, deferSyncBake: true });
        assert(!third.baked, "no rebake on resume");
      },
    },
    {
      name: "fixed param change rebuilds cache while another param animates",
      fn: async () => {
        clearKeyframeCaches();
        syncParamsFromDefinitions([
          {
            name: "t",
            latex: "t=0.5",
            exprId: "e1",
            min: 0,
            max: 1,
            value: 0.5,
            speed: 0.5,
            animating: true,
            animMode: "pingpong",
          },
          {
            name: "a",
            latex: "a=0.2",
            exprId: "e2",
            min: 0,
            max: 1,
            value: 0.2,
            animating: false,
            animMode: "pingpong",
          },
        ]);
        const latex = String.raw`\sin(x+t)\cos(y+a)`;
        const compiled = compileExpr(latex);
        const baseOpts = {
          layerId: "layer-fixed-param",
          latex,
          role: "cloud" as const,
          isoLevel: 0,
          paramNames: ["t"],
          compiled,
          half: 0.75,
          deg: 4,
          K: 3,
        };
        const first = sampleLayerKeyframes({ ...baseOpts, baseParams: getParamValues() });
        assert(first.baked, "initial bake");
        const fp0 = keyframeFixedParamsFingerprint(compiled.freeParams, ["t"], getParamValues());

        updateParam("a", { value: 0.85 });
        const fp1 = keyframeFixedParamsFingerprint(compiled.freeParams, ["t"], getParamValues());
        assert(fp0 !== fp1, "fp changes when fixed param moves");
        const second = sampleLayerKeyframes({ ...baseOpts, baseParams: getParamValues() });
        assert(second.baked, "rebake after fixed param moved");
        const prog = getKeyframeProgress("layer-fixed-param");
        assert(prog != null, "cache still present");
        assert(prog!.readyCount < prog!.totalFrames!, "grid refilling after invalidation");
      },
    },
    {
      name: "keyframeAnimParam: rejects multiple dirty or driven",
      fn: () => {
        syncParamsFromDefinitions([
          {
            name: "t",
            latex: "t=0.5",
            exprId: "e1",
            min: 0,
            max: 1,
            animating: true,
            animMode: "pingpong",
          },
          {
            name: "a",
            latex: "a=0.5",
            exprId: "e2",
            min: 0,
            max: 1,
            animating: true,
            animMode: "pingpong",
          },
        ]);
        assert(keyframeAnimParam(["t", "a"], new Set(["t", "a"])) === null, "multi dirty");
        syncParamsFromDefinitions([
          { name: "a", latex: "a=2b", exprId: "e2", min: -10, max: 10 },
          { name: "b", latex: "b=1", exprId: "e3", min: -10, max: 10 },
        ]);
        recompileParam("a");
        assert(keyframeAnimParam(["a"], new Set(["a"])) === null, "driven param");
      },
    },
    {
      name: "ensureLayerKeyframes sync-bakes blend pair",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.25);
        const result = ensureLayerKeyframes(keyframeOpts("layer-a", "t"));
        assert(result.baked, "sync baked");
        assert(result.readyCount >= 2, "blend pair ready");
        assert(result.frames.length === 3, "K=3 frames materialized");
        assert(hasActiveKeyframeCaches(), "cache active");
        assert(!result.complete, "async fill remains");
        assertNear(result.M, 5, 0.1, "grid M for deg=4");
      },
    },
    {
      name: "progressive: sync pair starts coarse at high target deg",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.25);
        const opts = { ...keyframeOpts("layer-prog-coarse", "t"), deg: 16, K: 3 };
        const result = ensureLayerKeyframes(opts);
        assert(result.baked, "sync baked");
        assertNear(result.M, 5, 0.1, "M=5 for coarse deg 4");
        assert(result.readyCount === 0, "not at target yet");
        const prog = getKeyframeProgress("layer-prog-coarse");
        assert(!!prog, "progress");
        assert(prog!.frameDeg[result.blend.i0] === 4, "i0 coarse");
        assert(prog!.frameDeg[result.blend.i1] === 4, "i1 coarse");
        assert(prog!.frameDeg[result.blend.i0] === prog!.frameDeg[result.blend.i1], "blend lockstep");
      },
    },
    {
      name: "progressive: async reaches target degree",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-prog-full", "t"), deg: 16, K: 3 };
        ensureLayerKeyframes(opts);
        await waitForComplete(20000);
        const prog = getKeyframeProgress("layer-prog-full");
        assert(!!prog, "progress");
        assert(prog!.readyCount === 3, "all K at target");
        for (let k = 0; k < 3; k++) {
          assert(prog!.frameDeg[k] === 16, `frame ${k} at target deg`);
        }
      },
    },
    {
      name: "progressive: active pair advances ahead of untouched slots",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-prog-phase", "t"), deg: 16, K: 3 };
        const result = ensureLayerKeyframes(opts);
        const afterSync = getKeyframeProgress("layer-prog-phase")!;
        const third = [0, 1, 2].find((k) => k !== result.blend.i0 && k !== result.blend.i1)!;
        assert(afterSync.frameDeg[third] === 0, "third slot still empty after sync");
        assert(afterSync.frameDeg[result.blend.i0] === 4, "blend coarse after sync");
        const t0 = Date.now();
        let sawPairAheadOfThird = false;
        while (!allKeyframesComplete()) {
          tickKeyframePump(1);
          const p = getKeyframeProgress("layer-prog-phase")!;
          if (
            p.frameDeg[result.blend.i0]! > p.frameDeg[third]! ||
            p.frameDeg[result.blend.i1]! > p.frameDeg[third]!
          ) {
            sawPairAheadOfThird = true;
          }
          if (Date.now() - t0 > 20000) {
            throw new Error(
              `active-pair fill timed out:\n${JSON.stringify(diagnoseKeyframeCaches(), null, 2)}`,
            );
          }
          await new Promise((r) => setTimeout(r, 0));
        }
        assert(
          sawPairAheadOfThird,
          "active pair should advance past the untouched third slot instead of waiting for it",
        );
      },
    },
    {
      name: "progressive: multi-layer fill completes independently",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        ensureLayerKeyframes({ ...keyframeOpts("layer-a", "t"), deg: 16, K: 3 });
        ensureLayerKeyframes({
          ...keyframeOpts("layer-b", "t"),
          layerId: "layer-b",
          latex: String.raw`\cos(x+t)`,
          compiled: compileExpr(String.raw`\cos(x+t)`),
          deg: 16,
          K: 3,
        });
        await waitForComplete(30000);
        assert(getKeyframeProgress("layer-a")!.readyCount === 3, "a complete");
        assert(getKeyframeProgress("layer-b")!.readyCount === 3, "b complete");
      },
    },
    {
      name: "progressive: regenerating one layer does not reset sibling",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        ensureLayerKeyframes({ ...keyframeOpts("layer-keep", "t"), deg: 8, K: 3 });
        ensureLayerKeyframes({
          ...keyframeOpts("layer-edit", "t"),
          layerId: "layer-edit",
          latex: String.raw`\cos(x+t)`,
          compiled: compileExpr(String.raw`\cos(x+t)`),
          deg: 8,
          K: 3,
        });
        await waitForComplete(20000);
        const keepBefore = getKeyframeProgress("layer-keep")!;
        assert(keepBefore.readyCount === 3, "keep ready");

        // Drop only the edited layer (simulates latex change sync).
        syncKeyframeCachesWithExpressions(
          [
            { id: "layer-keep", latex: String.raw`\sin(x+t)`, enabled: true },
            { id: "layer-edit", latex: String.raw`\cos(x+2t)`, enabled: true },
          ],
          { deg: 8, half: 0.75 },
        );
        assert(hasLayerKeyframeCache("layer-keep"), "keep cache survives");
        assert(!hasLayerKeyframeCache("layer-edit"), "edit cache dropped");

        ensureLayerKeyframes({
          ...keyframeOpts("layer-edit", "t"),
          layerId: "layer-edit",
          latex: String.raw`\cos(x+2t)`,
          compiled: compileExpr(String.raw`\cos(x+2t)`),
          deg: 8,
          K: 3,
        });
        const keepAfter = ensureLayerKeyframes({ ...keyframeOpts("layer-keep", "t"), deg: 8, K: 3 });
        assert(!keepAfter.baked, "keep not rebaked");
        assert(getKeyframeProgress("layer-keep")!.readyCount === 3, "keep still complete");
        assert(getKeyframeProgress("layer-keep")!.displayDeg.every((d) => d === 8), "keep deg");
      },
    },
    {
      name: "progressive: display blend keeps matched degrees",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-prog-match", "t"), deg: 16, K: 3 };
        ensureLayerKeyframes(opts);
        await waitForComplete(20000);
        const prog = getKeyframeProgress("layer-prog-match")!;
        const blend = peekKeyframeBlend("layer-prog-match");
        assert(!!blend, "blend peek");
        assert(
          prog.frameDeg[blend!.i0] === prog.frameDeg[blend!.i1],
          "display pair same degree",
        );
      },
    },
    {
      name: "progressive: blend pair stays lockstep during async fill",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-prog-lock", "t"), deg: 16, K: 3 };
        ensureLayerKeyframes(opts);
        await waitForProgress("layer-prog-lock", (p) => {
          const blend = peekKeyframeBlend("layer-prog-lock");
          if (!blend) return false;
          return p.frameDeg[blend.i0]! >= 8 && p.frameDeg[blend.i1]! >= 8;
        });
        const prog = getKeyframeProgress("layer-prog-lock")!;
        const { i0, i1 } = peekKeyframeBlend("layer-prog-lock")!;
        assert(prog.frameDeg[i0] === prog.frameDeg[i1], "lockstep during refine");
      },
    },
    {
      name: "progressive: current pair reaches target without waiting on distant slots",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t");
        updateParam("t", { value: 0, min: 0, max: 1 });
        const opts = { ...keyframeOpts("layer-play-coarse", "t"), deg: 16, K: 5 };
        ensureLayerKeyframes(opts);
        const t0 = Date.now();
        let sawPairAheadOfDistantSlot = false;
        while (!allKeyframesComplete()) {
          tickKeyframePump(1);
          const p = getKeyframeProgress("layer-play-coarse")!;
          // Displayed degree only ever moves forward, per slot.
          for (let k = 0; k < p.displayDeg.length; k++) {
            assert(p.displayDeg[k]! <= p.frameDeg[k]!, `no regression at slot ${k}`);
          }
          // Pair near t=0 (slots 0,1) should be able to outpace a distant,
          // never-visited slot (4) instead of waiting for it to catch up.
          if (p.frameDeg[0]! > 0 && p.frameDeg[4]! < p.frameDeg[0]!) {
            sawPairAheadOfDistantSlot = true;
          }
          if (Date.now() - t0 > 25000) {
            throw new Error(
              `play-coarse fill timed out:\n${JSON.stringify(diagnoseKeyframeCaches(), null, 2)}`,
            );
          }
          await new Promise((r) => setTimeout(r, 0));
        }
        assert(
          sawPairAheadOfDistantSlot,
          "current pair should progress past a distant, unvisited slot instead of waiting for it",
        );
      },
    },
    {
      name: "progressive: continuous playhead movement does not starve in-flight work",
      fn: async () => {
        // Regression test: prioritizing the active pair (previous test) is
        // recomputed fresh every pump tick from the *current* param value.
        // During real playback that value moves every frame, so the active
        // pair drifts continuously — if in-flight work is only ever checked
        // for the current pair, a job started for slot k gets abandoned the
        // instant the pair drifts past it, and progress scatters across many
        // slots without ever finishing any (visible as quality jumping
        // around / freezing instead of advancing smoothly).
        clearKeyframeCaches();
        setupAnimParam("t");
        updateParam("t", { value: 0, min: 0, max: 1 });
        const opts = { ...keyframeOpts("layer-continuous-play", "t"), deg: 16, K: 8 };
        ensureLayerKeyframes(opts);
        const t0 = Date.now();
        // Sweep the playhead forward across the whole range exactly once —
        // never revisiting slot 0's neighborhood — like a single play-through
        // rather than scrubbing back and forth (which would let a starved
        // slot get "rediscovered" once the sweep wraps around and masks the
        // bug). Slot 0 gets real work only in the first few ticks, then the
        // pair moves on for good.
        const sweepTicks = opts.K - 1; // one tick per segment boundary — minimal dwell time per pair
        for (let i = 0; i <= sweepTicks; i++) {
          updateParam("t", { value: i / sweepTicks });
          tickKeyframePump(1);
          if (Date.now() - t0 > 25000) throw new Error("sweep itself timed out");
          await new Promise((r) => setTimeout(r, 0));
        }
        const p = getKeyframeProgress("layer-continuous-play")!;
        assert(
          (p.displayDeg[0] ?? 0) >= 8,
          `slot 0 (visited only at the very start of the sweep) should have kept making real progress instead of being abandoned once the playhead moved on — got displayDeg=${p.displayDeg[0]}:\n${JSON.stringify(diagnoseKeyframeCaches(), null, 2)}`,
        );
      },
    },
    {
      name: "progressive: neighboring segment gets a head start before the playhead reaches it",
      fn: async () => {
        // Regression test: once the active pair (i0,i1) has nothing left to
        // do, the scheduler should start warming the *next* segment the
        // playhead is heading toward instead of sitting idle — otherwise
        // crossing into that segment starts it from zero baked data, and the
        // display visibly drops a rung right at the boundary ("rung
        // hopping").
        clearKeyframeCaches();
        setupAnimParam("t");
        updateParam("t", { value: 0.5, min: 0, max: 1 });
        const opts = { ...keyframeOpts("layer-lookahead", "t"), deg: 16, K: 5 };
        ensureLayerKeyframes(opts);
        const blend = peekKeyframeBlend("layer-lookahead")!;
        const t0 = Date.now();
        while (true) {
          tickKeyframePump(1);
          const p = getKeyframeProgress("layer-lookahead")!;
          if (p.frameDeg[blend.i0]! >= opts.deg && p.frameDeg[blend.i1]! >= opts.deg) break;
          if (Date.now() - t0 > 25000) {
            throw new Error(
              `active pair never completed:\n${JSON.stringify(diagnoseKeyframeCaches(), null, 2)}`,
            );
          }
          await new Promise((r) => setTimeout(r, 0));
        }
        // Give the scheduler a few more ticks to pick up lookahead work once
        // the active pair is done.
        for (let i = 0; i < 5; i++) tickKeyframePump(1);
        const p = getKeyframeProgress("layer-lookahead")!;
        const neighbor = blend.i1 + 1 < opts.K ? blend.i1 + 1 : blend.i0 - 1;
        assert(
          (p.frameDeg[neighbor] ?? 0) > 0,
          `neighboring segment ${neighbor} should get a head start once the active pair finishes — got frameDeg=${p.frameDeg[neighbor]}:\n${JSON.stringify(diagnoseKeyframeCaches(), null, 2)}`,
        );
      },
    },
    {
      name: "display resolution stays with the shown side while its neighbor is freshly (re)baking",
      fn: async () => {
        // Regression test: displayBlendForValue snaps t to exactly 0 or 1
        // whenever degrees mismatch, meaning only ONE side is ever actually
        // shown — isoBlendSceneM must follow that side's resolution rather
        // than always falling back to the lower of the two, or the display
        // visibly drops quality the instant an invisible neighbor gets any
        // bake at all (a "rung hop") even though nothing is being blended.
        //
        // segmentForValue always returns an adjacent pair (i0, i0+1), never
        // i0===i1, so a single value can't isolate one slot — instead, fully
        // bake segment (0,1), then step into segment (1,2): slot 1 carries
        // over fully baked, slot 2 starts completely fresh beside it.
        clearKeyframeCaches();
        setupAnimParam("t");
        updateParam("t", { value: 0, min: 0, max: 1 });
        const opts = { ...keyframeOpts("layer-blend-m", "t"), deg: 32, K: 3 };
        ensureLayerKeyframes(opts);
        const t0 = Date.now();
        while (
          (getKeyframeProgress("layer-blend-m")!.displayDeg[0] ?? 0) < opts.deg ||
          (getKeyframeProgress("layer-blend-m")!.displayDeg[1] ?? 0) < opts.deg
        ) {
          tickKeyframePump(1);
          if (Date.now() - t0 > 20000) throw new Error("segment (0,1) never reached target");
          await new Promise((r) => setTimeout(r, 0));
        }
        // Step to the boundary between segments (0,1) and (1,2): slot 1 is
        // now shown alone (t=0), slot 2 hasn't been touched yet.
        updateParam("t", { value: 0.5 });
        const mFull = getIsoBlendSceneM("layer-blend-m");
        assert(mFull > 0, "slot 1 should have a real grid resolution once baked");
        const t1 = Date.now();
        while ((getKeyframeProgress("layer-blend-m")!.displayDeg[2] ?? 0) <= 0) {
          tickKeyframePump(1);
          if (Date.now() - t1 > 20000) throw new Error("slot 2 never got its first bake");
          await new Promise((r) => setTimeout(r, 0));
        }
        const prog = getKeyframeProgress("layer-blend-m")!;
        assert(prog.displayDeg[1] === opts.deg, "slot 1 should still be at full target degree");
        assert(
          prog.displayDeg[2]! > 0 && prog.displayDeg[2]! < opts.deg,
          "slot 2 should be freshly, partially baked",
        );
        const mAfter = getIsoBlendSceneM("layer-blend-m");
        assert(
          mAfter === mFull,
          `display is snapped to slot 1 alone here, so its resolution shouldn't drop just because slot 2 started baking at a coarser degree — got ${mAfter}, expected ${mFull}`,
        );
      },
    },
    {
      name: "progressive: cache resets when target deg changes",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-prog-deg", "t"), deg: 8, K: 3 };
        ensureLayerKeyframes(opts);
        const second = ensureLayerKeyframes({ ...opts, deg: 16 });
        assert(second.baked, "rebuilt on deg change");
        const prog = getKeyframeProgress("layer-prog-deg");
        assert(!!prog, "progress");
        assert(prog!.targetDeg === 16, "new target");
        assert(prog!.frameDeg[second.blend.i0] === 4, "restarted coarse");
      },
    },
    {
      name: "progressive: flow completes at target degree",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...flowKeyframeOpts("layer-flow-prog", "t"), deg: 16, K: 3 };
        const first = ensureLayerKeyframes(opts);
        assertNear(first.M, 5, 0.1, "coarse flow grid");
        await waitForComplete(30000);
        const prog = getKeyframeProgress("layer-flow-prog");
        assert(!!prog, "progress");
        assert(prog!.readyCount === 3, "flow K complete");
        assert(prog!.frameDeg.every((d) => d === 16), "all flow frames at target");
      },
    },
    {
      name: "sampleLayerKeyframes lerps density",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = keyframeOpts("layer-b", "t");
        ensureLayerKeyframes(opts);
        const sampled = sampleLayerKeyframes(opts);
        assert(sampled.dens.length > 0, "lerped dens");
        assert(Number.isFinite(sampled.dens[0]!), "finite dens");
      },
    },
    {
      name: "isosurface keyframes include gradient channels",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = keyframeOpts("layer-iso", "t", "isosurface");
        const result = ensureLayerKeyframes(opts);
        const frame = result.frames[0];
        assert(!!frame?.gx && !!frame?.gy && !!frame?.gz, "grad channels");
      },
    },
    {
      name: "peekKeyframeBlend returns segment when pair ready",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = keyframeOpts("layer-c", "t");
        ensureLayerKeyframes(opts);
        const blend = peekKeyframeBlend("layer-c");
        assert(!!blend, "blend peek");
        assert(blend!.t >= 0 && blend!.t <= 1, "t in range");
      },
    },
    {
      name: "async fill completes all K frames",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const progress: number[] = [];
        let done = false;
        setKeyframeProgressHandler((info) => {
          if (info.index >= 0) progress.push(info.index);
          if (info.done) done = true;
        });
        ensureLayerKeyframes(keyframeOpts("layer-async", "t", "cloud"));
        await waitForComplete();
        assert(allKeyframesComplete(), "all frames ready");
        assert(progress.length >= 1, "progress callbacks");
        assert(done, "done progress");
        setKeyframeProgressHandler(null);
      },
    },
    {
      name: "getKeyframeLoadSummary fraction is monotonic during async fill",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        ensureLayerKeyframes({ ...keyframeOpts("layer-mono", "t"), deg: 16, K: 5 });
        let hi = 0;
        const t0 = Date.now();
        while (!allKeyframesComplete()) {
          tickKeyframePump(32);
          const { fraction } = getKeyframeLoadSummary();
          assert(
            fraction + 1e-9 >= hi,
            `load fraction regressed ${hi.toFixed(4)} → ${fraction.toFixed(4)}`,
          );
          hi = fraction;
          if (Date.now() - t0 > 20000) throw new Error("monotonic fill timed out");
          await new Promise((r) => setTimeout(r, 0));
        }
        assert(hi >= 0.99, "reaches full");
      },
    },
    {
      name: "probe: fill completes while anim param sweeps segments",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0);
        ensureLayerKeyframes({ ...keyframeOpts("layer-stall-sweep", "t"), deg: 16, K: 5 });
        const values = [0, 0.15, 0.35, 0.55, 0.75, 0.95, 0.4, 0.1];
        let vi = 0;
        const t0 = Date.now();
        let stalls = 0;
        while (!allKeyframesComplete()) {
          updateParam("t", { value: values[vi % values.length]! });
          vi++;
          const n = tickKeyframePump(8);
          const diag = diagnoseKeyframeCaches()[0];
          if (diag?.stalled) {
            stalls++;
            if (stalls > 3) {
              throw new Error(
                `keyframe stall while sweeping t:\n${JSON.stringify(diag, null, 2)}`,
              );
            }
          } else {
            stalls = 0;
          }
          if (n === 0 && !allKeyframesComplete() && Date.now() - t0 > 500) {
            const d = diagnoseKeyframeCaches();
            throw new Error(`pump idle before complete:\n${JSON.stringify(d, null, 2)}`);
          }
          if (Date.now() - t0 > 25000) {
            throw new Error(
              `sweep fill timed out:\n${JSON.stringify(diagnoseKeyframeCaches(), null, 2)}`,
            );
          }
          await new Promise((r) => setTimeout(r, 0));
        }
        const done = getKeyframeProgress("layer-stall-sweep");
        assert(!!done && done.readyCount === 5, "all slots ready after sweep");
        assert(done!.displayDeg.every((d) => d === 16), "display at target");
      },
    },
    {
      name: "probe: diagnose reports display vs staging split",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        ensureLayerKeyframes({ ...keyframeOpts("layer-diag", "t"), deg: 16, K: 3 });
        tickKeyframePump(4);
        const diag = diagnoseKeyframeCaches();
        assert(diag.length === 1, "one layer");
        assert(diag[0]!.slots.length === 3, "K slots");
        assert(typeof diag[0]!.blend.i0 === "number", "blend indices");
        await waitForComplete(20000);
        const after = diagnoseKeyframeCaches()[0]!;
        assert(!after.stalled, "not stalled when complete");
        assert(after.readyCount === 3, "ready");
      },
    },
    {
      name: "getKeyframeLoadSummary tracks coarse then full fill",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const idle = getKeyframeLoadSummary();
        assert(idle.complete, "idle complete");
        ensureLayerKeyframes({ ...keyframeOpts("layer-load-bar", "t"), deg: 16, K: 3 });
        const coarse = getKeyframeLoadSummary();
        assert(coarse.active, "loading after sync");
        assert(coarse.fraction > 0 && coarse.fraction < 1, "partial after coarse");
        assert(coarse.slotsTotal === 3, "K slots");
        await waitForComplete(20000);
        const done = getKeyframeLoadSummary();
        assert(done.complete, "done loading");
        assert(done.fraction >= 0.99, "full fraction");
        assert(done.slotsAtTarget === 3, "all at target");
      },
    },
    {
      name: "keyframesSplashReady after sync blend pair",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        ensureLayerKeyframes(keyframeOpts("layer-splash", "t"));
        assert(!allKeyframesComplete(), "async fill remains");
        assert(keyframesSplashReady(), "coarse blend enough for splash");
      },
    },
    {
      name: "clearKeyframeCaches resets state",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        ensureLayerKeyframes(keyframeOpts("layer-d", "t"));
        assert(hasActiveKeyframeCaches(), "has cache");
        clearKeyframeCaches();
        assert(!hasActiveKeyframeCaches(), "cleared");
        assert(allKeyframesComplete(), "vacuously complete");
      },
    },
    {
      name: "ensureLayerKeyframes forces sync when defer would leave empty pair",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-defer-guard", "t"), deg: 8, K: 3 };
        // First ensure creates cache; second call with mismatched latex rebuilds empty
        // while deferSyncBake=true — must still sync-bake the blend pair.
        ensureLayerKeyframes(opts);
        const rebuilt = ensureLayerKeyframes({
          ...opts,
          latex: String.raw`\cos(x+t)`,
          compiled: compileExpr(String.raw`\cos(x+t)`),
          deferSyncBake: true,
        });
        assert(rebuilt.baked, "forced sync bake");
        assert(rebuilt.M >= 5, `valid grid M, got ${rebuilt.M}`);
        const blend = peekKeyframeBlend("layer-defer-guard");
        assert(!!blend, "blend pair ready after forced sync");
      },
    },
    {
      name: "sweep to unloaded pair keeps playing at ready rung",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t");
        updateParam("t", { value: 0, min: 0, max: 1 });
        const opts = { ...keyframeOpts("e41", "t"), deg: 16, K: 5 };
        const first = ensureLayerKeyframes(opts);
        assert(first.blend.i0 === 0 && first.blend.i1 === 1, "starts on first segment");
        assert(first.M >= 5, "coarse grid");
        updateParam("t", { value: 1 });
        const peekHold = peekKeyframeBlend("e41");
        assert(!!peekHold, "peek stays ready without ensure");
        const swept = ensureLayerKeyframes({ ...opts, deferSyncBake: true });
        assert(swept.M >= 5, `kept display grid, got M=${swept.M}`);
        assert(swept.frames.length === 5, "materialized K");
        const blend = peekKeyframeBlend("e41");
        assert(!!blend, "GPU blend stays ready while (3,4) loads");
        assert(blend!.i0 >= 0 && blend!.i1 >= 0, "valid indices");
      },
    },
    {
      name: "sync bake promotes first missing blend partner",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t");
        updateParam("t", { value: 0, min: 0, max: 1 });
        const opts = { ...keyframeOpts("layer-partner-promote", "t"), deg: 8, K: 5 };
        ensureLayerKeyframes(opts);
        updateParam("t", { value: 1 });
        const forced = ensureLayerKeyframes(opts);
        const { i0, i1 } = forced.blend;
        assert(!!forced.rawFrames[i0], `slot ${i0} display-ready`);
        assert(!!forced.rawFrames[i1], `slot ${i1} display-ready`);
        const peek = peekKeyframeBlend("layer-partner-promote");
        assert(!!peek, "pair peekable");
        assert(peek!.i0 !== peek!.i1 || peek!.t === 0, "real pair or hold");
      },
    },
    {
      name: "ensureLayerKeyframes reuses complete cache on anim restart",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-restart", "t"), deg: 8, K: 3 };
        const first = ensureLayerKeyframes(opts);
        assert(first.baked, "initial sync bake");
        await waitForComplete(20000);
        assert(hasLayerKeyframeCache("layer-restart"), "memory cache");
        const restart = ensureLayerKeyframes({ ...opts, deferSyncBake: true });
        assert(!restart.baked, "no rebake on restart");
        assert(restart.complete, "still complete");
        assert(restart.readyCount === 3, "ready count preserved");
        const load = getKeyframeLoadSummary();
        assert(load.complete, "load bar done after restart");
      },
    },
    {
      name: "sync: progressive step deg must not drop keyframe cache",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-prog-pause", "t"), deg: 16, K: 3 };
        ensureLayerKeyframes(opts);
        await waitForComplete(20000);
        // Mimic pause progressive dens ladder: sync with intermediate fitDeg-like values
        // while UI target remains 16 (what pipeline must pass).
        syncKeyframeCachesWithExpressions(
          [{ id: "layer-prog-pause", latex: opts.latex, enabled: true }],
          { deg: 16, half: opts.half },
        );
        assert(hasLayerKeyframeCache("layer-prog-pause"), "kept at ui deg");
        // Wrong: syncing with progressive step deg would drop — document the contract.
        syncKeyframeCachesWithExpressions(
          [{ id: "layer-prog-pause", latex: opts.latex, enabled: true }],
          { deg: 4, half: opts.half },
        );
        assert(!hasLayerKeyframeCache("layer-prog-pause"), "step deg drops (caller bug)");
      },
    },
    {
      name: "sync: park on disable reuses cache on re-enable",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-park", "t"), deg: 8, K: 3 };
        ensureLayerKeyframes(opts);
        await waitForComplete(20000);
        const before = getKeyframeProgress("layer-park");
        assert(before?.readyCount === 3, "ready before park");

        syncKeyframeCachesWithExpressions(
          [{ id: "layer-park", latex: opts.latex, enabled: false }],
          { deg: opts.deg, half: opts.half },
        );
        assert(!hasActiveKeyframeCaches(), "parked not active");
        assert(allKeyframesComplete(), "parked ignored for complete");
        assert(getKeyframeLoadSummary().active === false, "load bar idle while parked");
        assert(diagnoseKeyframeCaches()[0]?.parked === true, "diag parked");

        syncKeyframeCachesWithExpressions(
          [{ id: "layer-park", latex: opts.latex, enabled: true }],
          { deg: opts.deg, half: opts.half },
        );
        const reused = ensureLayerKeyframes(opts);
        assert(!reused.baked, "no rebake on re-enable");
        const after = getKeyframeProgress("layer-park");
        assert(after?.readyCount === 3, "ready after unpark");
        assert(after?.displayDeg.every((d) => d === 8), "display preserved");
      },
    },
    {
      name: "sync: latex change while disabled discards cache",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-discard", "t"), deg: 8, K: 3 };
        ensureLayerKeyframes(opts);
        await waitForComplete(20000);

        syncKeyframeCachesWithExpressions(
          [{ id: "layer-discard", latex: opts.latex, enabled: false }],
          { deg: opts.deg, half: opts.half },
        );
        syncKeyframeCachesWithExpressions(
          [{ id: "layer-discard", latex: String.raw`\cos(x+t)`, enabled: false }],
          { deg: opts.deg, half: opts.half },
        );
        assert(diagnoseKeyframeCaches().length === 0, "discarded while disabled");

        const next = ensureLayerKeyframes({
          ...opts,
          latex: String.raw`\cos(x+t)`,
          compiled: compileExpr(String.raw`\cos(x+t)`),
        });
        assert(next.baked, "first enable after change bakes");
      },
    },
    {
      name: "sync: deg change discards without baking disabled layers",
      fn: async () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = { ...keyframeOpts("layer-deg-drop", "t"), deg: 8, K: 3 };
        ensureLayerKeyframes(opts);
        await waitForComplete(20000);
        syncKeyframeCachesWithExpressions(
          [{ id: "layer-deg-drop", latex: opts.latex, enabled: false }],
          { deg: 16, half: opts.half },
        );
        assert(diagnoseKeyframeCaches().length === 0, "deg mismatch drops");
        assert(allKeyframesComplete(), "no active work");
      },
    },
    {
      name: "beginKeyframePass and metrics accounting",
      fn: () => {
        clearKeyframeCaches();
        beginKeyframePass();
        setupAnimParam("t", 0.5);
        noteKeyframeLayer();
        ensureLayerKeyframes(keyframeOpts("layer-e", "t"));
        sampleLayerKeyframes(keyframeOpts("layer-e", "t"));
        const m = getKeyframeMetrics();
        assert(m.layers >= 1, "layer count");
        assert(m.K === DEFAULT_KEYFRAME_K, "default K in metrics");
        assert(m.bakeMs >= 0, "bake ms tracked");
      },
    },
    {
      name: "cache rebuilds when latex changes",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = keyframeOpts("layer-f", "t");
        const first = ensureLayerKeyframes(opts);
        const second = ensureLayerKeyframes({ ...opts, latex: String.raw`\cos(x+t)` });
        assert(second.baked, "rebuilt on latex change");
        assert(second.readyCount >= 2, "pair rebaked");
        assert(first.frames !== second.frames, "new frame array");
      },
    },
    {
      name: "flow keyframes bake velocity grids",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const result = ensureLayerKeyframes(flowKeyframeOpts("layer-flow", "t"));
        assert(result.baked, "sync baked");
        assert(result.readyCount >= 2, "blend pair ready");
        const frame = result.frames[0];
        assert(!!frame?.fx && !!frame?.fy && !!frame?.fz, "velocity channels");
        assert(!frame?.dens, "flow frame has no dens");
      },
    },
    {
      name: "sampleFlowLayerKeyframes lerps fx/fy/fz",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const opts = flowKeyframeOpts("layer-flow-sample", "t");
        ensureLayerKeyframes(opts);
        const sampled = sampleFlowLayerKeyframes(opts);
        assert(sampled.fx.length > 0, "lerped fx");
        assert(sampled.fy.length === sampled.fx.length, "matching fy");
        assert(sampled.fz.length === sampled.fx.length, "matching fz");
        assert(Number.isFinite(sampled.fx[0]!), "finite fx");
      },
    },
    {
      name: "logKeyframeBake prints bake summary",
      fn: () => {
        clearKeyframeCaches();
        setupAnimParam("t", 0.5);
        const logs: string[] = [];
        const origLog = console.log;
        console.log = (msg: string) => logs.push(String(msg));
        try {
          logKeyframeBake("test");
          assert(logs.length === 0, "no-op without bake");
          ensureLayerKeyframes(keyframeOpts("layer-log", "t"));
          logKeyframeBake("test");
          assert(logs.some((l) => l.includes("[keyframes]")), "logged bake");
        } finally {
          console.log = origLog;
        }
      },
    },
  ]);
}
