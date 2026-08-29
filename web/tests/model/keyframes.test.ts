import "../helpers/setup-dom.ts";
import { compileExpr } from "../../src/math/fit.ts";
import { compileVectorExpr } from "../../src/math/fitVector.ts";
import {
  allKeyframesComplete,
  beginKeyframePass,
  clearKeyframeCaches,
  DEFAULT_KEYFRAME_K,
  ensureLayerKeyframes,
  getKeyframeMetrics,
  hasActiveKeyframeCaches,
  keyframeAnimParam,
  logKeyframeBake,
  noteKeyframeLayer,
  peekKeyframeBlend,
  sampleFlowLayerKeyframes,
  sampleLayerKeyframes,
  setKeyframeProgressHandler,
} from "../../src/model/keyframes.ts";
import { syncParamsFromDefinitions, updateParam } from "../../src/model/params.ts";
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
    if (Date.now() - t0 > timeoutMs) {
      throw new Error("keyframe async fill timed out");
    }
    await new Promise((r) => setTimeout(r, 15));
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
      name: "keyframeAnimParam: rejects multiple dirty or driven",
      fn: () => {
        setupAnimParam("t");
        assert(keyframeAnimParam(["t", "a"], new Set(["t", "a"])) === null, "multi dirty");
        syncParamsFromDefinitions([
          { name: "a", latex: "a=2b", exprId: "e2", min: -10, max: 10 },
          { name: "b", latex: "b=1", exprId: "e3", min: -10, max: 10 },
        ]);
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
