import "../helpers/setup-dom.ts";
import {
  bootQualityForTier,
  detectDeviceTier,
  webGpuPowerPreference,
  isoComposeDownscaleFloor,
  effectiveIsoComposeDownscale,
  coarseIsoSteps,
  clampIsoStepsForTier,
} from "../../src/app/deviceTier.ts";
import { qualityToTrailSteps } from "../../src/app/qualityMapping.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function mockMatchMedia(queries: Record<string, boolean>) {
  const prev = globalThis.matchMedia;
  globalThis.matchMedia = ((query: string) => ({
    matches: !!queries[query],
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
  })) as typeof matchMedia;
  return () => {
    globalThis.matchMedia = prev;
  };
}

export async function run() {
  return runSuite("app / device-tier", [
    {
      name: "bootQualityForTier maps mobile/tablet/desktop presets",
      fn: () => {
        const mobile = bootQualityForTier("mobile");
        assert(mobile.scalarQuality === 25, "mobile scalar");
        assert(mobile.vectorQuality === 20, "mobile vector");
        assert(qualityToTrailSteps(mobile.vectorQuality) === 32, "mobile boot keeps full trails");
        const tablet = bootQualityForTier("tablet");
        assert(tablet.scalarQuality === 40, "tablet scalar");
        const desktop = bootQualityForTier("desktop");
        assert(desktop.scalarQuality === 50, "desktop scalar");
      },
    },
    {
      name: "detectDeviceTier: coarse + narrow → mobile",
      fn: () => {
        const restore = mockMatchMedia({
          "(max-width: 800px)": true,
          "(pointer: coarse)": true,
        });
        try {
          assert(detectDeviceTier() === "mobile", "mobile tier");
        } finally {
          restore();
        }
      },
    },
    {
      name: "detectDeviceTier: coarse + wide → tablet",
      fn: () => {
        const restore = mockMatchMedia({
          "(max-width: 800px)": false,
          "(pointer: coarse)": true,
        });
        try {
          assert(detectDeviceTier() === "tablet", "tablet tier");
        } finally {
          restore();
        }
      },
    },
    {
      name: "webGpuPowerPreference prefers low-power on mobile",
      fn: () => {
        assert(webGpuPowerPreference("mobile") === "low-power", "mobile low-power");
        assert(webGpuPowerPreference("desktop") === "high-performance", "desktop perf");
      },
    },
    {
      name: "mobile compose floor is 4× and raises a 2× slider",
      fn: () => {
        assert(isoComposeDownscaleFloor("mobile") === 4, "mobile floor");
        assert(isoComposeDownscaleFloor("tablet") === 2, "tablet floor");
        assert(isoComposeDownscaleFloor("desktop") === 1, "desktop no floor");
        assert(effectiveIsoComposeDownscale(2, "mobile") === 4, "2× slider becomes 4×");
        assert(effectiveIsoComposeDownscale(8, "mobile") === 8, "8× stays 8×");
        assert(effectiveIsoComposeDownscale(1, "desktop") === 1, "desktop 1× stays");
      },
    },
    {
      name: "mobile iso-step caps",
      fn: () => {
        assert(coarseIsoSteps(32, "mobile") === 16, "occupancy 16 steps");
        assert(coarseIsoSteps(32, "desktop") === 32, "desktop occupancy keeps steps");
        assert(clampIsoStepsForTier(44, "mobile") === 32, "boot q=25 steps cap at 32");
        assert(clampIsoStepsForTier(64, "desktop") === 64, "desktop keeps 64");
      },
    },
  ]);
}
