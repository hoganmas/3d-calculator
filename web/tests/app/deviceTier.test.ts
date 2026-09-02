import "../helpers/setup-dom.ts";
import {
  bootQualityForTier,
  detectDeviceTier,
  webGpuPowerPreference,
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
  ]);
}
