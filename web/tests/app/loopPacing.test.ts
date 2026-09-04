import {
  GPU_PATH_LAVA_INTERVAL_MOBILE_MS,
  GPU_PATH_LAVA_INTERVAL_TABLET_MS,
  shouldPresentThreeJs,
  threeJsPresentIntervalMs,
} from "../../src/app/loopPacing.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("app / loop-pacing", [
    {
      name: "WebGL fallback presents every rAF",
      fn: () => {
        assert(threeJsPresentIntervalMs("mobile", false) === 0, "mobile fallback");
        assert(threeJsPresentIntervalMs("desktop", false) === 0, "desktop fallback");
        assert(shouldPresentThreeJs(16, 0, 0), "interval 0 always presents");
        assert(shouldPresentThreeJs(16, 15, 0), "interval 0 even 1ms later");
      },
    },
    {
      name: "GPU iso path presents lava every rAF on every tier",
      fn: () => {
        assert(GPU_PATH_LAVA_INTERVAL_MOBILE_MS === 0, "mobile constant unthrottled");
        assert(GPU_PATH_LAVA_INTERVAL_TABLET_MS === 0, "tablet constant unthrottled");
        assert(threeJsPresentIntervalMs("mobile", true) === 0, "mobile");
        assert(threeJsPresentIntervalMs("tablet", true) === 0, "tablet");
        assert(threeJsPresentIntervalMs("desktop", true) === 0, "desktop every rAF");
      },
    },
    {
      name: "throttled presents wait for the interval",
      fn: () => {
        assert(shouldPresentThreeJs(0, 0, 50), "first frame");
        assert(shouldPresentThreeJs(49, 0, 50), "first frame even if last is 0");
        assert(!shouldPresentThreeJs(40, 10, 50), "40ms after last is early");
        assert(shouldPresentThreeJs(60, 10, 50), "50ms after last presents");
      },
    },
  ]);
}
