import {
  PERF_ADAPT_FPS_THRESHOLD,
  PERF_ADAPT_MAX_STEPDOWNS,
  PERF_ADAPT_STREAK_WINDOWS,
  nextPerfAdaptStreak,
  perfAdaptBlockedByUserOverride,
  shouldTriggerPerfStepDown,
  stepDownQualityValues,
} from "../../src/app/perfAdaptLogic.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("app / perf-adapt", [
    {
      name: "nextPerfAdaptStreak accumulates on low FPS",
      fn: () => {
        assert(nextPerfAdaptStreak(0, 15, 16) === 1, "streak +1");
        assert(nextPerfAdaptStreak(2, 15, 16) === 3, "streak reaches trigger");
        assert(nextPerfAdaptStreak(1, 60, 16) === 0, "good fps resets");
      },
    },
    {
      name: "shouldTriggerPerfStepDown at streak window",
      fn: () => {
        assert(!shouldTriggerPerfStepDown(PERF_ADAPT_STREAK_WINDOWS - 1), "below threshold");
        assert(shouldTriggerPerfStepDown(PERF_ADAPT_STREAK_WINDOWS), "at threshold");
      },
    },
    {
      name: "stepDownQualityValues decrements all sliders",
      fn: () => {
        const next = stepDownQualityValues({
          precisionQuality: 50,
          scalarQuality: 50,
          surfaceQuality: 50,
          vectorQuality: 50,
        });
        assert(next.scalarQuality === 40, "scalar -10");
        assert(next.vectorQuality === 40, "vector -10");
      },
    },
    {
      name: "perfAdaptBlockedByUserOverride honors cooldown",
      fn: () => {
        const now = 100_000;
        assert(perfAdaptBlockedByUserOverride(now, now - 1000), "recent override blocks");
        assert(!perfAdaptBlockedByUserOverride(now, now - 120_000), "old override allows");
      },
    },
    {
      name: "threshold constants are sane",
      fn: () => {
        assert(PERF_ADAPT_FPS_THRESHOLD >= 20 && PERF_ADAPT_FPS_THRESHOLD <= 30, "fps threshold");
        assert(PERF_ADAPT_STREAK_WINDOWS >= 2, "streak windows");
        assert(PERF_ADAPT_MAX_STEPDOWNS >= 1, "max step-downs");
      },
    },
  ]);
}
