import {
  qualityToDeg,
  qualityToMarchDownscale,
  qualityToParticleCount,
  qualityToSteps,
  qualityToTrailSteps,
} from "../../src/app/qualityMapping.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("app / quality", [
    {
      name: "qualityToMarchDownscale spans 8× down to 1×",
      fn: () => {
        assert(qualityToMarchDownscale(0) === 8, "low quality downscales heavily");
        assert(qualityToMarchDownscale(100) === 1, "max quality full resolution");
      },
    },
    {
      name: "qualityToSteps snaps to multiples of 4",
      fn: () => {
        const steps = qualityToSteps(50);
        assert(steps % 4 === 0, "steps aligned to 4");
        assert(steps >= 8 && steps <= 96, "steps in range");
      },
    },
    {
      name: "qualityToDeg increases with quality",
      fn: () => {
        assert(qualityToDeg(0) < qualityToDeg(100), "deg rises with quality");
        assert(qualityToDeg(0) >= 8 && qualityToDeg(100) <= 128, "deg in valid range");
      },
    },
    {
      name: "vector quality maps to particles and trail steps",
      fn: () => {
        assert(qualityToParticleCount(0) < qualityToParticleCount(100), "more particles at high Q");
        assert(qualityToTrailSteps(0) < qualityToTrailSteps(100), "longer trails at high Q");
      },
    },
  ]);
}
