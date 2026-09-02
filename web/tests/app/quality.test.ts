import {
  inferQualityFromSettings,
  qualityToDeg,
  qualityToIsoMarchDownscale,
  qualityToIsoRefineDownscale,
  qualityToIsoSteps,
  qualityToMarchDownscale,
  qualityToParticleCount,
  qualityToSteps,
  qualityToTrailSteps,
  qualityToVolumeSteps,
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
        assert(qualityToIsoMarchDownscale(50) === qualityToMarchDownscale(50), "iso uses same curve");
        assert(qualityToIsoRefineDownscale(50) === 1, "refine compose is display-sized");
        assert(qualityToIsoRefineDownscale(100) === 1, "max quality refine is full res");
      },
    },
    {
      name: "qualityToVolumeSteps snaps to multiples of 4",
      fn: () => {
        const steps = qualityToVolumeSteps(50);
        assert(steps % 4 === 0, "volume steps aligned to 4");
        assert(steps >= 8 && steps <= 96, "volume steps in range");
        assert(qualityToSteps(50) === steps, "legacy alias matches");
      },
    },
    {
      name: "qualityToIsoSteps is independent and can exceed volume max",
      fn: () => {
        const iso = qualityToIsoSteps(100);
        assert(iso % 4 === 0, "iso steps aligned to 4");
        assert(iso >= 16 && iso <= 192, "iso steps in Hermite range");
        assert(iso > qualityToVolumeSteps(100), "iso budget higher at max");
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
        assert(qualityToParticleCount(50) === 1000, "default Q matches HTML particle count");
        assert(qualityToTrailSteps(20) === 32, "mobile-boot Q keeps full trail length");
        assert(qualityToTrailSteps(50) === 32, "default Q matches HTML trail length");
        const inferred = inferQualityFromSettings({
          marchDownscale: 2,
          isoMarchDownscale: 2,
          deg: 32,
          steps: 16,
          isoSteps: 32,
          flowParticleCount: 1000,
          flowTrailSteps: 32,
        });
        assert(inferred.vectorQuality === 50, "HTML flow settings infer default vector quality");
      },
    },
  ]);
}
