import {
  advectFlowParticles,
  buildFlowParticleDensityGrid,
  FLOW_PARTICLE_DENSITY_GRID,
  FLOW_PARTICLE_STRIDE,
  flowParticleSpeedMinMax,
  flowSpeedMinMax,
  flowSpeedPercentileMinMax,
  MAX_FLOW_TRAIL_STEPS,
  pickLowDensitySpawn,
  pushFlowTrailHist,
  redistributeOvercrowdedFlowParticles,
  resolveFlowParticleColorRange,
  sampleVelGridAt,
  seedFlowParticles,
  seedFlowTrailHist,
  sortFlowParticlesByDepth,
} from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("flow / particles", [
    {
      name: "seedFlowParticles places particles on grid mask",
      fn: () => {
        const { posAge, layerIds } = seedFlowParticles(64, 1, 1, 0.4, false);
        let onGrid = 0;
        for (let i = 0; i < 64; i++) {
          const o = i * FLOW_PARTICLE_STRIDE;
          const x = posAge[o]!;
          const y = posAge[o + 1]!;
          const z = posAge[o + 2]!;
          const nearAxis = Math.min(Math.abs(x), Math.abs(y), Math.abs(z)) < 0.08;
          if (nearAxis) onGrid++;
          assert(layerIds[i] === 0, "single layer id");
          assert(posAge[o + 3] === 0, "fresh age");
        }
        assert(onGrid >= 8, "most seeds lie on grid lines");
      },
    },
    {
      name: "sampleVelGridAt trilinear on uniform field",
      fn: () => {
        const M = 4;
        const n = M * M * M;
        const fx = new Float32Array(n).fill(1);
        const fy = new Float32Array(n).fill(0);
        const fz = new Float32Array(n).fill(0);
        const [vx] = sampleVelGridAt(fx, fy, fz, M, 1, 0.2, 0, 0);
        assert(Math.abs(vx - 1) < 0.15, "uniform vx");
      },
    },
    {
      name: "advectFlowParticles moves along +X velocity",
      fn: () => {
        const M = 4;
        const n = M * M * M;
        const fx = new Float32Array(n).fill(2);
        const fy = new Float32Array(n).fill(0);
        const fz = new Float32Array(n).fill(0);
        const layers = [{ fx, fy, fz }];
        const { posAge, layerIds } = seedFlowParticles(8, 1, 1, 0.5, false);
        posAge[0] = 0;
        posAge[1] = 0;
        posAge[2] = 0;
        posAge[3] = 0;
        posAge[4] = 0;
        advectFlowParticles(posAge, layerIds, layers, M, {
          dt: 0.1,
          vMax: 10,
          half: 1,
          alpha: 0,
          gridSpacing: 0.5,
          gridPoints: false,
          ageMax: 2,
          frameIdx: 0,
        });
        assert(posAge[0]! > 0.15, "particle advected +X");
        assert(posAge[3]! > 0.09, "age incremented");
        assert(posAge[4]! > 1.9, "speed stored");
      },
    },
    {
      name: "sortFlowParticlesByDepth orders back-to-front",
      fn: () => {
        const posAge = new Float32Array([
          0, 0, 0, 0, 0,
          0, 0, 2, 0, 0,
        ]);
        const order = new Uint32Array(2);
        sortFlowParticlesByDepth(posAge, order, [0, 0, -5], [0, 0, 1]);
        assert(order[0] === 1, "far particle first");
        assert(order[1] === 0, "near particle second");
      },
    },
    {
      name: "expired particle respawn collapses trail history",
      fn: () => {
        const M = 4;
        const n = M * M * M;
        const fx = new Float32Array(n).fill(0);
        const fy = new Float32Array(n).fill(0);
        const fz = new Float32Array(n).fill(0);
        const layers = [{ fx, fy, fz }];
        const posAge = new Float32Array(5);
        posAge[0] = 0;
        posAge[1] = 0;
        posAge[2] = 0;
        posAge[3] = 5;
        const layerIds = new Uint32Array([0]);
        const trailSteps = 3;
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * FLOW_PARTICLE_STRIDE);
        trailHist[5] = 9;
        trailHist[6] = 9;
        trailHist[7] = 9;
        advectFlowParticles(posAge, layerIds, layers, M, {
          dt: 0.1,
          vMax: 10,
          half: 1,
          alpha: 0,
          gridSpacing: 0.5,
          gridPoints: false,
          ageMax: 1,
          frameIdx: 0,
        }, trailHist, trailSteps);
        assert(Math.abs(trailHist[5]! - trailHist[0]!) < 1e-5, "slot 1 matches respawn x");
        assert(Math.abs(trailHist[10]! - trailHist[0]!) < 1e-5, "slot 2 matches respawn x");
      },
    },
    {
      name: "flowParticleSpeedMinMax uses live particle spread",
      fn: () => {
        const posAge = new Float32Array([
          0, 0, 0, 0, 1,
          0, 0, 0, 0, 3,
        ]);
        const range = flowParticleSpeedMinMax(posAge, null, 2, 2);
        assert(range !== null, "range found");
        assert(Math.abs(range![0]! - 1) < 1e-6, "particle min");
        assert(Math.abs(range![1]! - 3) < 1e-6, "particle max");
      },
    },
    {
      name: "flowSpeedPercentileMinMax ignores extreme outliers",
      fn: () => {
        const fx = new Float32Array([1, 2, 3, 4, 100]);
        const fy = new Float32Array(5);
        const fz = new Float32Array(5);
        const [lo, hi] = flowSpeedPercentileMinMax(fx, fy, fz, 0.1, 0.75);
        assert(hi <= 4.01, "high percentile below outlier");
        assert(lo <= 2.01, "low percentile near slow speeds");
      },
    },
    {
      name: "flowSpeedMinMax spans non-stagnation speeds",
      fn: () => {
        const fx = new Float32Array([0, 2, 4]);
        const fy = new Float32Array(3);
        const fz = new Float32Array(3);
        const [lo, hi] = flowSpeedMinMax(fx, fy, fz);
        assert(Math.abs(lo - 2) < 1e-6, "min skips stagnation");
        assert(Math.abs(hi - 4) < 1e-6, "max speed");
      },
    },
    {
      name: "pushFlowTrailHist shifts history and inserts newest at slot 0",
      fn: () => {
        const posAge = new Float32Array([1, 2, 3, 0.5, 2]);
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * FLOW_PARTICLE_STRIDE);
        trailHist.set([9, 9, 9, 0, 1, 8, 8, 8, 0, 0.5]);
        pushFlowTrailHist(posAge, trailHist, 2, 1);
        assert(Math.abs(trailHist[0]! - 1) < 1e-6, "new x");
        assert(Math.abs(trailHist[5]! - 9) < 1e-6, "prev x shifted");
        assert(Math.abs(trailHist[9]! - 1) < 1e-6, "prev speed shifted");
      },
    },
    {
      name: "flowParticleSpeedMinMax handles uniform speeds",
      fn: () => {
        const posAge = new Float32Array([
          0, 0, 0, 0, 2,
          0, 0, 0, 0, 2,
        ]);
        const range = flowParticleSpeedMinMax(posAge, null, 2, 2);
        assert(range !== null, "uniform speeds still yield a range");
        assert(range![0]! < 2 && range![1]! >= 2, "range spans the shared speed");
      },
    },
    {
      name: "resolveFlowParticleColorRange uses vRef when trail speeds are zero",
      fn: () => {
        const posAge = new Float32Array([0, 0, 0, 0, 3]);
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * 5);
        const [lo, hi] = resolveFlowParticleColorRange(trailHist, 1, 2, 2);
        assert(Math.abs(lo) < 1e-6, "flowVMin stays at zero");
        assert(Math.abs(hi - 2) < 1e-6, "flowVMax falls back to vRef not head speed");
        void posAge;
      },
    },
    {
      name: "resolveFlowParticleColorRange prefers live trail max over vRef",
      fn: () => {
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * 5);
        trailHist[4] = 4;
        const [lo, hi] = resolveFlowParticleColorRange(trailHist, 1, 2, 2);
        assert(Math.abs(hi - 4) < 1e-6, "flowVMax uses trail max");
        assert(lo < hi, "flowVMin below flowVMax when spread widened");
      },
    },
    {
      name: "resolveFlowParticleColorRange ignores advection vRef when trails have speed",
      fn: () => {
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * 5);
        trailHist[4] = 3.3;
        const [lo, hi] = resolveFlowParticleColorRange(trailHist, 1, 2, 60);
        assert(Math.abs(hi - 3.3) < 1e-6, "flowVMax uses trail max not vMax clamp");
        assert(lo < hi, "min-max span for shader");
      },
    },
    {
      name: "resolveFlowParticleColorRange uses field range when trail spread is narrow",
      fn: () => {
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * 5);
        trailHist[4] = 2;
        trailHist[9] = 2.05;
        const [lo, hi] = resolveFlowParticleColorRange(trailHist, 1, 2, 60, [0.5, 4]);
        assert(Math.abs(lo - 0.5) < 1e-6, "field lo");
        assert(Math.abs(hi - 4) < 1e-6, "field hi");
      },
    },
    {
      name: "pickLowDensitySpawn prefers sparse cells",
      fn: () => {
        const res = FLOW_PARTICLE_DENSITY_GRID;
        const density = new Uint16Array(res * res * res);
        const crowded = res * res + 1;
        density[crowded] = 50;
        const pt = pickLowDensitySpawn(density, res, 1, 3, 0, 0, 0.5, false);
        assert(pt !== null, "spawn found");
      },
    },
    {
      name: "redistributeOvercrowdedFlowParticles thins dense cells",
      fn: () => {
        const count = 8;
        const posAge = new Float32Array(count * FLOW_PARTICLE_STRIDE);
        const layerIds = new Uint32Array(count);
        for (let i = 0; i < count; i++) {
          posAge[i * FLOW_PARTICLE_STRIDE] = 0;
          layerIds[i] = 0;
        }
        const before = buildFlowParticleDensityGrid(posAge, count, 1);
        const peakBefore = Math.max(...before);
        redistributeOvercrowdedFlowParticles(
          posAge, layerIds, count, 1, 0.5, false, 0, null, 0,
        );
        const after = buildFlowParticleDensityGrid(posAge, count, 1);
        const peakAfter = Math.max(...after);
        assert(peakAfter <= peakBefore, "peak density did not increase");
      },
    },
  ]);
}
