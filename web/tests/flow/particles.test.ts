import {
  advectFlowParticles,
  FLOW_PARTICLE_STRIDE,
  MAX_FLOW_TRAIL_STEPS,
  pushFlowTrailHist,
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
  ]);
}
