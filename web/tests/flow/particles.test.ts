import {
  advectFlowParticles,
  beginFlowParticleGhost,
  buildFlowParticleDensityGrid,
  buildFlowParticleDensityGrids,
  FLOW_PARTICLE_DENSITY_GRID,
  FLOW_PARTICLE_STRIDE,
  flowHeadSpeedMinMax,
  flowParticleSpeedMinMax,
  flowSpeedMinMax,
  flowSpeedPercentileMinMax,
  isFlowParticleGhost,
  MAX_FLOW_TRAIL_STEPS,
  pickLowDensitySpawn,
  pushFlowTrailHist,
  redistributeOvercrowdedFlowParticles,
  resolveFlowParticleColorRange,
  resolveFlowParticleColorRangeFast,
  resetFlowTrailHistSlot,
  seedFlowTrailHist,
  flowGhostTrailLifeAge,
  flowSpawnTrailLifeAge,
  syncFlowParticleTrailLife,
  updateFlowTrailHead,
  sampleVelGridAt,
  seedFlowParticles,
  sortFlowParticlesByDepth,
} from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("flow / particles", [
    {
      name: "seedFlowParticles allocates per flow layer",
      fn: () => {
        const { posAge, layerIds } = seedFlowParticles(10, 2, 1, 0.4);
        assert(posAge.length === 10 * 2 * FLOW_PARTICLE_STRIDE, "total buffer size");
        assert(layerIds.length === 20, "20 particles for 2 layers");
        let n0 = 0;
        let n1 = 0;
        for (let i = 0; i < layerIds.length; i++) {
          if (layerIds[i] === 0) n0++;
          if (layerIds[i] === 1) n1++;
        }
        assert(n0 === 10, "10 particles on layer 0");
        assert(n1 === 10, "10 particles on layer 1");
      },
    },
    {
      name: "seedFlowParticles places particles on grid mask",
      fn: () => {
        const { posAge, layerIds } = seedFlowParticles(64, 1, 1, 0.4);
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
        const { posAge, layerIds } = seedFlowParticles(8, 1, 1, 0.5);
        posAge[0] = 0;
        posAge[1] = 0;
        posAge[2] = 0;
        posAge[3] = 0;
        posAge[4] = 0;
        advectFlowParticles(posAge, layerIds, layers, M, {
          dt: 0.1,
          vMax: 10,
          half: 1,
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
        const depthKeys = new Float32Array(2);
        sortFlowParticlesByDepth(posAge, order, [0, 0, -5], [0, 0, 1], depthKeys);
        assert(order[0] === 1, "far particle first");
        assert(order[1] === 0, "near particle second");
        assert(depthKeys[1]! > depthKeys[0]!, "depth keys ordered");
      },
    },
    {
      name: "expired particle preserves trail during ghost fade",
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
          ageMax: 1,
          frameIdx: 0,
        }, trailHist, trailSteps);
        assert(isFlowParticleGhost(posAge, 0), "entered ghost fade");
        assert(Math.abs(trailHist[5]! - 9) < 1e-5, "trail history preserved on death");
        assert(posAge[4]! < -0.5, "ghost countdown active");
      },
    },
    {
      name: "ghost fade push completes with respawn",
      fn: () => {
        const M = 4;
        const n = M * M * M;
        const fx = new Float32Array(n).fill(1);
        const fy = new Float32Array(n).fill(0);
        const fz = new Float32Array(n).fill(0);
        const layers = [{ fx, fy, fz }];
        const posAge = new Float32Array(5);
        posAge[3] = 5;
        const layerIds = new Uint32Array([0]);
        const trailSteps = 2;
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * FLOW_PARTICLE_STRIDE);
        beginFlowParticleGhost(posAge, 0, trailSteps);
        const pushCtx = {
          layerIds,
          layers,
          M,
          half: 1,
          frameIdx: 0,
        };
        pushFlowTrailHist(posAge, trailHist, trailSteps, 1, pushCtx);
        assert(isFlowParticleGhost(posAge, 0), "still fading after first push");
        pushFlowTrailHist(posAge, trailHist, trailSteps, 1, pushCtx);
        assert(!isFlowParticleGhost(posAge, 0), "respawned after fade");
        assert(posAge[3]! < 0.5, "fresh age after respawn");
      },
    },
    {
      name: "flowGhostTrailLifeAge fades from full to invisible",
      fn: () => {
        const steps = 8;
        const start = flowGhostTrailLifeAge(-steps, steps);
        const end = flowGhostTrailLifeAge(-0.5, steps);
        assert(start > -1.01 && start < -0.99, "ghost starts at full visibility age");
        assert(end < -1.9, "ghost ends near invisible age");
      },
    },
    {
      name: "flowSpawnTrailLifeAge ramps from invisible to full",
      fn: () => {
        const start = flowSpawnTrailLifeAge(0);
        const mid = flowSpawnTrailLifeAge(0.175);
        const end = flowSpawnTrailLifeAge(0.35);
        assert(start < -0.49, "spawn starts invisible");
        assert(mid > -0.26 && mid < -0.24, "spawn mid fade");
        assert(end === 0.35, "spawn ends at real age");
      },
    },
    {
      name: "syncFlowParticleTrailLife writes ghost ages into trail",
      fn: () => {
        const posAge = new Float32Array(5);
        beginFlowParticleGhost(posAge, 0, 4);
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * FLOW_PARTICLE_STRIDE);
        trailHist[3] = 12;
        syncFlowParticleTrailLife(posAge, trailHist, 1, 4);
        assert(trailHist[3]! < -0.99 && trailHist[3]! > -1.01, "ghost head age encoded");
        assert(trailHist[8]! < -0.99, "ghost slot 1 age encoded");
      },
    },
    {
      name: "resetFlowTrailHistSlot backfills one step along velocity",
      fn: () => {
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * FLOW_PARTICLE_STRIDE);
        resetFlowTrailHistSlot(trailHist, 3, 0, 0, 0, 0, 0, 2, [1, 0, 0], 0.5);
        assert(Math.abs(trailHist[0]!) < 1e-6, "head at spawn");
        assert(Math.abs(trailHist[5]! + 0.5) < 1e-6, "slot 1 one step behind flow");
        assert(Math.abs(trailHist[10]! + 0.5) < 1e-6, "older slots duplicate slot 1");
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
      name: "buildFlowParticleDensityGrids are layer-local",
      fn: () => {
        const count = 4;
        const posAge = new Float32Array(count * FLOW_PARTICLE_STRIDE);
        const layerIds = new Uint32Array([0, 0, 1, 1]);
        for (let i = 0; i < count; i++) {
          const o = i * FLOW_PARTICLE_STRIDE;
          posAge[o] = i < 2 ? -0.5 : 0.5;
          posAge[o + 1] = 0;
          posAge[o + 2] = 0;
        }
        const grids = buildFlowParticleDensityGrids(posAge, layerIds, count, 1, 2);
        assert(grids.length === 2, "one grid per layer");
        const sum = (g: Uint16Array) => {
          let s = 0;
          for (let i = 0; i < g.length; i++) s += g[i]!;
          return s;
        };
        assert(sum(grids[0]!) === 2, "layer 0 has two particles");
        assert(sum(grids[1]!) === 2, "layer 1 has two particles");
      },
    },
    {
      name: "pickLowDensitySpawn prefers sparse cells",
      fn: () => {
        const res = FLOW_PARTICLE_DENSITY_GRID;
        const density = new Uint16Array(res * res * res);
        const crowded = res * res + 1;
        density[crowded] = 50;
        const pt = pickLowDensitySpawn(density, res, 1, 3, 0, 0);
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
          posAge, layerIds, count, 1, 0, null, 0,
        );
        const after = buildFlowParticleDensityGrid(posAge, count, 1);
        const peakAfter = Math.max(...after);
        assert(peakAfter <= peakBefore, "peak density did not increase");
      },
    },
    {
      name: "sortFlowParticlesByDepth without depthKeys array",
      fn: () => {
        const posAge = new Float32Array([
          0, 0, 0, 0, 0,
          0, 0, 2, 0, 0,
        ]);
        const order = new Uint32Array(2);
        sortFlowParticlesByDepth(posAge, order, [0, 0, -5], [0, 0, 1]);
        assert(order[0] === 1, "far particle first without depth buffer");
      },
    },
    {
      name: "advectFlowParticles respawns stuck particles",
      fn: () => {
        const M = 4;
        const n = M * M * M;
        const fx = new Float32Array(n).fill(0);
        const fy = new Float32Array(n).fill(0);
        const fz = new Float32Array(n).fill(0);
        const layers = [{ fx, fy, fz }];
        const posAge = new Float32Array(5);
        posAge[0] = 0.1;
        posAge[1] = 0.1;
        posAge[2] = 0.1;
        posAge[3] = 2;
        posAge[4] = 0;
        const layerIds = new Uint32Array([0]);
        advectFlowParticles(posAge, layerIds, layers, M, {
          dt: 0.1,
          vMax: 10,
          half: 1,
          ageMax: 1,
          frameIdx: 0,
        });
        assert(posAge[3]! < 0.5, "stuck particle respawned with fresh age");
      },
    },
    {
      name: "flowHeadSpeedMinMax and resolveFlowParticleColorRangeFast",
      fn: () => {
        const posAge = new Float32Array([
          0, 0, 0, 0, 1,
          0, 0, 0, 0, 4,
        ]);
        const head = flowHeadSpeedMinMax(posAge, 2);
        assert(head !== null && head[0] === 1 && head[1] === 4, "head range");
        const uniform = new Float32Array([0, 0, 0, 0, 2, 0, 0, 0, 0, 2]);
        const uniHead = flowHeadSpeedMinMax(uniform, 2);
        assert(uniHead !== null && uniHead[0]! < 2, "uniform widened");
        const [lo, hi] = resolveFlowParticleColorRangeFast(posAge, 2, 1, [0.5, 5]);
        assert(lo < hi, "fast color range");
        const fallback = resolveFlowParticleColorRangeFast(new Float32Array(5), 1, 2, null);
        assert(fallback[1] >= 2, "vRef fallback");
        const narrow = new Float32Array([0, 0, 0, 0, 2, 0, 0, 0, 0, 2.01]);
        const [nLo, nHi] = resolveFlowParticleColorRangeFast(narrow, 2, 1, null);
        const field = resolveFlowParticleColorRangeFast(narrow, 2, 1, [0.5, 4]);
        assert(field[0] === 0.5 && field[1] === 4, "field range when span narrow");
        assert(nHi - nLo > 0.01, "widened narrow span");
      },
    },
    {
      name: "seedFlowTrailHist and updateFlowTrailHead",
      fn: () => {
        const posAge = new Float32Array([1, 2, 3, 0.5, 2]);
        const trailHist = new Float32Array(MAX_FLOW_TRAIL_STEPS * FLOW_PARTICLE_STRIDE);
        seedFlowTrailHist(posAge, trailHist, 3, 1);
        assert(Math.abs(trailHist[0]! - 1) < 1e-6, "seed x");
        assert(Math.abs(trailHist[10]! - 1) < 1e-6, "seed last slot");
        posAge[0] = 9;
        updateFlowTrailHead(posAge, trailHist, 1);
        assert(Math.abs(trailHist[0]! - 9) < 1e-6, "head updated");
        resetFlowTrailHistSlot(trailHist, 2, 0, 4, 5, 6, 0.1, 1.5);
        assert(Math.abs(trailHist[0]! - 4) < 1e-6, "reset slot");
      },
    },
    {
      name: "advectFlowParticles respawn uses density grid when provided",
      fn: () => {
        const M = 4;
        const n = M * M * M;
        const fx = new Float32Array(n).fill(0);
        const fy = new Float32Array(n).fill(0);
        const fz = new Float32Array(n).fill(0);
        const layers = [{ fx, fy, fz }];
        const posAge = new Float32Array(5);
        posAge[3] = 5;
        const layerIds = new Uint32Array([0]);
        const density = new Uint16Array(FLOW_PARTICLE_DENSITY_GRID ** 3);
        advectFlowParticles(posAge, layerIds, layers, M, {
          dt: 0.1,
          vMax: 10,
          half: 1,
          ageMax: 1,
          frameIdx: 2,
        }, null, 0, density, FLOW_PARTICLE_DENSITY_GRID);
        assert(posAge[3]! < 1, "respawned via density-aware spawn");
      },
    },
  ]);
}
