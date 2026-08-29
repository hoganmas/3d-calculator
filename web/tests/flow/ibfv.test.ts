import {
  compileVectorExpr,
  fitVectorField,
  flowPresenceSlice,
  flowSpeedSlice,
  FLOW_DYE_AGE,
  FLOW_DYE_CHANNELS,
  FLOW_DYE_TOTAL,
  ibfvAdvectStep,
  ibfvBackgroundGridlines,
  ibfvBackgroundGridPoints,
  ibfvClampVelocity,
  seedFlowDyeGridlines,
} from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function dyeBuf(M: number): Float32Array {
  return new Float32Array(M * M * M * FLOW_DYE_CHANNELS);
}

function setDye(buf: Float32Array, vi: number, amount: number, age = 0): void {
  const o = vi * FLOW_DYE_CHANNELS;
  buf[o + FLOW_DYE_TOTAL] = amount;
  buf[o + FLOW_DYE_AGE] = age;
}

function maxDye(buf: Float32Array): number {
  let max = 0;
  for (let i = FLOW_DYE_TOTAL; i < buf.length; i += FLOW_DYE_CHANNELS) {
    max = Math.max(max, buf[i]!);
  }
  return max;
}

function dyeAt(buf: Float32Array, vi: number, ch: number): number {
  return buf[vi * FLOW_DYE_CHANNELS + ch]!;
}

export async function run() {
  return runSuite("flow / ibfv", [
    {
      name: "flowSpeedSlice matches vector magnitude",
      fn: () => {
        const fx = new Float32Array([3, 0]);
        const fy = new Float32Array([4, 0]);
        const fz = new Float32Array([0, 0]);
        const speed = flowSpeedSlice(fx, fy, fz, 2);
        assert(Math.abs(speed[0]! - 5) < 1e-6, "speed[0]");
        assert(Math.abs(speed[1]! - 0) < 1e-6, "speed[1]");
      },
    },
    {
      name: "flowPresenceSlice masks stagnation only",
      fn: () => {
        const fx = new Float32Array([1, 0]);
        const fy = new Float32Array([0, 0]);
        const fz = new Float32Array([0, 0]);
        const presence = flowPresenceSlice(fx, fy, fz, 2);
        assert(presence[0] === 1, "moving cell");
        assert(presence[1] === 0, "stagnation");
      },
    },
    {
      name: "grid points peak at lattice nodes only",
      fn: () => {
        const s = 0.5;
        assert(ibfvBackgroundGridPoints(0, 0, 0, s) > 0.9, "origin on lattice");
        assert(ibfvBackgroundGridPoints(0.25, 0, 0, s) < 0.1, "edge mid not a point");
        assert(
          ibfvBackgroundGridPoints(0, 0.25, 0, s) < ibfvBackgroundGridlines(0, 0.25, 0, s),
          "axis line point dimmer than full line",
        );
      },
    },
    {
      name: "gridlines stay in 0..1 and peak on axis planes",
      fn: () => {
        const s = 0.5;
        assert(ibfvBackgroundGridlines(0, 0, 0, s) > 0.9, "origin on grid");
        assert(ibfvBackgroundGridlines(0.25, 0.25, 0.25, s) < 0.1, "cell center off grid");
        for (let i = 0; i < 16; i++) {
          const g = ibfvBackgroundGridlines(i * 0.13, i * 0.17, i * 0.11, s);
          assert(g >= 0 && g <= 1, `grid out of range: ${g}`);
        }
      },
    },
    {
      name: "injection at upstream footpoint adds grid structure",
      fn: () => {
        const M = 8;
        const half = 2;
        const dyeIn = dyeBuf(M);
        const dyeOut = dyeBuf(M);
        ibfvAdvectStep(
          dyeIn,
          dyeOut,
          () => [1, 0, 0],
          M,
          { alpha: 0.5, gridSpacing: 0.3, dt: 0.05, vMax: 5, frameIdx: 0, half },
        );
        assert(maxDye(dyeOut) > 0.2, `injection should seed gridlines: max=${maxDye(dyeOut)}`);
      },
    },
    {
      name: "injection keeps age near zero",
      fn: () => {
        const M = 8;
        const half = 2;
        const dyeIn = dyeBuf(M);
        const dyeOut = dyeBuf(M);
        ibfvAdvectStep(
          dyeIn,
          dyeOut,
          () => [1, 0, 0],
          M,
          { alpha: 0.8, gridSpacing: 0.3, dt: 0.05, vMax: 5, frameIdx: 0, half },
        );
        let maxAge = 0;
        for (let vi = 0; vi < M * M * M; vi++) {
          if (dyeAt(dyeOut, vi, FLOW_DYE_TOTAL) > 0.1) {
            maxAge = Math.max(maxAge, dyeAt(dyeOut, vi, FLOW_DYE_AGE));
          }
        }
        assert(maxAge < 0.02, `fresh injection age should be ~0: maxAge=${maxAge}`);
      },
    },
    {
      name: "pure advection increments age",
      fn: () => {
        const M = 8;
        const half = 2;
        const dt = 0.1;
        const dyeIn = dyeBuf(M);
        const dyeOut = dyeBuf(M);
        const mid = Math.floor(M / 2);
        setDye(dyeIn, mid + mid * M + mid * M * M, 1, 0);
        ibfvAdvectStep(
          dyeIn,
          dyeOut,
          () => [1, 0, 0],
          M,
          { alpha: 0, gridSpacing: 0.3, dt, vMax: 5, frameIdx: 0, half },
        );
        const downstream = mid + 1 + mid * M + mid * M * M;
        assert(Math.abs(dyeAt(dyeOut, downstream, FLOW_DYE_AGE) - dt) < 0.02, "age should advance by dt");
      },
    },
    {
      name: "stagnation cells receive no injection",
      fn: () => {
        const M = 4;
        const half = 2;
        const dyeIn = dyeBuf(M);
        const dyeOut = dyeBuf(M);
        ibfvAdvectStep(
          dyeIn,
          dyeOut,
          () => [0, 0, 0],
          M,
          { alpha: 0.5, gridSpacing: 0.3, dt: 0.05, vMax: 2, frameIdx: 0, half },
        );
        assert(maxDye(dyeOut) === 0, "stagnation should stay zero");
      },
    },
    {
      name: "vMax clamp limits step magnitude",
      fn: () => {
        const [vx, vy, vz] = ibfvClampVelocity(10, 0, 0, 2);
        assert(Math.abs(Math.hypot(vx, vy, vz) - 2) < 1e-6, "clamped to vMax");
      },
    },
    {
      name: "uniform +x flow advects dye downstream",
      fn: () => {
        const M = 8;
        const half = 2;
        const dyeIn = dyeBuf(M);
        const dyeOut = dyeBuf(M);
        const seedIx = 2;
        const mid = Math.floor(M / 2);
        setDye(dyeIn, seedIx + mid * M + mid * M * M, 1);

        ibfvAdvectStep(
          dyeIn,
          dyeOut,
          () => [1, 0, 0],
          M,
          { alpha: 0, gridSpacing: 0.5, dt: 0.25, vMax: 5, frameIdx: 0, half },
        );

        const downstream = dyeAt(dyeOut, (seedIx + 1) + mid * M + mid * M * M, FLOW_DYE_TOTAL);
        assert(downstream > 0.1, `dye should appear downstream: ${downstream}`);
      },
    },
    {
      name: "swirl field has nonzero speed away from origin",
      fn: () => {
        const compiled = compileVectorExpr("(-y, x, 0)");
        const fit = fitVectorField(compiled, compiled.bind({}), 2.5, 8);
        const speed = flowSpeedSlice(fit.fx, fit.fy, fit.fz, fit.M);
        let maxSpeed = 0;
        for (let i = 0; i < speed.length; i++) maxSpeed = Math.max(maxSpeed, speed[i]!);
        assert(maxSpeed > 0.5, `swirl max speed too low: ${maxSpeed}`);
      },
    },
    {
      name: "swirl presence is nonzero on every z slice",
      fn: () => {
        const compiled = compileVectorExpr("(-y, x, 0)");
        const fit = fitVectorField(compiled, compiled.bind({}), 2.5, 8);
        const M = fit.M;
        const presence = flowPresenceSlice(fit.fx, fit.fy, fit.fz, M);
        for (let iz = 0; iz < M; iz++) {
          let any = false;
          for (let iy = 0; iy < M; iy++) {
            for (let ix = 0; ix < M; ix++) {
              if (presence[ix + iy * M + iz * M * M]! > 0.5) any = true;
            }
          }
          assert(any, `z slice ${iz} should have moving cells`);
        }
      },
    },
    {
      name: "+z flow advects dye along z",
      fn: () => {
        const M = 8;
        const half = 2;
        const dyeIn = dyeBuf(M);
        const dyeOut = dyeBuf(M);
        const mid = Math.floor(M / 2);
        setDye(dyeIn, mid + mid * M + 0, 1);

        ibfvAdvectStep(
          dyeIn,
          dyeOut,
          () => [0, 0, 1],
          M,
          { alpha: 0, gridSpacing: 0.5, dt: 0.25, vMax: 5, frameIdx: 0, half },
        );

        const upstream = dyeAt(dyeOut, mid + mid * M + 1 * M * M, FLOW_DYE_TOTAL);
        assert(upstream > 0.1, `dye should advect along +z: ${upstream}`);
      },
    },
    {
      name: "swirl advection deforms seeded gridlines",
      fn: () => {
        const compiled = compileVectorExpr("(-y, x, 0)");
        const field = compiled.bind({});
        const M = 8;
        const half = 2;
        const n = M * M * M;
        const bufA = seedFlowDyeGridlines(M, half, 0.3, 1);
        const bufB = dyeBuf(M);
        let read = bufA;
        let write = bufB;
        const params = { alpha: 0, gridSpacing: 0.3, dt: 0.05, vMax: 6, frameIdx: 0, half };
        for (let f = 0; f < 40; f++) {
          ibfvAdvectStep(read, write, field, M, { ...params, frameIdx: f });
          const tmp = read;
          read = write;
          write = tmp;
        }
        let min = Infinity;
        let max = -Infinity;
        for (let i = 0; i < n; i++) {
          const a = dyeAt(read, i, FLOW_DYE_TOTAL);
          min = Math.min(min, a);
          max = Math.max(max, a);
        }
        assert(max - min > 0.015, `swirl IBFV should vary: min=${min} max=${max}`);
      },
    },
  ]);
}
