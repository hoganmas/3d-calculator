import {
  compileVectorExpr,
  fitVectorField,
  flowPhaseAt,
  flowPhaseOpacity,
  flowPresenceSlice,
  flowSoftBand,
  flowSpatialPhase,
  flowSpeedSlice,
} from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("flow / phase", [
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
      name: "soft band stays in 0..1",
      fn: () => {
        for (let i = 0; i < 64; i++) {
          const b = flowSoftBand(i * 0.37);
          assert(b >= 0 && b <= 1, `band out of range: ${b}`);
        }
      },
    },
    {
      name: "opacity follows soft sine band",
      fn: () => {
        const o = 0.15;
        const peak = flowPhaseOpacity(1, Math.PI * 0.5, o);
        const trough = flowPhaseOpacity(1, -Math.PI * 0.5, o);
        assert(peak > trough, "band should modulate opacity");
        assert(Math.abs(peak - o) < 1e-6, "peak near full opacity");
        assert(Math.abs(trough - o * 0.12) < 1e-5, "trough near 12% of base");
        assert(flowPhaseOpacity(0, 0, o) === 0, "stagnation");
      },
    },
    {
      name: "phase increases along uniform flow direction",
      fn: () => {
        const stripe = 4;
        const time = 3;
        const p0 = flowPhaseAt(1, 0, 0, 0, 0, 0, 0, stripe, time);
        const p1 = flowPhaseAt(1, 0, 0, 0.5, 0, 0, 0, stripe, time);
        assert(p1 > p0, "phase should increase along +x for V=(1,0,0)");
      },
    },
    {
      name: "phase decreases over time at fixed point",
      fn: () => {
        const stripe = 4;
        const time = 3;
        const p0 = flowPhaseAt(1, 0, 0, 1, 0, 0, 0, stripe, time);
        const p1 = flowPhaseAt(1, 0, 0, 1, 0, 0, 0.1, stripe, time);
        assert(p1 < p0, "phase should decrease over time");
      },
    },
    {
      name: "temporal rate is independent of |V|",
      fn: () => {
        const dt = 0.1;
        const time = 3;
        const dSlow = flowPhaseAt(1, 0, 0, 0, 0, 0, dt, 5, time) - flowPhaseAt(1, 0, 0, 0, 0, 0, 0, 5, time);
        const dFast = flowPhaseAt(10, 0, 0, 0, 0, 0, dt, 5, time) - flowPhaseAt(10, 0, 0, 0, 0, 0, 0, 5, time);
        assert(Math.abs(dSlow - dFast) < 1e-6, "|V| should not change animation rate");
        assert(Math.abs(dSlow + dt * time) < 1e-6, "time scale sets temporal frequency");
      },
    },
    {
      name: "fixed base ignores vector magnitude",
      fn: () => {
        const o = 0.15;
        const phi = 0;
        assert(flowPhaseOpacity(2, phi, o) === flowPhaseOpacity(100, phi, o), "speed independent");
      },
    },
    {
      name: "swirl has azimuthal spatial phase",
      fn: () => {
        const half = 2.5;
        const s1 = flowSpatialPhase(0, 1, 0, 1, 0, 0, half);
        const s2 = flowSpatialPhase(-1, 0, 0, 0, 1, 0, half);
        assert(Math.abs(s1) < 1e-6, "along-flow term zero on x-axis");
        assert(Math.abs(s2) > 0.5, "azimuth varies around z");
        assert(Math.abs(s1 - s2) > 0.5, "different angles → different phase");
      },
    },
    {
      name: "xyz swirls differ at the same point",
      fn: () => {
        const half = 2.5;
        const px = 1; const py = 2; const pz = 1;
        const sZ = flowSpatialPhase(-py, px, 0, px, py, pz, half);
        const sX = flowSpatialPhase(0, -pz, py, px, py, pz, half);
        const sY = flowSpatialPhase(pz, 0, -px, px, py, pz, half);
        assert(Math.abs(sZ - sX) > 0.2, "z vs x swirl");
        assert(Math.abs(sZ - sY) > 0.2, "z vs y swirl");
        assert(Math.abs(sX - sY) > 0.2, "x vs y swirl");
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
  ]);
}
