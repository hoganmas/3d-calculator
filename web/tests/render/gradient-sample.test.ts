import {
  flowSpeedColorEndpoints,
  sampleGradStops,
  sampleGradStopsFractBug,
} from "../helpers/gradientSample.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

const ORANGE: [number, number, number] = [1, 0.27, 0];
const BLUE: [number, number, number] = [0, 0.4, 1];

export async function run() {
  return runSuite("render / gradient-sample", [
    {
      name: "sampleGradStops at t=1 returns last stop (not fract bug)",
      fn: () => {
        const fixed = sampleGradStops([ORANGE, BLUE], 1);
        const buggy = sampleGradStopsFractBug([ORANGE, BLUE], 1);
        assertNear(fixed[0]!, BLUE[0]!, 1e-9, "fixed endpoint r");
        assertNear(buggy[0]!, ORANGE[0]!, 1e-9, "buggy endpoint r");
        assert(fixed[0] !== buggy[0], "t=1 endpoints differ");
      },
    },
    {
      name: "sampleGradStops at t=0 returns first stop",
      fn: () => {
        const rgb = sampleGradStops([ORANGE, BLUE], 0);
        assertNear(rgb[0]!, ORANGE[0]!, 1e-9, "start r");
      },
    },
    {
      name: "flowSpeedColor uses distinct col1/col2 at max speed",
      fn: () => {
        const fixed = flowSpeedColorEndpoints([ORANGE, BLUE], 1, sampleGradStops);
        const buggy = flowSpeedColorEndpoints([ORANGE, BLUE], 1, sampleGradStopsFractBug);
        assertNear(fixed[0]!, BLUE[0]!, 1e-9, "fast particles get col2");
        assertNear(buggy[0]!, ORANGE[0]!, 1e-9, "bug leaves fast particles at col1");
      },
    },
    {
      name: "multi-stop sample at interior t",
      fn: () => {
        const mid = sampleGradStops(
          [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [1, 1, 1],
          ],
          0.5,
        );
        assertNear(mid[0]!, 0.5, 1e-9, "mid stop");
      },
    },
  ]);
}
