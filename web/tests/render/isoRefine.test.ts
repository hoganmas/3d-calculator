import {
  isoFineFramebufferSize,
  isoNeedRefineFromCoarse2x2,
  isoDilatedOccupancyNeedsRefine,
  isoRefineKindFromCoarse2x2,
  isoRefineEnabled,
  ISO_OCC_HIT,
  ISO_REFINE_EDGE,
  ISO_REFINE_INTERSECT,
} from "../../src/render/webgpu/isoRefine.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("render / iso-refine", [
    {
      name: "fine size is display / fineDownscale, never coarser than occupancy",
      fn: () => {
        const full = isoFineFramebufferSize(800, 600, 800, 600, 1);
        assert(full.fw === 800 && full.fh === 600, "1× fine stays display");
        assert(!isoRefineEnabled(800, 600, 800, 600, 1), "no refine at 1×");
        const half = isoFineFramebufferSize(400, 300, 800, 600, 1);
        assert(half.fw === 800 && half.fh === 600, "2× coarse + 1× fine composes at display");
        assert(isoRefineEnabled(400, 300, 800, 600, 1), "refine when fine > coarse");
        const q50 = isoFineFramebufferSize(50, 38, 800, 600, 4);
        assert(q50.fw === 200 && q50.fh === 150, "q=50 compose is 4×");
        assert(isoRefineEnabled(50, 38, 800, 600, 4), "16× occupancy refines into 4× compose");
        const q0 = isoFineFramebufferSize(50, 38, 800, 600, 16);
        assert(q0.fw === 50 && q0.fh === 38, "q=0 compose matches 16× occupancy");
        assert(!isoRefineEnabled(50, 38, 800, 600, 16), "no refine when fine equals coarse");
      },
    },
    {
      name: "2×2 occupancy edge test",
      fn: () => {
        assert(!isoNeedRefineFromCoarse2x2(1, 1, 1, 1), "all miss is interior empty");
        assert(!isoNeedRefineFromCoarse2x2(0.2, 0.2, 0.21, 0.19), "flat hit is interior");
        assert(isoNeedRefineFromCoarse2x2(0.2, 1, 1, 1), "mixed occupancy refines");
        assert(isoNeedRefineFromCoarse2x2(0.1, 0.1, 0.1, 0.5), "depth crease refines");
        assert(ISO_OCC_HIT > 0.9, "hit threshold near far plane");
      },
    },
    {
      name: "2×2 mixed iso layers refine as intersection",
      fn: () => {
        assert(
          !isoNeedRefineFromCoarse2x2(0.2, 0.2, 0.21, 0.19, 1, 1, 1, 1),
          "same layer stays coarse",
        );
        assert(
          isoNeedRefineFromCoarse2x2(0.2, 0.2, 0.21, 0.19, 1, 2, 1, 1),
          "two isos in a 2×2 remarch",
        );
        assert(
          isoRefineKindFromCoarse2x2(0.2, 0.2, 0.21, 0.19, 1, 2, 1, 1) === ISO_REFINE_INTERSECT,
          "mixed layers are intersection, not just an edge",
        );
        assert(
          isoRefineKindFromCoarse2x2(0.2, 1, 1, 1) === ISO_REFINE_EDGE,
          "silhouette stays edge",
        );
        assert(
          isoNeedRefineFromCoarse2x2(0.2, 0.2, 0.21, 0.19, 1, 1, 1, 1, 1, 0, 0, 0),
          "box-face hit remarchs even when occupancy is flat",
        );
        assert(!isoDilatedOccupancyNeedsRefine(16), "4×4 all hits stays coarse");
        assert(!isoDilatedOccupancyNeedsRefine(0), "4×4 all miss stays empty");
        assert(isoDilatedOccupancyNeedsRefine(15), "1-texel ring miss remarchs silhouette band");
      },
    },
  ]);
}
