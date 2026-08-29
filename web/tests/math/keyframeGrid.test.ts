import {
  bakeOrderND,
  hypercellBlend,
  KEYFRAME_GRID_K,
  linearIndex,
  MAX_KEYFRAME_GRID_CELLS,
  totalFrameCount,
} from "../../src/math/keyframeGrid.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("keyframeGrid", [
    {
      name: "totalFrameCount is K^N (512 when N=3 and K=8)",
      fn: () => {
        const K = KEYFRAME_GRID_K;
        if (totalFrameCount(K, 1) !== 8) throw new Error(`1D ${totalFrameCount(K, 1)}`);
        if (totalFrameCount(K, 2) !== 64) throw new Error(`2D ${totalFrameCount(K, 2)}`);
        if (totalFrameCount(K, 3) !== MAX_KEYFRAME_GRID_CELLS) {
          throw new Error(`3D ${totalFrameCount(K, 3)}`);
        }
      },
    },
    {
      name: "linearIndex round-trips coords",
      fn: () => {
        const K = 4;
        const nDims = 2;
        for (let y = 0; y < K; y++) {
          for (let x = 0; x < K; x++) {
            const idx = linearIndex([x, y], K);
            const hc = hypercellBlend([0, 0], [1, 1], K, [x / (K - 1), y / (K - 1)]);
            let wsum = 0;
            for (const c of hc.corners) wsum += c.weight;
            if (Math.abs(wsum - 1) > 1e-6) throw new Error(`weights sum ${wsum}`);
            if (idx < 0 || idx >= totalFrameCount(K, nDims)) throw new Error(`bad idx ${idx}`);
          }
        }
      },
    },
    {
      name: "bakeOrderND prioritizes corners",
      fn: () => {
        const order = bakeOrderND(3, 2, [0, 4]);
        if (order[0] !== 0 || order[1] !== 4) throw new Error(`corners first: ${order.slice(0, 4)}`);
        if (order.length !== 9) throw new Error(`length ${order.length}`);
      },
    },
  ]);
}
