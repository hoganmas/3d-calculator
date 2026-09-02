/**
 * Coarse iso → upsample uniforms → refine iso must not clobber drawParamBuf
 * while the coarse pass is still pending (NDC 0 would sit at the bottom-right
 * of the smaller coarse target).
 */
import { uniformWriteNeedsFlush } from "../../src/render/webgpu/gpuSubmit.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("render / gpu-submit", [
    {
      name: "rewrite after a different buffer still flushes",
      fn: () => {
        const written = new Set<string>(["drawParamBuf"]);
        assert(
          uniformWriteNeedsFlush(1, written, "drawParamBuf"),
          "same buffer with pending cmds flushes",
        );
        written.add("isoUpsampleParamBuf");
        assert(
          uniformWriteNeedsFlush(2, written, "drawParamBuf"),
          "coarse params still flush after upsample uniforms",
        );
        assert(
          !uniformWriteNeedsFlush(2, written, "fxaaParamBuf"),
          "first write of a new buffer does not flush",
        );
        assert(
          !uniformWriteNeedsFlush(2, written, "drawParamBufRefine"),
          "refine uniforms on a separate buffer do not flush coarse",
        );
        written.add("isoDrawParam1");
        assert(
          !uniformWriteNeedsFlush(3, written, "isoDrawParam0"),
          "a second iso layer on its own uniform buffer does not flush",
        );
        written.add("drawParamBufRefine");
        assert(
          uniformWriteNeedsFlush(3, written, "drawParamBufRefine"),
          "mid then fine refine rewrite of the same buffer flushes",
        );
        assert(
          !uniformWriteNeedsFlush(0, written, "drawParamBuf"),
          "no pending cmds: overwrite in place",
        );
      },
    },
  ]);
}
