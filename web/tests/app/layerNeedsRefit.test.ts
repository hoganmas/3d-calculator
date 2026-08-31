import { layerNeedsRefit } from "../../src/app/layerFitPolicy.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("app / layer-needs-refit", [
    {
      name: "static refit only when content dirty",
      fn: () => {
        assert(!layerNeedsRefit(false, false, false), "clean static layer skips");
        assert(layerNeedsRefit(false, true, false), "edited static layer refits");
      },
    },
    {
      name: "anim refit when param depends",
      fn: () => {
        assert(layerNeedsRefit(true, false, true), "param-driven layer anim refits");
        assert(!layerNeedsRefit(true, false, false), "unrelated layer skips anim pass");
      },
    },
    {
      name: "anim refit when latex changed even without param deps",
      fn: () => {
        assert(layerNeedsRefit(true, true, false), "edited unrelated layer refits during anim");
      },
    },
  ]);
}
