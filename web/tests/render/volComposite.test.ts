import {
  beerClipCascade,
  beerDeferredIsoDepth,
  beerHasMidPass,
  beerIsoClipDepth,
  isoFinerThanVolume,
  volumePixelIsoFootprint,
} from "../../src/render/webgpu/volComposite.ts";
import { isoNeedRefineFromCoarse2x2 } from "../../src/render/webgpu/isoRefine.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("render / vol-composite", [
    {
      name: "same-res volume pixel maps to one iso texel",
      fn: () => {
        const a = volumePixelIsoFootprint(10, 20, 100, 200, 100, 200);
        assert(a.x0 === 10 && a.x1 === 10 && a.y0 === 20 && a.y1 === 20, "1:1");
        assert(!isoFinerThanVolume(100, 200, 100, 200), "not finer");
      },
    },
    {
      name: "2× finer iso spans two texels per volume pixel",
      fn: () => {
        const a = volumePixelIsoFootprint(0, 0, 100, 100, 200, 200);
        assert(a.x0 === 0 && a.x1 === 1 && a.y0 === 0 && a.y1 === 1, "first pixel");
        const b = volumePixelIsoFootprint(3, 5, 100, 100, 200, 200);
        assert(b.x0 === 6 && b.x1 === 7 && b.y0 === 10 && b.y1 === 11, "scaled pixel");
        const phone = volumePixelIsoFootprint(0, 0, 197, 321, 393, 641);
        assert(phone.x1 >= 1 && phone.y1 >= 1, "393 vs 197 still spans two iso texels");
        assert(isoFinerThanVolume(197, 321, 393, 641), "phone 2× cloud vs 1× iso");
        assert(beerDeferredIsoDepth(197, 321, 393, 641), "beer defers occl when iso is finer");
        assert(!beerDeferredIsoDepth(393, 641, 393, 641), "same-res still clips in beer");
      },
    },
    {
      name: "iso occupancy edges are the tiles that remarch beer at compose res",
      fn: () => {
        assert(
          isoNeedRefineFromCoarse2x2(0.2, 1, 1, 1),
          "silhouette tile is a fine-beer tile",
        );
        assert(
          !isoNeedRefineFromCoarse2x2(1, 1, 1, 1),
          "empty tile keeps volume-res beer",
        );
        assert(
          !isoNeedRefineFromCoarse2x2(0.2, 0.2, 0.21, 0.19),
          "flat iso interior is not a fine-beer tile",
        );
        assert(beerIsoClipDepth(0.2, 0.25) === 0.2, "interior all-hit clips to nearest iso");
        assert(beerIsoClipDepth(0.2, 1) === 1, "mixed footprint defers clip");
        assert(beerIsoClipDepth(1, 1) === 1, "empty footprint does not clip");
      },
    },
    {
      name: "16-4-1 beer ladder remarchs coarse-mixed tiles at 4×",
      fn: () => {
        assert(beerClipCascade(true) === "coarse", "3-tier cheap beer clips 16× interiors");
        assert(beerHasMidPass(true), "3-tier remarchs coarse-mixed tiles at 4×");
        assert(beerClipCascade(false) === "compose", "2-tier still clips cheap beer to compose");
        assert(!beerHasMidPass(false), "2-tier has no 4× beer pass");
        assert(!isoFinerThanVolume(197, 321, 25, 40), "phone coarse is coarser than 2× beer → nearest");
        assert(!isoFinerThanVolume(98, 160, 98, 160), "4× beer is 1:1 with mid iso");
        assert(isoFinerThanVolume(197, 321, 393, 641), "compose 1× vs 2× beer mixes cascades if used as clip");
      },
    },
  ]);
}
