import {
  layerNeedsRefit,
  animTickNeedsCpuFit,
  bakeLatexDrift,
} from "../../src/app/layerFitPolicy.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("app / layer-needs-refit", [
    {
      name: "static refit when content dirty or params depend",
      fn: () => {
        assert(!layerNeedsRefit(false, false, false), "clean static layer skips");
        assert(layerNeedsRefit(false, true, false), "edited static layer refits");
        assert(layerNeedsRefit(false, false, true), "static param-dependent layer refits");
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
    {
      name: "1-D GPU blend skip CPU fit",
      fn: () => {
        const layers = [
          { id: "iso-a", latex: "f=t", freeParams: ["t"], keyframes: { length: 8 } },
          { id: "iso-b", latex: "g=t", freeParams: ["t"], keyframes: { length: 8 } },
        ];
        const ready = (id: string) => id === "iso-a" || id === "iso-b";
        assert(
          !animTickNeedsCpuFit(layers, new Set(["t"]), ready),
          "two 1-D GPU isos skip compile",
        );
      },
    },
    {
      name: "CPU fit when GPU blend missing or 2-D dirty",
      fn: () => {
        const layer = {
          id: "iso-a",
          latex: "f=t",
          freeParams: ["t", "s"],
          keyframes: { length: 8 },
        };
        assert(
          animTickNeedsCpuFit([layer], new Set(["t", "s"]), () => true),
          "two dirty sliders need CPU",
        );
        assert(
          animTickNeedsCpuFit([layer], new Set(["t"]), () => false),
          "missing GPU blend needs CPU",
        );
      },
    },
    {
      name: "unstamped bake still skips when GPU keyframes are ready",
      fn: () => {
        const layers = [{ id: "iso-a", keyframes: { length: 8 } }];
        assert(
          !animTickNeedsCpuFit(layers, new Set(["t"]), () => true),
          "K>1 without freeParams still GPU-blend",
        );
        assert(
          animTickNeedsCpuFit(layers, new Set(["t"]), () => false),
          "unstamped bake without GPU blend forces CPU",
        );
      },
    },
    {
      name: "flow / independent dirty layer still needs CPU",
      fn: () => {
        const layers = [
          { id: "iso-a", freeParams: ["t"], keyframes: { length: 8 } },
          { id: "flow-a", freeParams: ["t"], keyframes: { length: 0 } },
        ];
        assert(
          animTickNeedsCpuFit(layers, new Set(["t"]), (id) => id === "iso-a"),
          "flow depending on t still CPU-fits",
        );
      },
    },
    {
      name: "bake latex drift vs live field rows",
      fn: () => {
        const layers = [{ id: "iso-a", latex: "x^2+y^2+z^2-t" }];
        assert(
          !bakeLatexDrift(layers, new Map([["iso-a", "x^2+y^2+z^2-t"], ["p1", "t=0"]])),
          "param rows are not bake membership",
        );
        assert(
          bakeLatexDrift(layers, new Map([["iso-a", "x^2+y^2+z^2-s"]])),
          "latex change is drift",
        );
        assert(bakeLatexDrift(layers, new Map()), "missing live field is drift");
        assert(
          !bakeLatexDrift([{ id: "iso-a" }], new Map([["iso-a", "x^2"]])),
          "unstamped latex is not drift",
        );
      },
    },
  ]);
}
