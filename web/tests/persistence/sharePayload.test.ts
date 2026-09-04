import {
  decodeSharePayload,
  fragmentFromSharePayload,
  normalizeSharePayload,
  sharePanelsFromRows,
  validateSharePayload,
} from "../../../api/_lib/sharePayload.ts";
import { encodeExpressionsFragment } from "../../src/app/persistence/exprShare.ts";
import type { ExprItem } from "../../src/types/models.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function sampleExpr(overrides: Partial<ExprItem> = {}): ExprItem {
  return {
    id: "e1",
    latex: String.raw`\exp(-r^2)`,
    color: "#ff4500",
    color2: "#ffec00",
    colors: ["#ff4500", "#ffec00"],
    enabled: true,
    sliderMin: -10,
    sliderMax: 10,
    sliderSpeed: 0.35,
    sliderAnimating: false,
    sliderPhase: 0.2,
    sliderAnimMode: "pingpong",
    autoParam: false,
    ...overrides,
  };
}

export async function run() {
  return runSuite("persistence / sharePayload (api)", [
    {
      name: "sharePanelsFromRows keeps a plain graphed expression",
      fn: () => {
        const panels = sharePanelsFromRows([sampleExpr({ latex: "x^2+y^2+z^2=1" })]);
        assert(panels.length === 1, "one panel");
        assert(panels[0]!.latex === "x^2+y^2+z^2=1", "latex preserved");
      },
    },
    {
      // Regression: a regex that excluded any `name=…` latex used to also
      // exclude legitimate graphed surfaces/aliases like `z=…` or `T=…`,
      // silently producing empty OG images for that whole class of shares.
      name: "sharePanelsFromRows keeps single-letter-named graphed surfaces (z=..., T=...)",
      fn: () => {
        const panels = sharePanelsFromRows([
          sampleExpr({ id: "e1", latex: String.raw`z=-\cos\left(x\right)\sin\left(2y\right)` }),
          sampleExpr({ id: "e2", latex: "T=x^2+y^2" }),
        ]);
        assert(panels.length === 2, "both rows kept");
        assert(panels[0]!.latex.startsWith("z="), "z= row kept");
        assert(panels[1]!.latex.startsWith("T="), "T= row kept");
      },
    },
    {
      name: "sharePanelsFromRows excludes autoParam rows from becoming panels",
      fn: () => {
        const panels = sharePanelsFromRows([
          sampleExpr({ id: "e1", latex: "a=1", autoParam: true }),
          sampleExpr({ id: "e2", latex: "z=x+y" }),
        ]);
        assert(panels.length === 1, "only the graphed row is a panel");
        assert(panels[0]!.latex === "z=x+y", "graphed row kept");
      },
    },
    {
      name: "sharePanelsFromRows excludes rows with empty latex",
      fn: () => {
        const panels = sharePanelsFromRows([
          sampleExpr({ id: "e1", latex: "" }),
          sampleExpr({ id: "e2", latex: "   " }),
          sampleExpr({ id: "e3", latex: "z=x" }),
        ]);
        assert(panels.length === 1, "blank rows dropped");
      },
    },
    {
      name: "sharePanelsFromRows respects the max panel count",
      fn: () => {
        const rows = Array.from({ length: 5 }, (_, i) =>
          sampleExpr({ id: `e${i}`, latex: `z=x+${i}` }),
        );
        const panels = sharePanelsFromRows(rows, 3);
        assert(panels.length === 3, "truncated to max");
      },
    },
    {
      name: "normalizeSharePayload strips # and e= prefixes",
      fn: () => {
        assert(normalizeSharePayload("#e=abc") === "abc", "strips # and e=");
        assert(normalizeSharePayload("e=abc") === "abc", "strips e=");
        assert(normalizeSharePayload("abc") === "abc", "leaves bare payload");
      },
    },
    {
      name: "validateSharePayload rejects garbage and accepts a real encoded payload",
      fn: async () => {
        let threw = false;
        try {
          validateSharePayload("not-a-real-payload");
        } catch {
          threw = true;
        }
        assert(threw, "garbage payload rejected");

        const fragment = await encodeExpressionsFragment([sampleExpr({ latex: "z=x" })]);
        const payload = normalizeSharePayload(fragment);
        assert(validateSharePayload(payload) === payload, "valid payload accepted");
      },
    },
    {
      // The OG capture loads decodeSharePayload's full row set together in
      // one scene (renderShareOgPng) rather than isolating panels, so an
      // animated expression's parameter row just needs to survive the
      // encode/decode round-trip alongside its visual row — no separate
      // "carry the param to its panel" step to regress.
      name: "decodeSharePayload round-trips a param row alongside its dependent visual row",
      fn: async () => {
        const paramRow = sampleExpr({ id: "e1", latex: "t=0", autoParam: true });
        const visualRow = sampleExpr({ id: "e2", latex: String.raw`y=\sin\left(x+t\right)` });
        const fragment = await encodeExpressionsFragment([paramRow, visualRow]);
        const payload = normalizeSharePayload(fragment);

        const rows = await decodeSharePayload(payload);
        assert(rows.length === 2, "both rows decoded");
        assert(rows[0]?.latex === "t=0" && rows[0]?.autoParam === true, "param row round-trips");
        assert(rows[1]?.latex?.includes("\\sin"), "visual row round-trips");

        const panels = sharePanelsFromRows(rows);
        assert(panels.length === 1, "only the visual row becomes a panel");
        assert(panels[0]!.latex.includes("\\sin"), "panel latex is the visual row");
      },
    },
    {
      name: "fragmentFromSharePayload round-trips through decodeExpressionsFragment shape",
      fn: () => {
        assert(fragmentFromSharePayload("1.abc") === "e=1.abc", "wraps payload as e=…");
      },
    },
  ]);
}
