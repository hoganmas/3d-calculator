import { gzipSync } from "node:zlib";
import {
  decodeExpressionsFragment,
  encodeExpressionsFragment,
  EXPR_SHARE_VERSION,
} from "../../src/app/persistence/exprShare.ts";
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
    role: "auto",
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

function bytesToBase64Url(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("base64url");
}

export async function run() {
  return runSuite("persistence / exprShare", [
    {
      name: "round-trips a simple expression (raw)",
      fn: async () => {
        const fragment = await encodeExpressionsFragment([sampleExpr()]);
        assert(fragment.startsWith(`e=${EXPR_SHARE_VERSION}.`), "uses raw encoding for tiny payload");
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded != null, "decoded");
        assert(decoded![0]?.latex === sampleExpr().latex, "latex");
        assert(decoded![0]?.role === undefined || decoded![0]?.role === "auto", "default role omitted");
      },
    },
    {
      name: "round-trips animated param row + field",
      fn: async () => {
        const exprs = [
          sampleExpr({
            id: "e1",
            latex: "t=0",
            autoParam: true,
            sliderAnimating: true,
            sliderMin: 0,
            sliderMax: 1,
            sliderSpeed: 0.12,
            sliderAnimMode: "loop",
            sliderPhase: 0,
          }),
          sampleExpr({
            id: "e2",
            latex: String.raw`y=\sin(x+2\pi t)\cos(z)`,
          }),
        ];
        const fragment = await encodeExpressionsFragment(exprs);
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.length === 2, "two rows");
        assert(decoded![0]?.autoParam === true, "autoParam");
        assert(decoded![0]?.sliderAnimating === true, "animating");
        assert(decoded![0]?.sliderAnimMode === "loop", "loop mode");
        assert(decoded![1]?.latex.includes("\\sin"), "field latex");
      },
    },
    {
      name: "uses gzip for larger payloads",
      fn: async () => {
        const exprs = Array.from({ length: 8 }, (_, i) =>
          sampleExpr({
            id: `e${i + 1}`,
            latex: String.raw`\exp(-${i}((x-a)^2+(y-b)^2+(z-${i})^2))`,
            role: i % 2 === 0 ? "cloud" : "isosurface",
          }),
        );
        const fragment = await encodeExpressionsFragment(exprs);
        assert(fragment.startsWith(`e=${EXPR_SHARE_VERSION}z.`), "uses gzip encoding");
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.length === 8, "row count");
        assert(decoded![1]?.role === "isosurface", "role preserved");
      },
    },
    {
      name: "decodes manually gzipped fragment",
      fn: async () => {
        const payload = [{ l: "x^2+y^2+z^2=1", r: "isosurface" as const }];
        const gz = gzipSync(JSON.stringify(payload));
        const fragment = `e=${EXPR_SHARE_VERSION}z.${bytesToBase64Url(gz)}`;
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.[0]?.latex === payload[0]!.l, "latex");
        assert(decoded?.[0]?.role === "isosurface", "role");
      },
    },
    {
      name: "returns null for unrelated hash",
      fn: async () => {
        const decoded = await decodeExpressionsFragment("#webmcp=1");
        assert(decoded === null, "ignored");
      },
    },
    {
      name: "rejects unsupported version",
      fn: async () => {
        let threw = false;
        try {
          await decodeExpressionsFragment("e=99.eyJsIjoidCJ9");
        } catch (e) {
          threw = true;
          assert(String(e).includes("version"), "version error");
        }
        assert(threw, "should throw");
      },
    },
    {
      name: "strips trailing blank row before encode",
      fn: async () => {
        const fragment = await encodeExpressionsFragment([
          sampleExpr({ latex: "x" }),
          sampleExpr({ latex: "" }),
        ]);
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.length === 1, "blank row omitted");
        assert(decoded![0]?.latex === "x", "kept row");
      },
    },
  ]);
}
