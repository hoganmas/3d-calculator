import { gzipSync } from "node:zlib";
import {
  applyExpressionsFromFragment,
  applyExpressionsFromQuery,
  buildExpressionShareUrl,
  decodeExpressionsFragment,
  encodeExpressionsFragment,
  EXPR_SHARE_VERSION,
  fragmentFromSharePayload,
  isValidSharePayload,
  normalizeSharePayload,
} from "../../src/app/persistence/exprShare.ts";
import { listExpressions, setExpressions } from "../../src/model/expressions.ts";
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
        assert(decoded!.rows[0]?.latex === sampleExpr().latex, "latex");
        assert(decoded!.boxSize === undefined, "no box size when not shared");
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
        assert(decoded?.rows.length === 2, "two rows");
        assert(decoded!.rows[0]?.autoParam === true, "autoParam");
        assert(decoded!.rows[0]?.sliderAnimating === true, "animating");
        assert(decoded!.rows[0]?.sliderAnimMode === "loop", "loop mode");
        assert(decoded!.rows[1]?.latex.includes("\\sin"), "field latex");
      },
    },
    {
      name: "uses deflate for larger payloads",
      fn: async () => {
        const exprs = Array.from({ length: 8 }, (_, i) =>
          sampleExpr({
            id: `e${i + 1}`,
            latex: String.raw`\exp(-${i}((x-a)^2+(y-b)^2+(z-${i})^2))`,
          }),
        );
        const fragment = await encodeExpressionsFragment(exprs);
        assert(fragment.startsWith(`e=${EXPR_SHARE_VERSION}d.`), "uses deflate encoding");
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.rows.length === 8, "row count");
        assert(decoded!.rows[1]?.latex.includes("(x-a)"), "latex preserved");
      },
    },
    {
      name: "omits default palette colors and round-trips explicit palette index",
      fn: async () => {
        const exprs = [
          sampleExpr({ id: "e1", latex: "x" }),
          sampleExpr({
            id: "e2",
            latex: "y",
            color: "#ff4500",
            color2: "#ffec00",
            colors: ["#ff4500", "#ffec00"],
          }),
          sampleExpr({
            id: "e3",
            latex: "z",
            color: "#ff1493",
            color2: "#7b2fff",
            colors: ["#ff1493", "#7b2fff"],
          }),
        ];
        const fragment = await encodeExpressionsFragment(exprs);
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.rows.length === 3, "three rows");
        assert(decoded!.rows[2]?.color === "#ff1493", "palette colors restored");
        assert(decoded!.rows[2]?.color2 === "#7b2fff", "palette colors restored");
      },
    },
    {
      name: "decodes manually gzipped fragment",
      fn: async () => {
        const payload = [{ l: "x^2+y^2+z^2=1" }];
        const gz = gzipSync(JSON.stringify(payload));
        const fragment = `e=${EXPR_SHARE_VERSION}z.${bytesToBase64Url(gz)}`;
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.rows[0]?.latex === payload[0]!.l, "latex");
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
        assert(decoded?.rows.length === 1, "blank row omitted");
        assert(decoded!.rows[0]?.latex === "x", "kept row");
      },
    },
    {
      name: "round-trips box size, omits it when default",
      fn: async () => {
        const withDefault = await encodeExpressionsFragment([sampleExpr()], 5);
        const decodedDefault = await decodeExpressionsFragment(withDefault);
        assert(decodedDefault?.boxSize === undefined, "default box size omitted from payload");

        const withCustom = await encodeExpressionsFragment([sampleExpr()], 8.5);
        const decodedCustom = await decodeExpressionsFragment(withCustom);
        assert(decodedCustom?.boxSize === 8.5, "custom box size round-trips");
        assert(decodedCustom?.rows.length === 1, "box size sentinel not counted as a row");
        assert(decodedCustom!.rows[0]?.latex === sampleExpr().latex, "row content unaffected");
      },
    },
    {
      name: "legacy bare-array fragment (no box size) still decodes",
      fn: async () => {
        const payload = [{ l: "x+y+z=1" }];
        const fragment = `e=${EXPR_SHARE_VERSION}.${bytesToBase64Url(new TextEncoder().encode(JSON.stringify(payload)))}`;
        const decoded = await decodeExpressionsFragment(fragment);
        assert(decoded?.boxSize === undefined, "no box size in legacy fragment");
        assert(decoded?.rows[0]?.latex === "x+y+z=1", "latex");
      },
    },
    {
      name: "buildExpressionShareUrl produces a /s/<payload> URL, not a hash or query link",
      fn: async () => {
        // /s/ is a Vercel serverless bot-gate (api/share.ts): crawlers get a
        // static page with a dynamically-rendered og:image, humans get
        // redirected into the app via ?e=. Fragments are avoided entirely —
        // chat/email clients route links through their own redirect for
        // link scanning, and fragments never reach that server hop.
        setExpressions([sampleExpr({ id: "e1", latex: "x^2" })]);
        const url = await buildExpressionShareUrl("https://laplaci.com/");
        const parsed = new URL(url);
        assert(parsed.hash === "", "no hash used");
        assert(parsed.search === "", "no query string used");
        assert(parsed.pathname.startsWith(`/s/${EXPR_SHARE_VERSION}.`), "payload in /s/ path");
      },
    },
    {
      name: "applyExpressionsFromQuery round-trips a /s/ payload via the ?e= redirect target",
      fn: async () => {
        // Mirrors api/share.ts's redirect for non-crawler UAs: /?e=<payload>.
        setExpressions([sampleExpr({ id: "e1", latex: "z" })]);
        const shareUrl = await buildExpressionShareUrl("https://laplaci.com/");
        const payload = new URL(shareUrl).pathname.slice("/s/".length);
        setExpressions([sampleExpr({ id: "e1", latex: "unrelated-local-state" })]);
        const applied = await applyExpressionsFromQuery(`?e=${payload}`);
        assert(applied, "loaded from query");
        assert(listExpressions()[0]?.latex === "z", "query content wins over prior local state");
      },
    },
    {
      name: "applyExpressionsFromQuery returns false when there's no e param",
      fn: async () => {
        assert((await applyExpressionsFromQuery("")) === false, "empty search");
        assert((await applyExpressionsFromQuery("?webmcp=1")) === false, "unrelated param only");
      },
    },
    {
      name: "applyExpressionsFromFragment still restores legacy #e= links",
      fn: async () => {
        setExpressions([sampleExpr({ id: "e1", latex: "w" })]);
        const fragment = await encodeExpressionsFragment(listExpressions());
        setExpressions([sampleExpr({ id: "e1", latex: "unrelated-local-state" })]);
        const applied = await applyExpressionsFromFragment(`#${fragment}`);
        assert(applied, "loaded from legacy fragment");
        assert(listExpressions()[0]?.latex === "w", "fragment content wins over prior local state");
      },
    },
    {
      name: "normalizes share payload for /s/ paths",
      fn: async () => {
        const fragment = await encodeExpressionsFragment([sampleExpr()]);
        const body = normalizeSharePayload(fragment);
        assert(!body.startsWith("e="), "strips e= prefix");
        assert(isValidSharePayload(body), "valid payload");
        const roundTrip = await decodeExpressionsFragment(fragmentFromSharePayload(body));
        assert(roundTrip?.rows[0]?.latex === sampleExpr().latex, "round-trip via payload");
      },
    },
    {
      name: "rejects invalid share payload",
      fn: async () => {
        assert(!isValidSharePayload(""), "empty");
        assert(!isValidSharePayload("not-a-share"), "garbage");
      },
    },
  ]);
}
