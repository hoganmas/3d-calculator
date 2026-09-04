#!/usr/bin/env node
/**
 * Render an example OG image for a given laplaci share URL — the exact same
 * pipeline api/og.ts uses (decode payload -> headless-render each expression
 * -> composite), run locally so capture.mjs/composite.mjs can be iterated on
 * without a Vercel deploy.
 *
 * The app is loaded from the given URL's own origin, so point this at
 * whichever instance you want to render against: local dev server, a Vercel
 * preview deployment, or prod.
 *
 * Usage:
 *   tsx scripts/og/render-from-url.mjs <url> [outPath]
 *
 *   <url> accepts either share-link shape:
 *     http://localhost:5173/s/<payload>
 *     http://localhost:5173/?e=<payload>
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { decodeSharePayload, sharePanelsFromRows } from "../../../api/_lib/sharePayload.ts";
import { renderShareOgPng } from "./renderShareOg.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "../../..");

function payloadFromUrl(input) {
  const url = new URL(input);
  const sPrefix = "/s/";
  if (url.pathname.startsWith(sPrefix)) {
    return decodeURIComponent(url.pathname.slice(sPrefix.length));
  }
  const qe = url.searchParams.get("e");
  if (qe) return qe;
  throw new Error(`No share payload found in ${input} (expected /s/<payload> or ?e=<payload>)`);
}

async function main() {
  const [, , inputUrl, outArg] = process.argv;
  if (!inputUrl) {
    console.error("Usage: render-from-url.mjs <url> [outPath]");
    process.exit(1);
  }

  const url = new URL(inputUrl);
  const payload = payloadFromUrl(inputUrl);
  const rows = await decodeSharePayload(payload);
  const panels = sharePanelsFromRows(rows);
  if (!panels.length) throw new Error("No visual expressions in payload");

  const logoSvg = readFileSync(join(root, "web/public/logo.svg"), "utf8");
  const png = await renderShareOgPng({
    siteUrl: url.origin,
    rows,
    logoSvg,
    // Omit unless OG_CAPTURE_DEG is set: ogCapture.ts forces max quality by
    // default; this is only an escape hatch for fast local iteration.
    ogDeg: process.env.OG_CAPTURE_DEG ? Number(process.env.OG_CAPTURE_DEG) : undefined,
  });

  const outPath = outArg ?? join(root, "web/scripts/og/out", `og-${Date.now()}.png`);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, png);
  console.log(`Wrote ${outPath} (${panels.length} expression${panels.length === 1 ? "" : "s"}: ${panels.map((p) => p.label).join(", ")})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
