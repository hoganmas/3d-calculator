#!/usr/bin/env node
/**
 * Generate a batch of example OG images for visual review before deploying.
 *
 * Each sample is routed through the *exact* same pipeline production uses
 * (api/og.ts): encode -> share payload -> decodeSharePayload ->
 * sharePanelsFromRows -> renderOgComposite. This isn't a shortcut — it also
 * exercises the panel-selection logic (sharePanelsFromRows), so a regression
 * there (e.g. a graphed expression getting silently dropped, or a parameter
 * dependency not carrying over to its panel) shows up here too, not just
 * rendering bugs.
 *
 * Usage: tsx scripts/og/generate-samples.mjs [siteUrl] [outDir]
 *   siteUrl defaults to https://localhost:5173
 *   outDir defaults to scripts/og/out/samples
 */
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
// exprShareCodec.ts directly, not exprShare.ts's wrapper: exprShare.ts pulls
// in dom.ts (for box-size reads) at module scope, which eagerly calls
// document.getElementById and throws in a plain Node/tsx process with no
// DOM. The codec module is the Node-safe layer exprShare.ts itself wraps.
import { encodeCompactFragment } from "../../src/app/persistence/exprShareCodec.ts";
import {
  decodeSharePayload,
  normalizeSharePayload,
  sharePanelsFromRows,
} from "../../../api/_lib/sharePayload.ts";
import { renderOgComposite } from "./renderShareOg.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "../../..");

function row(overrides) {
  return {
    id: "e1",
    latex: "z=x",
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

// Deliberately covers the cases that have broken before: a plain surface, a
// vector field, a volumetric/abs expression, a single-letter alias (the
// sharePanelsFromRows regex regression), an animated/parametric expression
// (the paramRows regression), and a multi-panel share.
const SAMPLES = {
  "static-wave": [row({ latex: String.raw`z=-\cos\left(x\right)\sin\left(2y\right)` })],
  "vector-field": [row({ latex: String.raw`\left(0,z,-y\right)` })],
  "implicit-abs": [row({ latex: String.raw`e^{-2.5r}\abs\left(2z^{2}-x^{2}-y^{2}\right)` })],
  "alias-named-T": [row({ latex: "T=x^2+y^2-z" })],
  "animated-param": [
    row({ id: "e1", latex: "t=0.4", autoParam: true, sliderAnimating: true, sliderMin: 0, sliderMax: 1 }),
    row({ id: "e2", latex: String.raw`y=\sin\left(x+2\pi t\right)\cos\left(z\right)` }),
  ],
  "multi-panel-scene": [
    row({ id: "e1", latex: "z=x^2-y^2" }),
    row({ id: "e2", latex: String.raw`z=\sin\left(x\right)\cos\left(y\right)` }),
    row({ id: "e3", latex: "x^2+y^2+z^2=4" }),
  ],
};

async function main() {
  const [, , siteUrlArg, outDirArg] = process.argv;
  const siteUrl = siteUrlArg ?? "https://localhost:5173";
  const outDir = outDirArg ?? join(root, "web/scripts/og/out/samples");
  mkdirSync(outDir, { recursive: true });

  const logoSvg = readFileSync(join(root, "web/public/logo.svg"), "utf8");
  const names = Object.keys(SAMPLES);
  let failed = 0;

  for (const name of names) {
    try {
      const fragment = await encodeCompactFragment(SAMPLES[name], "auto");
      const payload = normalizeSharePayload(fragment);
      const rows = await decodeSharePayload(payload);
      const panels = sharePanelsFromRows(rows);
      if (!panels.length) throw new Error("sharePanelsFromRows returned no panels");

      console.log(`Rendering ${name} (${panels.length} panel${panels.length === 1 ? "" : "s"})...`);
      const png = await renderOgComposite({ siteUrl, scenes: panels, logoSvg });
      const outPath = join(outDir, `${name}.png`);
      writeFileSync(outPath, png);
      console.log(`  -> ${outPath}`);
    } catch (err) {
      failed++;
      console.error(`  FAILED ${name}: ${err instanceof Error ? err.message : err}`);
    }
  }

  console.log(`\n${names.length - failed}/${names.length} sample images written to ${outDir}`);
  if (failed) process.exit(1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
