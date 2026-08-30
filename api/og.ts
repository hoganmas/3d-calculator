import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { VercelRequest, VercelResponse } from "@vercel/node";
import {
  decodeSharePayload,
  normalizeSharePayload,
  sharePanelsFromRows,
  siteUrl,
  validateSharePayload,
} from "./_lib/sharePayload.js";
import { renderShareOgPng } from "../web/scripts/og/renderShareOg.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const fallbackPng = join(root, "web/dist/og-image.png");
const logoSvg = join(root, "web/public/logo.svg");

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const raw = String(req.query.e ?? "");
  if (!raw) {
    res.status(400).send("Missing e parameter");
    return;
  }

  let payload: string;
  try {
    payload = validateSharePayload(raw);
  } catch {
    res.status(400).send("Invalid share payload");
    return;
  }

  try {
    const rows = await decodeSharePayload(payload);
    const panels = sharePanelsFromRows(rows);
    if (!panels.length) {
      res.status(400).send("No visual expressions in payload");
      return;
    }

    const png = await renderShareOgPng({
      siteUrl: siteUrl(),
      panels,
      logoSvg: readFileSync(logoSvg, "utf8"),
      ogDeg: Number(process.env.OG_CAPTURE_DEG ?? 16),
    });

    res.setHeader("Content-Type", "image/png");
    res.setHeader("Cache-Control", "public, s-maxage=86400, stale-while-revalidate=604800");
    res.status(200).send(png);
  } catch (err) {
    console.error("[api/og]", err);
    try {
      const fallback = readFileSync(fallbackPng);
      res.setHeader("Content-Type", "image/png");
      res.setHeader("Cache-Control", "public, s-maxage=300");
      res.status(200).send(fallback);
    } catch {
      res.status(500).send("OG render failed");
    }
  }
}
