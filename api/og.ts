import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { head, put } from "@vercel/blob";
import {
  decodeSharePayload,
  normalizeSharePayload,
  sharePanelsFromRows,
  siteUrl,
  validateSharePayload,
} from "./_lib/sharePayload.js";
import { ogBlobKey } from "./_lib/ogBlob.js";
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

  // Fast path: the sharer's own browser already captured and uploaded this
  // scene at share-creation time (shareCapture.ts) — redirect to Vercel
  // Blob's CDN instead of paying for a headless-Chromium render.
  const key = ogBlobKey(payload);
  try {
    const stored = await head(key);
    if (stored?.url) {
      res.setHeader("Cache-Control", "public, s-maxage=86400, stale-while-revalidate=604800");
      res.redirect(302, stored.url);
      return;
    }
  } catch {
    // Not stored yet (client capture failed, raced this request, or
    // predates the feature) — fall through to the server-side render.
  }

  try {
    const rows = await decodeSharePayload(payload);
    if (!sharePanelsFromRows(rows).length) {
      res.status(400).send("No visual expressions in payload");
      return;
    }

    const png = await renderShareOgPng({
      siteUrl: siteUrl(),
      rows,
      logoSvg: readFileSync(logoSvg, "utf8"),
      // Omit unless OG_CAPTURE_DEG is explicitly set: ogCapture.ts forces
      // max quality by default, this is only an escape hatch to deliberately
      // trade quality for speed/cost.
      ogDeg: process.env.OG_CAPTURE_DEG ? Number(process.env.OG_CAPTURE_DEG) : undefined,
    });

    // Opportunistically cache this render so future requests for the same
    // payload hit the fast path above instead of re-rendering every time.
    // Best-effort: shouldn't fail the actual response if the store fails.
    put(key, png, { access: "public", contentType: "image/png", addRandomSuffix: false }).catch((err) =>
      console.error("[api/og] cache-store failed", err),
    );

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
