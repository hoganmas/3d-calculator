import type { VercelRequest, VercelResponse } from "@vercel/node";
import { isCrawlerUserAgent } from "./_lib/bots.js";
import {
  decodeSharePayload,
  normalizeSharePayload,
  sharePanelsFromRows,
  siteUrl,
  validateSharePayload,
} from "./_lib/sharePayload.js";

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildBotHtml(payload: string, title: string, description: string) {
  const origin = siteUrl();
  const ogImage = `${origin}/api/og?e=${encodeURIComponent(payload)}`;
  const pageUrl = `${origin}/s/${payload}`;
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(title)}</title>
  <meta name="description" content="${escapeHtml(description)}" />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="${escapeHtml(pageUrl)}" />
  <meta property="og:title" content="${escapeHtml(title)}" />
  <meta property="og:description" content="${escapeHtml(description)}" />
  <meta property="og:image" content="${escapeHtml(ogImage)}" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="${escapeHtml(title)}" />
  <meta name="twitter:description" content="${escapeHtml(description)}" />
  <meta name="twitter:image" content="${escapeHtml(ogImage)}" />
</head>
<body>
  <p><a href="${escapeHtml(`${origin}/?e=${encodeURIComponent(payload)}`)}">Open laplaci</a></p>
</body>
</html>`;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const raw = String(req.query.payload ?? "");
  if (!raw) {
    res.status(400).send("Missing payload");
    return;
  }

  let payload: string;
  try {
    payload = validateSharePayload(raw);
  } catch {
    res.status(400).send("Invalid share payload");
    return;
  }

  const ua = req.headers["user-agent"];
  if (!isCrawlerUserAgent(Array.isArray(ua) ? ua[0] : ua)) {
    res.setHeader("Location", `/?e=${encodeURIComponent(payload)}`);
    res.status(302).end();
    return;
  }

  try {
    const rows = await decodeSharePayload(payload);
    const panels = sharePanelsFromRows(rows);
    const primary = panels[0]?.label ?? "shared graph";
    const title = `laplaci — ${primary}`;
    const description = panels.length > 1
      ? `Shared scene with ${panels.length} expressions on laplaci.`
      : `Shared expression: ${primary}`;
    res.setHeader("Content-Type", "text/html; charset=utf-8");
    // Vary: without this, a shared/CDN cache keys purely by URL — a crawler
    // fetching this exact link first would let the edge serve this same
    // bot-targeted HTML to every human visitor for the next hour instead of
    // redirecting them into the app.
    res.setHeader("Vary", "User-Agent");
    res.setHeader("Cache-Control", "public, s-maxage=3600, stale-while-revalidate=86400");
    res.status(200).send(buildBotHtml(normalizeSharePayload(payload), title, description));
  } catch {
    res.status(400).send("Could not decode share payload");
  }
}
