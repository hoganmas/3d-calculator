/**
 * Compact expression-list encoding for share paths (/s/…) and legacy URL
 * fragments (#e=…, kept for old links already shared before /s/ existed).
 */
import { listExpressions, setExpressions } from "../../model/expressions.js";
import type { ExprItem } from "../../types/models.js";
import { els } from "../dom.js";
import { BOUNDS_SIZE_MIN, BOUNDS_SIZE_MAX } from "../state.js";
import {
  decodeCompactFragment,
  encodeCompactFragment,
  type DecodedSharePayload,
  FRAGMENT_RE,
} from "./exprShareCodec.js";

/**
 * Mirrors presentation.ts's `readBoundsSize()` without importing it — that
 * module pulls in the WebGPU/device-tier chain, which isn't Node-safe and
 * would break this file's test coverage (exprShare.test.ts / exprShareBench).
 */
function currentBoxSize(): number {
  const n = Number(els.boxSize.value) || 5;
  return Math.min(BOUNDS_SIZE_MAX, Math.max(BOUNDS_SIZE_MIN, Math.round(n * 10) / 10));
}

export {
  EXPR_SHARE_VERSION,
  COMPRESS_THRESHOLD,
  GZIP_THRESHOLD,
  compactExprPayload,
  DEFAULT_BOX_SIZE,
  type CompactExprRow,
  type CompressMode,
  type DecodedSharePayload,
} from "./exprShareCodec.js";

/** Strip optional `#` / `e=` prefix from a share payload body. */
export function normalizeSharePayload(input: string): string {
  let s = input.trim();
  if (s.startsWith("#")) s = s.slice(1);
  if (s.startsWith("e=")) s = s.slice(2);
  return s;
}

export function fragmentFromSharePayload(payload: string): string {
  return `e=${normalizeSharePayload(payload)}`;
}

export function isValidSharePayload(payload: string): boolean {
  const body = normalizeSharePayload(payload);
  return Boolean(body && FRAGMENT_RE.test(`e=${body}`));
}

/** Encode expression rows (+ box size, if given) to a fragment body (`e=1…` / `e=1d…`). */
export async function encodeExpressionsFragment(exprs: ExprItem[], boxSize?: number): Promise<string> {
  return encodeCompactFragment(exprs, "auto", boxSize);
}

/** Decode a fragment body into expression patches (+ box size, if present). */
export async function decodeExpressionsFragment(hash: string): Promise<DecodedSharePayload | null> {
  return decodeCompactFragment(hash);
}

/**
 * Build a full share URL for the current expression list and box size.
 *
 * Routed through /s/<payload> (a Vercel serverless bot-gate, see api/share.ts)
 * rather than a bare query param: crawlers/link-unfurlers hitting /s/ get a
 * static HTML page with a dynamically-rendered og:image, while everyone else
 * gets redirected straight into the app.
 */
export async function buildExpressionShareUrl(baseUrl = location.href): Promise<string> {
  const fragment = await encodeExpressionsFragment(listExpressions(), currentBoxSize());
  const payload = normalizeSharePayload(fragment);
  const url = new URL(baseUrl);
  return `${url.origin}/s/${payload}`;
}

/**
 * Apply `?e=` when present; returns true when expressions were loaded.
 * Sets `els.boxSize.value` directly when the payload carries a box size, but
 * doesn't refresh its label/liquid-thumb UI (that's presentation.ts, kept
 * out of this file's import graph — see `currentBoxSize`). Callers should
 * follow a successful restore with `syncBoundsSlider()`.
 *
 * `?e=` (not `#e=`) because chat/email clients commonly route links through
 * their own redirect for link scanning/unfurling — fragments never reach
 * that server hop and get silently dropped, while query params survive it.
 * (api/share.ts's redirect target uses this same query-param form.)
 */
export async function applyExpressionsFromQuery(search = location.search): Promise<boolean> {
  const qe = new URLSearchParams(search).get("e");
  if (!qe) return false;
  if (!isValidSharePayload(qe)) return false;
  const decoded = await decodeExpressionsFragment(fragmentFromSharePayload(qe));
  if (!decoded) return false;
  setExpressions(decoded.rows);
  if (decoded.boxSize != null) {
    const clamped = Math.min(BOUNDS_SIZE_MAX, Math.max(BOUNDS_SIZE_MIN, decoded.boxSize));
    els.boxSize.value = String(clamped);
  }
  return true;
}

/** Apply legacy `#e=…` when present; returns true when expressions were loaded. */
export async function applyExpressionsFromFragment(hash = location.hash): Promise<boolean> {
  if (!hash || hash === "#") return false;
  const decoded = await decodeExpressionsFragment(hash);
  if (!decoded) return false;
  setExpressions(decoded.rows);
  if (decoded.boxSize != null) {
    const clamped = Math.min(BOUNDS_SIZE_MAX, Math.max(BOUNDS_SIZE_MIN, decoded.boxSize));
    els.boxSize.value = String(clamped);
  }
  return true;
}

export async function copyExpressionShareLink(btn?: HTMLButtonElement): Promise<boolean> {
  const result = await shareExpressionLink();
  if (result === "failed") return false;
  if (btn) {
    const prev = btn.textContent;
    btn.textContent = result === "shared" ? "Shared" : "Copied";
    window.setTimeout(() => {
      btn.textContent = prev;
    }, 1600);
  }
  return true;
}

export type ShareLinkResult = "shared" | "copied" | "failed";

/** Share the current scene as a laplaci.com URL (native share sheet or clipboard). */
export async function shareExpressionLink(): Promise<ShareLinkResult> {
  const url = await buildExpressionShareUrl();
  // Best-effort client-side OG image capture, ahead of actually sharing the
  // link — dynamic import (not a static one) keeps this file's own import
  // graph Node-test-safe, since shareCapture.ts touches the live canvas/
  // scene. A failure here never blocks sharing: api/og.ts falls back to a
  // server-side render if no image was uploaded for this payload.
  try {
    const { captureAndUploadOgImage } = await import("../shareCapture.js");
    await captureAndUploadOgImage(url);
  } catch {
    // ignored — server-side fallback covers this
  }
  // No `text` field: some share targets (and the OS share sheet's own
  // "copy" action) concatenate title/text with the url, so anything here
  // ends up pasted alongside the link wherever it's shared.
  const shareData: ShareData = {
    title: "laplaci",
    url,
  };
  if (typeof navigator.share === "function") {
    try {
      if (typeof navigator.canShare !== "function" || navigator.canShare(shareData)) {
        await navigator.share(shareData);
        return "shared";
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return "failed";
    }
  }
  try {
    await navigator.clipboard.writeText(url);
    return "copied";
  } catch {
    return "failed";
  }
}
