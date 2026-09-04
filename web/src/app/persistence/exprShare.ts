/**
 * Compact expression-list encoding for URL fragments (#e=…).
 */
import { listExpressions, setExpressions } from "../../model/expressions.js";
import type { ExprItem } from "../../types/models.js";
import { els } from "../dom.js";
import { BOUNDS_SIZE_MIN, BOUNDS_SIZE_MAX } from "../state.js";
import {
  decodeCompactFragment,
  encodeCompactFragment,
  type DecodedSharePayload,
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

/** Encode expression rows (+ box size, if given) to a fragment body (`e=1…` / `e=1d…`). */
export async function encodeExpressionsFragment(exprs: ExprItem[], boxSize?: number): Promise<string> {
  return encodeCompactFragment(exprs, "auto", boxSize);
}

/** Decode a fragment body or `#…` hash into expression patches (+ box size, if present). */
export async function decodeExpressionsFragment(hash: string): Promise<DecodedSharePayload | null> {
  return decodeCompactFragment(hash);
}

/** Build a full share URL for the current expression list and box size. */
export async function buildExpressionShareUrl(baseUrl = location.href): Promise<string> {
  const fragment = await encodeExpressionsFragment(listExpressions(), currentBoxSize());
  const url = new URL(baseUrl);
  url.hash = fragment;
  return url.toString();
}

/**
 * Apply `#e=…` when present; returns true when expressions were loaded.
 * Sets `els.boxSize.value` directly when the fragment carries a box size, but
 * doesn't refresh its label/liquid-thumb UI (that's presentation.ts, kept
 * out of this file's import graph — see `currentBoxSize`). Callers should
 * follow a successful restore with `syncBoundsSlider()`.
 */
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
  const shareData: ShareData = {
    title: "laplaci",
    text: "Open this link to view the graph",
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
