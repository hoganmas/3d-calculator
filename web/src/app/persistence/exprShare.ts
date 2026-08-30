/**
 * Compact expression-list encoding for URL fragments (#e=…).
 */
import { listExpressions, setExpressions } from "../../model/expressions.js";
import type { ExprItem } from "../../types/models.js";
import {
  decodeCompactFragment,
  encodeCompactFragment,
} from "./exprShareCodec.js";

export {
  EXPR_SHARE_VERSION,
  COMPRESS_THRESHOLD,
  GZIP_THRESHOLD,
  compactExprPayload,
  type CompactExprRow,
  type CompressMode,
} from "./exprShareCodec.js";

/** Encode expression rows to a fragment body (`e=1…` / `e=1d…` / legacy `e=1z…`). */
export async function encodeExpressionsFragment(exprs: ExprItem[]): Promise<string> {
  return encodeCompactFragment(exprs, "auto");
}

/** Decode a fragment body or `#…` hash into expression patches. */
export async function decodeExpressionsFragment(hash: string): Promise<Partial<ExprItem>[] | null> {
  return decodeCompactFragment(hash);
}

/** Build a full share URL for the current expression list. */
export async function buildExpressionShareUrl(baseUrl = location.href): Promise<string> {
  const fragment = await encodeExpressionsFragment(listExpressions());
  const url = new URL(baseUrl);
  url.hash = fragment;
  return url.toString();
}

/** Apply `#e=…` when present; returns true when expressions were loaded. */
export async function applyExpressionsFromFragment(hash = location.hash): Promise<boolean> {
  if (!hash || hash === "#") return false;
  const rows = await decodeExpressionsFragment(hash);
  if (!rows) return false;
  setExpressions(rows);
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
