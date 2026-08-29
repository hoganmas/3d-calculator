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
  GZIP_THRESHOLD,
  compactExprPayload,
  type CompactExprRow,
  type CompressMode,
} from "./exprShareCodec.js";

/** Encode expression rows to a fragment body (`e=1…` / `e=1z…`). */
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
  const url = await buildExpressionShareUrl();
  try {
    await navigator.clipboard.writeText(url);
    if (btn) {
      const prev = btn.textContent;
      btn.textContent = "Copied";
      window.setTimeout(() => {
        btn.textContent = prev;
      }, 1600);
    }
    return true;
  } catch {
    return false;
  }
}
