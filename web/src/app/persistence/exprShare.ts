/**
 * Compact expression-list encoding for URL fragments (#e=…).
 */
import { listExpressions, setExpressions } from "../../model/expressions.js";
import type { AnimMode, ExprItem, ExprRole } from "../../types/models.js";

export const EXPR_SHARE_VERSION = 1;
const FRAGMENT_PREFIX = "e=";
const GZIP_THRESHOLD = 150;
const MAX_EXPR_ROWS = 100;
const MAX_JSON_BYTES = 64_000;

type CompactExprRow = {
  l: string;
  r?: ExprRole;
  e?: 0;
  c?: string;
  c2?: string;
  cs?: string[];
  a?: 1;
  mn?: number;
  mx?: number;
  sp?: number;
  am?: AnimMode;
  ph?: number;
  ap?: 1;
};

const FRAGMENT_RE = /^e=(\d+)(z)?\.([A-Za-z0-9_-]+)$/;

function stripTrailingBlank(exprs: ExprItem[]): ExprItem[] {
  const copy = exprs.slice();
  while (copy.length > 0) {
    const last = copy[copy.length - 1]!;
    if (String(last.latex || "").trim()) break;
    copy.pop();
  }
  return copy;
}

function isDefaultRole(role: ExprRole | undefined) {
  return !role || role === "auto";
}

function compactRow(item: ExprItem): CompactExprRow {
  const row: CompactExprRow = { l: item.latex };
  if (!isDefaultRole(item.role)) row.r = item.role;
  if (item.enabled === false) row.e = 0;
  if (item.color) row.c = item.color;
  if (item.color2) row.c2 = item.color2;
  if (item.colors?.length && (item.colors.length > 2 || item.colors.some((c, i) => i < 2 && c !== (i === 0 ? item.color : item.color2)))) {
    row.cs = item.colors.slice();
  }
  if (item.sliderAnimating) {
    row.a = 1;
    row.mn = item.sliderMin;
    row.mx = item.sliderMax;
    row.sp = item.sliderSpeed;
    if (item.sliderAnimMode !== "pingpong") row.am = item.sliderAnimMode;
    if (Number.isFinite(item.sliderPhase)) row.ph = item.sliderPhase;
  }
  if (item.autoParam) row.ap = 1;
  return row;
}

function expandRow(row: CompactExprRow): Partial<ExprItem> {
  if (typeof row.l !== "string") throw new Error("expression row missing latex");
  const out: Partial<ExprItem> = { latex: row.l };
  if (row.r) out.role = row.r;
  if (row.e === 0) out.enabled = false;
  if (row.c) out.color = row.c;
  if (row.c2) out.color2 = row.c2;
  if (row.cs?.length) out.colors = row.cs.slice();
  if (row.a === 1) {
    out.sliderAnimating = true;
    if (row.mn != null) out.sliderMin = row.mn;
    if (row.mx != null) out.sliderMax = row.mx;
    if (row.sp != null) out.sliderSpeed = row.sp;
    if (row.am) out.sliderAnimMode = row.am;
    if (row.ph != null) out.sliderPhase = row.ph;
  }
  if (row.ap === 1) out.autoParam = true;
  return out;
}

function validateCompactPayload(payload: unknown): CompactExprRow[] {
  if (!Array.isArray(payload)) throw new Error("expected expression array");
  if (payload.length === 0) throw new Error("expression list is empty");
  if (payload.length > MAX_EXPR_ROWS) throw new Error(`too many expressions (max ${MAX_EXPR_ROWS})`);
  return payload.map((row, i) => {
    if (!row || typeof row !== "object" || Array.isArray(row)) {
      throw new Error(`expressions[${i}]: expected object`);
    }
    const r = row as CompactExprRow;
    if (typeof r.l !== "string") throw new Error(`expressions[${i}].l: expected string`);
    return r;
  });
}

function bytesToBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlToBytes(encoded: string): Uint8Array {
  const pad = encoded.length % 4 ? "=".repeat(4 - (encoded.length % 4)) : "";
  const b64 = encoded.replace(/-/g, "+").replace(/_/g, "/") + pad;
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function gzipBytes(input: Uint8Array): Promise<Uint8Array> {
  if (typeof CompressionStream === "undefined") {
    throw new Error("CompressionStream unavailable");
  }
  const bytes = Uint8Array.from(input);
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function gunzipBytes(input: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("DecompressionStream unavailable");
  }
  const bytes = Uint8Array.from(input);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Encode expression rows to a fragment body (`e=1…` / `e=1z…`). */
export async function encodeExpressionsFragment(exprs: ExprItem[]): Promise<string> {
  const payload = stripTrailingBlank(exprs).map(compactRow);
  if (!payload.length) throw new Error("nothing to share");
  const json = JSON.stringify(payload);
  if (json.length > MAX_JSON_BYTES) throw new Error("expression list too large to share");

  const jsonBytes = new TextEncoder().encode(json);
  const useGzip = jsonBytes.length >= GZIP_THRESHOLD;
  if (useGzip) {
    const compressed = await gzipBytes(jsonBytes);
    return `${FRAGMENT_PREFIX}${EXPR_SHARE_VERSION}z.${bytesToBase64Url(compressed)}`;
  }
  return `${FRAGMENT_PREFIX}${EXPR_SHARE_VERSION}.${bytesToBase64Url(jsonBytes)}`;
}

/** Decode a fragment body or `#…` hash into expression patches. */
export async function decodeExpressionsFragment(hash: string): Promise<Partial<ExprItem>[] | null> {
  const body = hash.startsWith("#") ? hash.slice(1) : hash;
  const match = FRAGMENT_RE.exec(body);
  if (!match) return null;

  const version = Number(match[1]);
  if (version !== EXPR_SHARE_VERSION) throw new Error(`unsupported expression share version: ${version}`);

  const compressed = match[2] === "z";
  const encoded = match[3]!;
  const rawBytes = base64UrlToBytes(encoded);
  const jsonBytes = compressed ? await gunzipBytes(rawBytes) : rawBytes;
  if (jsonBytes.length > MAX_JSON_BYTES) throw new Error("shared expression payload too large");

  const json = new TextDecoder().decode(jsonBytes);
  const payload = validateCompactPayload(JSON.parse(json));
  return payload.map(expandRow);
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
