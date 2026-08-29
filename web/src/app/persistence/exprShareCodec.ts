/**
 * Low-level expression fragment codec — shared by production encode/decode and benchmarks.
 */
import type { AnimMode, ExprItem, ExprRole } from "../../types/models.js";

export const EXPR_SHARE_VERSION = 1;
export const FRAGMENT_PREFIX = "e=";
export const GZIP_THRESHOLD = 150;
export const MAX_EXPR_ROWS = 100;
export const MAX_JSON_BYTES = 64_000;

export type CompactExprRow = {
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

export type CompressMode = "none" | "gzip" | "deflate" | "auto";

export const FRAGMENT_RE = /^e=(\d+)([zd]?)\.([A-Za-z0-9_-]+)$/;

export function stripTrailingBlank(exprs: ExprItem[]): ExprItem[] {
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

export function compactExprRow(item: ExprItem): CompactExprRow {
  const row: CompactExprRow = { l: item.latex };
  if (!isDefaultRole(item.role)) row.r = item.role;
  if (item.enabled === false) row.e = 0;
  if (item.color) row.c = item.color;
  if (item.color2) row.c2 = item.color2;
  if (
    item.colors?.length &&
    (item.colors.length > 2 ||
      item.colors.some((c, i) => i < 2 && c !== (i === 0 ? item.color : item.color2)))
  ) {
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

export function compactExprPayload(exprs: ExprItem[]): CompactExprRow[] {
  return stripTrailingBlank(exprs).map(compactExprRow);
}

export function expandCompactRow(row: CompactExprRow): Partial<ExprItem> {
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

export function validateCompactPayload(payload: unknown): CompactExprRow[] {
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

export function bytesToBase64Url(bytes: Uint8Array): string {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function base64UrlToBytes(encoded: string): Uint8Array {
  const pad = encoded.length % 4 ? "=".repeat(4 - (encoded.length % 4)) : "";
  const b64 = encoded.replace(/-/g, "+").replace(/_/g, "/") + pad;
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function compressFlag(mode: Exclude<CompressMode, "auto">): "" | "z" | "d" {
  if (mode === "gzip") return "z";
  if (mode === "deflate") return "d";
  return "";
}

export async function compressBytes(input: Uint8Array, mode: Exclude<CompressMode, "auto">): Promise<Uint8Array> {
  if (mode === "none") return input;
  const format = mode === "gzip" ? "gzip" : "deflate";
  if (typeof CompressionStream === "undefined") {
    throw new Error("CompressionStream unavailable");
  }
  const bytes = Uint8Array.from(input);
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream(format));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function decompressBytes(input: Uint8Array, flag: "" | "z" | "d"): Promise<Uint8Array> {
  if (!flag) return input;
  const format = flag === "z" ? "gzip" : "deflate";
  if (typeof DecompressionStream === "undefined") {
    throw new Error("DecompressionStream unavailable");
  }
  const bytes = Uint8Array.from(input);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(format));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function pickCompressMode(jsonByteLength: number, mode: CompressMode): Exclude<CompressMode, "auto"> {
  if (mode !== "auto") return mode;
  return jsonByteLength >= GZIP_THRESHOLD ? "gzip" : "none";
}

/** Build `e=1[zd]?.…` from JSON bytes and a compression mode. */
export async function buildFragmentFromJsonBytes(
  jsonBytes: Uint8Array,
  mode: CompressMode = "auto",
  version = EXPR_SHARE_VERSION,
): Promise<string> {
  const picked = pickCompressMode(jsonBytes.length, mode);
  const body = picked === "none" ? jsonBytes : await compressBytes(jsonBytes, picked);
  return `${FRAGMENT_PREFIX}${version}${compressFlag(picked)}.${bytesToBase64Url(body)}`;
}

export async function encodeCompactFragment(
  exprs: ExprItem[],
  mode: CompressMode = "auto",
): Promise<string> {
  const payload = compactExprPayload(exprs);
  if (!payload.length) throw new Error("nothing to share");
  const json = JSON.stringify(payload);
  if (json.length > MAX_JSON_BYTES) throw new Error("expression list too large to share");
  return buildFragmentFromJsonBytes(new TextEncoder().encode(json), mode);
}

export async function decodeCompactFragment(hash: string): Promise<Partial<ExprItem>[] | null> {
  const body = hash.startsWith("#") ? hash.slice(1) : hash;
  const match = FRAGMENT_RE.exec(body);
  if (!match) return null;

  const version = Number(match[1]);
  if (version !== EXPR_SHARE_VERSION) throw new Error(`unsupported expression share version: ${version}`);

  const flag = (match[2] || "") as "" | "z" | "d";
  const encoded = match[3]!;
  const rawBytes = base64UrlToBytes(encoded);
  const jsonBytes = await decompressBytes(rawBytes, flag);
  if (jsonBytes.length > MAX_JSON_BYTES) throw new Error("shared expression payload too large");

  const json = new TextDecoder().decode(jsonBytes);
  const payload = validateCompactPayload(JSON.parse(json));
  return payload.map(expandCompactRow);
}
