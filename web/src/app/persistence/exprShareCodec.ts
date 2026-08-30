/**
 * Low-level expression fragment codec — shared by production encode/decode and benchmarks.
 */
import { EXPR_GRADIENTS } from "../../model/expressions.js";
import type { AnimMode, ExprItem } from "../../types/models.js";

export const EXPR_SHARE_VERSION = 1;
export const FRAGMENT_PREFIX = "e=";
/** Minimum JSON byte length before auto mode compresses the payload. */
export const COMPRESS_THRESHOLD = 150;
/** @deprecated Use COMPRESS_THRESHOLD */
export const GZIP_THRESHOLD = COMPRESS_THRESHOLD;
export const MAX_EXPR_ROWS = 100;
export const MAX_JSON_BYTES = 64_000;

export type CompactExprRow = {
  l: string;
  /** Palette gradient index (0–4) when colors match EXPR_GRADIENTS but differ from row default. */
  g?: number;
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

function normHex(color: string) {
  const s = String(color || "").trim().toLowerCase();
  return s.startsWith("#") ? s : `#${s}`;
}

function defaultGradientForIndex(index: number) {
  return EXPR_GRADIENTS[index % EXPR_GRADIENTS.length]!;
}

function paletteIndexForPair(color: string, color2: string) {
  const c = normHex(color);
  const c2 = normHex(color2);
  for (let i = 0; i < EXPR_GRADIENTS.length; i++) {
    const grad = EXPR_GRADIENTS[i]!;
    if (normHex(grad.color) === c && normHex(grad.color2) === c2) return i;
  }
  return null;
}

function hasCustomGradStops(item: ExprItem) {
  if (!item.colors?.length) return false;
  if (item.colors.length > 2) return true;
  return item.colors.some(
    (c, i) => normHex(c) !== normHex(i === 0 ? item.color : item.color2),
  );
}

function compactAnimFields(row: CompactExprRow, item: ExprItem) {
  if (!item.sliderAnimating) return;
  row.a = 1;
  row.mn = item.sliderMin;
  row.mx = item.sliderMax;
  row.sp = item.sliderSpeed;
  if (item.sliderAnimMode !== "pingpong") row.am = item.sliderAnimMode;
  if (Number.isFinite(item.sliderPhase)) row.ph = item.sliderPhase;
}

function compactColorFields(row: CompactExprRow, item: ExprItem, index: number) {
  if (hasCustomGradStops(item)) {
    row.cs = item.colors.map((c) => normHex(c));
    return;
  }

  const defaultGrad = defaultGradientForIndex(index);
  const matchesDefault =
    normHex(item.color) === normHex(defaultGrad.color) &&
    normHex(item.color2) === normHex(defaultGrad.color2);
  if (matchesDefault) return;

  const paletteIdx = paletteIndexForPair(item.color, item.color2);
  if (paletteIdx != null) {
    row.g = paletteIdx;
    return;
  }

  row.c = normHex(item.color);
  row.c2 = normHex(item.color2);
}

export function compactExprRow(item: ExprItem, index: number): CompactExprRow {
  const row: CompactExprRow = { l: item.latex };
  if (item.enabled === false) row.e = 0;
  compactColorFields(row, item, index);
  compactAnimFields(row, item);
  if (item.autoParam) row.ap = 1;
  return row;
}

export function compactExprPayload(exprs: ExprItem[]): CompactExprRow[] {
  return stripTrailingBlank(exprs).map((item, index) => compactExprRow(item, index));
}

export function expandCompactRow(row: CompactExprRow, index = 0): Partial<ExprItem> {
  if (typeof row.l !== "string") throw new Error("expression row missing latex");
  const out: Partial<ExprItem> = { latex: row.l };
  if (row.e === 0) out.enabled = false;
  if (row.cs?.length) {
    out.colors = row.cs.map((c) => normHex(c));
    out.color = out.colors[0];
    out.color2 = out.colors[out.colors.length - 1];
  } else if (row.g != null) {
    const grad = EXPR_GRADIENTS[row.g] ?? defaultGradientForIndex(index);
    out.color = grad.color;
    out.color2 = grad.color2;
    out.colors = [grad.color, grad.color2];
  } else if (row.c) {
    out.color = normHex(row.c);
    out.color2 = normHex(row.c2 ?? row.c);
    out.colors = [out.color, out.color2];
  }
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
    if (r.g != null && (!Number.isInteger(r.g) || r.g < 0 || r.g >= EXPR_GRADIENTS.length)) {
      throw new Error(`expressions[${i}].g: invalid palette index`);
    }
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

/** Build `e=1[zd]?.…` from JSON bytes and a compression mode. */
export async function buildFragmentFromJsonBytes(
  jsonBytes: Uint8Array,
  mode: CompressMode = "auto",
  version = EXPR_SHARE_VERSION,
): Promise<string> {
  if (mode !== "auto") {
    const body = mode === "none" ? jsonBytes : await compressBytes(jsonBytes, mode);
    return `${FRAGMENT_PREFIX}${version}${compressFlag(mode)}.${bytesToBase64Url(body)}`;
  }

  const rawFragment = `${FRAGMENT_PREFIX}${version}.${bytesToBase64Url(jsonBytes)}`;
  const deflated = await compressBytes(jsonBytes, "deflate");
  const deflateFragment = `${FRAGMENT_PREFIX}${version}d.${bytesToBase64Url(deflated)}`;
  return deflateFragment.length < rawFragment.length ? deflateFragment : rawFragment;
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
  return payload.map((row, i) => expandCompactRow(row, i));
}
