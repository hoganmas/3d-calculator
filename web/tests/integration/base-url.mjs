/** Default dev URL (localhost — Vite may not bind 127.0.0.1 on all hosts). */
export const BASE =
  process.env.LAPLACI_TEST_URL ?? "http://localhost:5173/3d-calculator/?webmcp=1";

/** Unwrap WebMCP tool results: `{ ok, data }` or legacy `{ content }`. */
export function unwrapToolResult(result) {
  if (result == null) return null;
  if (result.data != null && typeof result.data === "object") return result.data;
  if (result.content != null && typeof result.content === "object") return result.content;
  return result;
}
