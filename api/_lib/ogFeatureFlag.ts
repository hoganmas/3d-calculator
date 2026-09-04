/**
 * Kill switch for the whole dynamic-per-share OG image system (client
 * capture/upload + server headless-render fallback + Blob cache). Off by
 * default: every share just gets the static build-time og-image.png, which
 * also means no client-facing write surface exists to squat/inject at all.
 * Flip on only in environments where that's an accepted trade-off (testing)
 * until there's an account/auth system to gate uploads by identity instead.
 */
export function isDynamicOgEnabled(): boolean {
  const v = (process.env.OG_DYNAMIC_ENABLED ?? "").trim().toLowerCase();
  return v === "1" || v === "true";
}
