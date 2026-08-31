/** Curated MathLive inline-shortcut whitelist state (no UI or MathLive deps). */

const STORAGE_KEY = "laplacian-curated-shortcuts";

let cached: boolean | null = null;

function isProdBuild(): boolean {
  return import.meta.env?.PROD === true;
}

function readUrlOverride(): boolean | null {
  if (typeof window === "undefined") return null;
  try {
    const v = new URLSearchParams(window.location.search).get("curatedShortcuts");
    if (v === "1" || v === "true") return true;
    if (v === "0" || v === "false") return false;
  } catch {
    /* ignore */
  }
  return null;
}

function readStored(): boolean | null {
  if (typeof window === "undefined") return null;
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "1") return true;
    if (v === "0") return false;
  } catch {
    /* ignore */
  }
  return null;
}

/** Curated shortcut whitelist is on by default; dev can disable via settings or `?curatedShortcuts=0`. */
export function isCuratedShortcutsEnabled(): boolean {
  if (isProdBuild()) return true;
  if (cached !== null) return cached;
  cached = readUrlOverride() ?? readStored() ?? true;
  return cached;
}

export function setCuratedShortcutsState(on: boolean): void {
  if (isProdBuild()) return;
  cached = on;
  if (typeof window === "undefined") return;
  try {
    if (on) localStorage.setItem(STORAGE_KEY, "1");
    else localStorage.setItem(STORAGE_KEY, "0");
  } catch {
    /* ignore */
  }
}

export function resetCuratedShortcutsCache(): void {
  cached = null;
}
