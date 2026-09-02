/** False-color iso refine: cyan = coarse, orange = occupancy edge, magenta = iso intersection.
 *  Enable via ?isoRefineDebug=1, localStorage laplacianIsoRefineDebug=1,
 *  or window.__laplacianIsoRefineDebug(true). */

let enabled: boolean | null = null;

function computeEnabled(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return (
      new URLSearchParams(window.location.search).has("isoRefineDebug") ||
      localStorage.getItem("laplacianIsoRefineDebug") === "1" ||
      (window as Window & { __laplacianIsoRefineDebug?: unknown }).__laplacianIsoRefineDebug === true
    );
  } catch {
    return false;
  }
}

export function isIsoRefineDebugEnabled(): boolean {
  if (enabled === null) enabled = computeEnabled();
  return enabled;
}

export function setIsoRefineDebugEnabled(on: boolean): void {
  enabled = on;
  if (typeof window === "undefined") return;
  if (on) localStorage.setItem("laplacianIsoRefineDebug", "1");
  else localStorage.removeItem("laplacianIsoRefineDebug");
  console.log(
    `[iso-refine] debug ${on ? "ON — cyan = coarse, orange = edge, magenta = iso intersection" : "disabled"}`,
  );
}

export function initIsoRefineDebug(): void {
  if (typeof window === "undefined") return;
  if (enabled === null) enabled = computeEnabled();
  const w = window as Window & {
    __laplacianIsoRefineDebug?: boolean | ((on?: boolean) => boolean);
  };
  w.__laplacianIsoRefineDebug = (on?: boolean) => {
    if (typeof on === "boolean") setIsoRefineDebugEnabled(on);
    return isIsoRefineDebugEnabled();
  };
  if (isIsoRefineDebugEnabled()) {
    console.log(
      "[iso-refine] debug ON — cyan = coarse, orange = edge, magenta = iso intersection. " +
        "Toggle: __laplacianIsoRefineDebug(false)",
    );
  }
}
