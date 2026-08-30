/** Diagnostic logging for iso/keyframe buffer tearing. Enable via ?tearDebug=1 or window.__laplacianTearDebug = true */

const MAX_EVENTS = 200;
const events: { t: number; event: string; data: Record<string, unknown> }[] = [];
const onceKeys = new Set<string>();

let enabled: boolean | null = null;

function computeEnabled(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return (
      new URLSearchParams(window.location.search).has("tearDebug") ||
      localStorage.getItem("laplacianTearDebug") === "1" ||
      (window as Window & { __laplacianTearDebug?: boolean }).__laplacianTearDebug === true
    );
  } catch {
    return false;
  }
}

export function isTearDebugEnabled(): boolean {
  if (enabled === null) enabled = computeEnabled();
  return enabled;
}

export function setTearDebugEnabled(on: boolean): void {
  enabled = on;
  if (typeof window === "undefined") return;
  if (on) localStorage.setItem("laplacianTearDebug", "1");
  else localStorage.removeItem("laplacianTearDebug");
  console.log(`[tear] debug ${on ? "enabled" : "disabled"}`);
}

export function gridMFromDens(dens: Float32Array | undefined | null): number {
  if (!dens?.length) return 0;
  return Math.round(Math.cbrt(dens.length));
}

export function tearLog(event: string, data: Record<string, unknown> = {}): void {
  if (!isTearDebugEnabled()) return;
  const row = { t: performance.now(), event, data };
  events.push(row);
  if (events.length > MAX_EVENTS) events.shift();
  console.log(`[tear] ${event}`, data);
}

export function tearLogOnce(key: string, event: string, data: Record<string, unknown> = {}): void {
  if (!isTearDebugEnabled() || onceKeys.has(key)) return;
  onceKeys.add(key);
  tearLog(event, data);
}

export function tearLogBlendChange(
  layerId: string,
  prev: { i0: number; i1: number; t: number; d0?: number; d1?: number; M0?: number; M1?: number } | null,
  next: { i0: number; i1: number; t: number; d0?: number; d1?: number; M0?: number; M1?: number },
): void {
  if (!isTearDebugEnabled()) return;
  if (
    prev &&
    prev.i0 === next.i0 &&
    prev.i1 === next.i1 &&
    Math.abs(prev.t - next.t) < 0.002 &&
    prev.d0 === next.d0 &&
    prev.d1 === next.d1 &&
    prev.M0 === next.M0 &&
    prev.M1 === next.M1
  ) {
    return;
  }
  tearLog("blend-change", { layerId, prev, next });
}

export function getTearDebugSnapshot() {
  return {
    enabled: isTearDebugEnabled(),
    eventCount: events.length,
    events: events.slice(),
  };
}

export function initTearDebug(): void {
  if (typeof window === "undefined") return;
  if (enabled === null) enabled = computeEnabled();
  const w = window as Window & {
    __laplacianTearDebug?: boolean | (() => ReturnType<typeof getTearDebugSnapshot>);
    __laplacianTear?: () => ReturnType<typeof getTearDebugSnapshot>;
    __laplacianKeyframes?: () => Promise<{
      load: unknown;
      layers: unknown;
      tear: ReturnType<typeof getTearDebugSnapshot>;
    }>;
  };
  w.__laplacianTear = getTearDebugSnapshot;
  w.__laplacianTearDebug = w.__laplacianTearDebug === true ? true : getTearDebugSnapshot;
  w.__laplacianKeyframes = async () => {
    const kf = await import("../model/keyframes.js");
    return {
      load: kf.getKeyframeLoadSummary(),
      layers: kf.diagnoseKeyframeCaches(),
      tear: getTearDebugSnapshot(),
    };
  };
  if (isTearDebugEnabled()) {
    console.log(
      '[tear] debug ON — filter console with "[tear]", copy logs via copy(JSON.stringify(window.__laplacianTear())); ' +
        "keyframe stall probe: await window.__laplacianKeyframes()",
    );
  }
}
