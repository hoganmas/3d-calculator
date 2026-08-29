/** Boot / first-fit timing. Enable via ?startupProfile=1 or auto during splash (data-booting). */

export interface StartupMark {
  name: string;
  t: number;
  ms: number;
  durationMs?: number;
  detail?: Record<string, unknown>;
}

const origin = typeof performance !== "undefined" ? performance.now() : 0;
const marks: StartupMark[] = [];
const openSpans = new Map<string, number>();
let reported = false;

let enabled: boolean | null = null;

function computeEnabled(): boolean {
  if (typeof window === "undefined") return false;
  if (document.documentElement.hasAttribute("data-booting")) return true;
  try {
    return (
      new URLSearchParams(window.location.search).has("startupProfile") ||
      localStorage.getItem("laplacianStartupProfile") === "1" ||
      (window as Window & { __laplacianStartupProfile?: boolean }).__laplacianStartupProfile === true
    );
  } catch {
    return false;
  }
}

export function isStartupProfilingActive(): boolean {
  if (enabled === null) enabled = computeEnabled();
  return enabled;
}

export function setStartupProfilingEnabled(on: boolean): void {
  enabled = on;
  if (typeof window === "undefined") return;
  if (on) localStorage.setItem("laplacianStartupProfile", "1");
  else localStorage.removeItem("laplacianStartupProfile");
}

function pushMark(name: string, detail?: Record<string, unknown>, durationMs?: number): void {
  if (!isStartupProfilingActive()) return;
  marks.push({
    name,
    t: performance.now(),
    ms: performance.now() - origin,
    durationMs,
    detail,
  });
}

/** Instant milestone (time since boot origin). */
export function startupMark(name: string, detail?: Record<string, unknown> = {}): void {
  pushMark(name, detail);
}

/** Start a timed span; pair with startupEnd. */
export function startupBegin(name: string): void {
  if (!isStartupProfilingActive()) return;
  openSpans.set(name, performance.now());
}

/** End a timed span started with startupBegin. */
export function startupEnd(name: string, detail?: Record<string, unknown> = {}): void {
  if (!isStartupProfilingActive()) return;
  const tStart = openSpans.get(name);
  if (tStart == null) return;
  openSpans.delete(name);
  pushMark(name, detail, performance.now() - tStart);
}

export function getStartupProfileSnapshot() {
  const sorted = marks.slice().sort((a, b) => a.t - b.t);
  const gaps: { after: string; before: string; gapMs: number }[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1]!;
    const cur = sorted[i]!;
    gaps.push({
      after: prev.name,
      before: cur.name,
      gapMs: Math.round((cur.t - prev.t) * 10) / 10,
    });
  }
  const open = [...openSpans.keys()];
  return {
    originMs: 0,
    markCount: sorted.length,
    marks: sorted,
    gaps: gaps.filter((g) => g.gapMs >= 1),
    openSpans: open,
    reported,
  };
}

function formatDetail(detail?: Record<string, unknown>): string {
  if (!detail || !Object.keys(detail).length) return "";
  const parts: string[] = [];
  for (const [k, v] of Object.entries(detail)) {
    if (v == null) continue;
    parts.push(`${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`);
  }
  return parts.length ? ` · ${parts.join(" · ")}` : "";
}

/** Print a human-readable boot timeline (once unless force). */
export function startupReport(reason = "manual"): void {
  if (!isStartupProfilingActive()) return;
  if (reported && reason !== "manual") return;
  if (reason !== "manual") reported = true;

  const snap = getStartupProfileSnapshot();
  const sorted = snap.marks;
  if (!sorted.length) {
    console.log("[startup] (no marks recorded)");
    return;
  }

  const lines: string[] = [`[startup] profile · ${reason}`];
  for (const m of sorted) {
    const dur = m.durationMs != null ? ` (${m.durationMs.toFixed(1)}ms)` : "";
    lines.push(`  +${m.ms.toFixed(0)}ms  ${m.name}${dur}${formatDetail(m.detail)}`);
  }

  const bigGaps = snap.gaps.filter((g) => g.gapMs >= 50);
  if (bigGaps.length) {
    lines.push("[startup] gaps ≥50ms:");
    for (const g of bigGaps) {
      lines.push(`  ${g.gapMs.toFixed(0)}ms  ${g.after} → ${g.before}`);
    }
  }

  if (snap.openSpans.length) {
    lines.push(`[startup] open spans: ${snap.openSpans.join(", ")}`);
  }

  console.log(lines.join("\n"));
}

export function initStartupProfile(): void {
  if (typeof window === "undefined") return;
  if (enabled === null) enabled = computeEnabled();
  startupMark("boot.module-loaded");
  const w = window as Window & {
    __laplacianStartup?: () => ReturnType<typeof getStartupProfileSnapshot>;
    __laplacianStartupReport?: () => void;
    __laplacianStartupProfile?: boolean;
  };
  w.__laplacianStartup = getStartupProfileSnapshot;
  w.__laplacianStartupReport = () => startupReport("manual");
  if (isStartupProfilingActive()) {
    console.log(
      "[startup] profiling ON — timeline on splash dismiss; copy via JSON.stringify(window.__laplacianStartup())",
    );
  }
}
