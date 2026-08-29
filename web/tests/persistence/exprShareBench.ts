/**
 * URL-length benchmarks for expression fragment encodings.
 * Used by exprShare.bench.test.ts — not imported by production code.
 */
import { deflateSync, gzipSync } from "node:zlib";
import { PRESETS } from "../../src/math/fit.ts";
import {
  bytesToBase64Url,
  compactExprPayload,
  EXPR_SHARE_VERSION,
  FRAGMENT_PREFIX,
  GZIP_THRESHOLD,
  pickCompressMode,
  type CompactExprRow,
} from "../../src/app/persistence/exprShareCodec.ts";
import type { ExprItem } from "../../src/types/models.ts";

const SAMPLE_ORIGIN = "https://hoganmas.github.io/3d-calculator/";

export type BenchStrategyId =
  | "compact-raw"
  | "compact-gzip"
  | "compact-deflate"
  | "compact-auto"
  | "latex-json-raw"
  | "latex-json-gzip"
  | "latex-lines-gzip"
  | "full-json-gzip";

export type BenchStrategy = {
  id: BenchStrategyId;
  label: string;
  lossy: boolean;
  note?: string;
};

export const BENCH_STRATEGIES: BenchStrategy[] = [
  { id: "compact-raw", label: "compact/raw", lossy: false },
  { id: "compact-gzip", label: "compact/gzip", lossy: false },
  { id: "compact-deflate", label: "compact/deflate", lossy: false },
  { id: "compact-auto", label: "compact/auto", lossy: false, note: "production" },
  { id: "latex-json-raw", label: "latex[]/raw", lossy: true, note: "latex only" },
  { id: "latex-json-gzip", label: "latex[]/gzip", lossy: true, note: "latex only" },
  { id: "latex-lines-gzip", label: "latex\\n/gzip", lossy: true, note: "latex only" },
  { id: "full-json-gzip", label: "full/gzip", lossy: false, note: "verbose JSON" },
];

/** Upper bounds for production (compact-auto) fragment length — update when optimizing. */
export const PRODUCTION_FRAGMENT_BASELINES: Record<string, number> = {
  blob: 110,
  sincos: 260,
  lavalamp: 320,
  swirl: 90,
  "12-clouds": 380,
  "20-long": 1450,
};

export function benchExpr(overrides: Partial<ExprItem> = {}): ExprItem {
  return {
    id: "e1",
    latex: String.raw`\exp(-r^2)`,
    color: "#ff4500",
    color2: "#ffec00",
    colors: ["#ff4500", "#ffec00"],
    enabled: true,
    sliderMin: -10,
    sliderMax: 10,
    sliderSpeed: 0.35,
    sliderAnimating: false,
    sliderPhase: 0.2,
    sliderAnimMode: "pingpong",
    autoParam: false,
    ...overrides,
  };
}

function fullSerializeRow(item: ExprItem) {
  return {
    id: item.id,
    latex: item.latex,
    enabled: item.enabled,
    color: item.color,
    color2: item.color2,
    colors: item.colors?.slice(),
    sliderMin: item.sliderMin,
    sliderMax: item.sliderMax,
    sliderSpeed: item.sliderSpeed,
    sliderAnimating: item.sliderAnimating,
    sliderPhase: item.sliderPhase,
    sliderAnimMode: item.sliderAnimMode,
    autoParam: item.autoParam,
  };
}

function encodeNodeFragment(jsonBytes: Uint8Array, mode: "none" | "gzip" | "deflate"): string {
  const body =
    mode === "none" ? jsonBytes : Uint8Array.from(mode === "gzip" ? gzipSync(jsonBytes) : deflateSync(jsonBytes));
  const flag = mode === "gzip" ? "z" : mode === "deflate" ? "d" : "";
  return `${FRAGMENT_PREFIX}${EXPR_SHARE_VERSION}${flag}.${bytesToBase64Url(body)}`;
}

function latexOnlyPayload(exprs: ExprItem[]): string[] {
  return compactExprPayload(exprs).map((row) => row.l);
}

export function buildFixture(name: string, exprs: Partial<ExprItem>[]): { name: string; exprs: ExprItem[] } {
  return {
    name,
    exprs: exprs.map((row, i) => benchExpr({ id: `e${i + 1}`, ...row })),
  };
}

const LONG_LATEX = String.raw`\int_{0}^{1}\int_{0}^{1}\int_{0}^{1} \exp\left(-\left((x-a t)^2+(y-b t)^2+(z-c t)^2\right)\right)\,dx\,dy\,dz + \curl\left(\sin(x t),\cos(y t),\sin(z t)\right)\cdot\grad\left(x^2+y^2+z^2\right)`;

export function benchFixtures(): { name: string; exprs: ExprItem[] }[] {
  const fromPreset = (name: string) => {
    const preset = PRESETS[name];
    if (!preset) throw new Error(`missing preset: ${name}`);
    const rows = preset.expressions?.length
      ? preset.expressions
      : preset.latex
        ? [{ latex: preset.latex }]
        : [];
    return buildFixture(name, rows);
  };

  return [
    fromPreset("blob"),
    fromPreset("sincos"),
    fromPreset("lavalamp"),
    fromPreset("swirl"),
    buildFixture(
      "12-clouds",
      Array.from({ length: 12 }, (_, i) => ({
        latex: String.raw`\exp(-${i}((x-a)^2+(y-b)^2+(z-${i})^2))`,
      })),
    ),
    buildFixture(
      "20-long",
      Array.from({ length: 20 }, (_, i) => ({
        latex: LONG_LATEX.replace(/t/g, String(i)),
        color: "#ff4500",
        color2: "#ffec00",
        sliderAnimating: i % 3 === 0,
        sliderMin: 0,
        sliderMax: 1,
        sliderSpeed: 0.1,
        sliderAnimMode: "loop" as const,
        sliderPhase: i * 0.05,
      })),
    ),
  ];
}

export type BenchRow = {
  fixture: string;
  exprCount: number;
  jsonBytes: number;
  strategies: Record<BenchStrategyId, number>;
  fullUrlProduction: number;
  bestLossless: { id: BenchStrategyId; chars: number };
  bestProduction: { id: BenchStrategyId; chars: number };
  bestLossy: { id: BenchStrategyId; chars: number };
  productionGap: number;
  deflateSavings: number;
};

function measureStrategy(
  strategy: BenchStrategyId,
  exprs: ExprItem[],
  compact: CompactExprRow[],
  jsonBytes: Uint8Array,
): number {
  switch (strategy) {
    case "compact-raw":
      return encodeNodeFragment(jsonBytes, "none").length;
    case "compact-gzip":
      return encodeNodeFragment(jsonBytes, "gzip").length;
    case "compact-deflate":
      return encodeNodeFragment(jsonBytes, "deflate").length;
    case "compact-auto": {
      const mode = pickCompressMode(jsonBytes.length, "auto");
      return encodeNodeFragment(jsonBytes, mode).length;
    }
    case "latex-json-raw": {
      const payload = JSON.stringify(latexOnlyPayload(exprs));
      return encodeNodeFragment(new TextEncoder().encode(payload), "none").length;
    }
    case "latex-json-gzip": {
      const payload = JSON.stringify(latexOnlyPayload(exprs));
      return encodeNodeFragment(new TextEncoder().encode(payload), "gzip").length;
    }
    case "latex-lines-gzip": {
      const payload = latexOnlyPayload(exprs).join("\n");
      return encodeNodeFragment(new TextEncoder().encode(payload), "gzip").length;
    }
    case "full-json-gzip": {
      const payload = JSON.stringify(exprs.map(fullSerializeRow));
      return encodeNodeFragment(new TextEncoder().encode(payload), "gzip").length;
    }
  }
}

export function measureFixture(name: string, exprs: ExprItem[]): BenchRow {
  const compact = compactExprPayload(exprs);
  const jsonBytes = new TextEncoder().encode(JSON.stringify(compact));
  const strategies = {} as Record<BenchStrategyId, number>;
  for (const s of BENCH_STRATEGIES) {
    strategies[s.id] = measureStrategy(s.id, exprs, compact, jsonBytes);
  }

  const lossless = BENCH_STRATEGIES.filter((s) => !s.lossy).map((s) => ({
    id: s.id,
    chars: strategies[s.id],
  }));
  const productionCandidates = lossless.filter((s) => s.id !== "compact-deflate" && s.id !== "full-json-gzip");
  const lossy = BENCH_STRATEGIES.filter((s) => s.lossy).map((s) => ({
    id: s.id,
    chars: strategies[s.id],
  }));
  lossless.sort((a, b) => a.chars - b.chars);
  productionCandidates.sort((a, b) => a.chars - b.chars);
  lossy.sort((a, b) => a.chars - b.chars);

  const production = strategies["compact-auto"];
  return {
    fixture: name,
    exprCount: compact.length,
    jsonBytes: jsonBytes.length,
    strategies,
    fullUrlProduction: SAMPLE_ORIGIN.length + production,
    bestLossless: lossless[0]!,
    bestProduction: productionCandidates[0]!,
    bestLossy: lossy[0]!,
    productionGap: production - productionCandidates[0]!.chars,
    deflateSavings: production - strategies["compact-deflate"],
  };
}

export function runExprShareBenchmarks(): BenchRow[] {
  return benchFixtures().map(({ name, exprs }) => measureFixture(name, exprs));
}

function pad(value: string | number, width: number) {
  return String(value).padStart(width);
}

export function formatBenchmarkTable(rows: BenchRow[]): string {
  const headers = ["fixture", "n", "jsonB", ...BENCH_STRATEGIES.map((s) => s.label), "url", "gap"];
  const widths = [12, 3, 6, ...BENCH_STRATEGIES.map(() => 14), 6, 4];
  const lines = [
    headers.map((h, i) => pad(h, widths[i]!)).join(" | "),
    widths.map((w) => "-".repeat(w)).join("-+-"),
  ];

  for (const row of rows) {
    lines.push(
      [
        pad(row.fixture, widths[0]!),
        pad(row.exprCount, widths[1]!),
        pad(row.jsonBytes, widths[2]!),
        ...BENCH_STRATEGIES.map((s, i) => pad(row.strategies[s.id], widths[3 + i]!)),
        pad(row.fullUrlProduction, widths[widths.length - 2]!),
        pad(row.productionGap, widths[widths.length - 1]!),
      ].join(" | "),
    );
  }

  lines.push("");
  lines.push(`origin: ${SAMPLE_ORIGIN}`);
  lines.push(`gzip threshold (compact-auto): ${GZIP_THRESHOLD} json bytes`);
  lines.push("gap = production fragment chars minus shortest raw/gzip/auto strategy");
  lines.push("deflate column shows potential savings if deflate encoding is added");
  return lines.join("\n");
}

export function formatOptimizationHints(rows: BenchRow[]): string {
  const hints: string[] = ["optimization hints:"];
  for (const row of rows) {
    if (row.productionGap > 0) {
      hints.push(
        `  ${row.fixture}: production could save ${row.productionGap} chars with ${row.bestProduction.id} (${row.strategies["compact-auto"]} → ${row.bestProduction.chars})`,
      );
    }
    if (row.deflateSavings > 0) {
      hints.push(
        `  ${row.fixture}: deflate would save ${row.deflateSavings} chars vs production (${row.strategies["compact-deflate"]} vs ${row.strategies["compact-auto"]})`,
      );
    }
    const prod = row.strategies["compact-auto"];
    const lossy = row.bestLossy.chars;
    if (lossy + 20 < prod) {
      hints.push(
        `  ${row.fixture}: lossy ${row.bestLossy.id} is ${prod - lossy} chars shorter (${lossy} vs ${prod})`,
      );
    }
  }
  if (hints.length === 1) hints.push("  production already matches best raw/gzip/auto strategy on all fixtures");
  return hints.join("\n");
}
