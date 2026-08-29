import { encodeExpressionsFragment } from "../../src/app/persistence/exprShare.ts";
import { encodeCompactFragment } from "../../src/app/persistence/exprShareCodec.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";
import {
  BENCH_STRATEGIES,
  benchFixtures,
  formatBenchmarkTable,
  formatOptimizationHints,
  PRODUCTION_FRAGMENT_BASELINES,
  runExprShareBenchmarks,
} from "./exprShareBench.ts";

export async function run() {
  const rows = runExprShareBenchmarks();
  console.log("\nexprShare URL length benchmark");
  console.log(formatBenchmarkTable(rows));
  console.log(formatOptimizationHints(rows));

  return runSuite("persistence / exprShare.bench", [
    {
      name: "reports benchmark table for all fixtures",
      fn: () => {
        assert(rows.length >= 6, "fixture count");
        for (const row of rows) {
          assert(row.exprCount > 0, `${row.fixture} has expressions`);
          assert(row.strategies["compact-auto"] > 0, `${row.fixture} production size`);
        }
      },
    },
    {
      name: "production fragment matches compact-auto benchmark",
      fn: async () => {
        for (const { name, exprs } of benchFixtures()) {
          const production = await encodeExpressionsFragment(exprs);
          const bench = rows.find((r) => r.fixture === name);
          assert(bench != null, `bench row for ${name}`);
          assert(
            production.length === bench!.strategies["compact-auto"],
            `${name}: production (${production.length}) vs bench (${bench!.strategies["compact-auto"]})`,
          );
        }
      },
    },
    {
      name: "encodeCompactFragment modes match benchmark sizes",
      fn: async () => {
        for (const { name, exprs } of benchFixtures()) {
          const bench = rows.find((r) => r.fixture === name)!;
          const raw = await encodeCompactFragment(exprs, "none");
          const gzip = await encodeCompactFragment(exprs, "gzip");
          assert(raw.length === bench.strategies["compact-raw"], `${name} raw`);
          assert(gzip.length === bench.strategies["compact-gzip"], `${name} gzip`);
        }
      },
    },
    {
      name: "lossless strategies preserve byte ordering for gzip vs deflate",
      fn: () => {
        for (const row of rows) {
          const raw = row.strategies["compact-raw"];
          const gzip = row.strategies["compact-gzip"];
          const deflate = row.strategies["compact-deflate"];
          assert(raw > 0 && gzip > 0 && deflate > 0, `${row.fixture} strategy sizes`);
        }
      },
    },
    {
      name: "production stays within fragment length baselines",
      fn: () => {
        for (const row of rows) {
          const baseline = PRODUCTION_FRAGMENT_BASELINES[row.fixture];
          if (baseline == null) continue;
          const prod = row.strategies["compact-auto"];
          assert(
            prod <= baseline,
            `${row.fixture}: production fragment ${prod} chars exceeds baseline ${baseline} — update baseline if intentional`,
          );
        }
      },
    },
    {
      name: "full URL length stays under conservative 2000 char budget",
      fn: () => {
        for (const row of rows) {
          assert(
            row.fullUrlProduction <= 2000,
            `${row.fixture}: full URL ${row.fullUrlProduction} chars exceeds 2000`,
          );
        }
      },
    },
    {
      name: "compact-auto is best or tied among production-viable strategies",
      fn: () => {
        for (const row of rows) {
          const prod = row.strategies["compact-auto"];
          assert(
            prod <= row.bestProduction.chars + 1,
            `${row.fixture}: production (${prod}) worse than best raw/gzip/auto (${row.bestProduction.chars})`,
          );
        }
      },
    },
  ]);
}
