/**
 * Lobatto vs Gauss Chebyshev accuracy benchmark at high degrees.
 * Usage: npx tsx scripts/bench-lobatto-accuracy.mjs [--deg 8,16,32] [--quick]
 */
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compileExpr, fitChebyshev3D } from "../src/math/fit.ts";
import {
  fitChebyshevLobatto3D,
  fitChebyshevLobattoProgressive,
  gaussWorld,
  idctLobatto3D,
  lobattoWorld,
  probeRelL2Cheb,
  probeRelL2Lobatto,
} from "../src/math/chebLobatto.ts";
import { idctCheb3D } from "../src/math/idct.ts";

const root = dirname(fileURLToPath(import.meta.url));
const outPath = join(root, "../../research/poly/results/lobatto_accuracy_benchmark.json");

const CASES = [
  { id: "quadratic", latex: "x^2+y^2+z^2" },
  { id: "blob", latex: String.raw`\exp(-(x^2+y^2+z^2))` },
  { id: "two-blobs", latex: String.raw`\exp(-4((x-0.7)^2+y^2+z^2))+\exp(-4((x+0.7)^2+y^2+z^2))` },
  { id: "sincos", latex: String.raw`\sin(x)\cos(z)` },
  { id: "shell", latex: String.raw`\exp(-12(r-0.9)^2)` },
  { id: "soft-ellipsoid", latex: String.raw`\exp(-(x^2+0.5y^2+2z^2))` },
];

const DEFAULT_DEGS = [8, 12, 16, 24, 32, 48, 64];
const QUICK_DEGS = [8, 16, 32];

function parseArgs() {
  const quick = process.argv.includes("--quick");
  const degArg = process.argv.find((a) => a.startsWith("--deg="));
  const degs = degArg
    ? degArg
        .slice("--deg=".length)
        .split(",")
        .map((s) => Number(s.trim()))
        .filter((n) => n > 0)
    : quick
      ? QUICK_DEGS
      : DEFAULT_DEGS;
  return { degs, probes: quick ? 8 : 12 };
}

function densIndex(ix, iy, iz, M) {
  return ix + iy * M + iz * M * M;
}

function gridMaxErrGauss(cheb, deg, half, fn) {
  const { dens, M } = idctCheb3D(cheb, deg, deg + 1);
  let maxErr = 0;
  for (let ix = 0; ix < M; ix++) {
    const x = gaussWorld(ix, deg, half);
    for (let iy = 0; iy < M; iy++) {
      const y = gaussWorld(iy, deg, half);
      for (let iz = 0; iz < M; iz++) {
        const z = gaussWorld(iz, deg, half);
        maxErr = Math.max(maxErr, Math.abs((dens[densIndex(ix, iy, iz, M)] ?? 0) - fn(x, y, z)));
      }
    }
  }
  return maxErr;
}

function gridMaxErrLobatto(cheb, deg, half, fn) {
  const { dens, M } = idctLobatto3D(cheb, deg, deg + 1);
  let maxErr = 0;
  for (let ix = 0; ix < M; ix++) {
    const x = lobattoWorld(ix, deg, half);
    for (let iy = 0; iy < M; iy++) {
      const y = lobattoWorld(iy, deg, half);
      for (let iz = 0; iz < M; iz++) {
        const z = lobattoWorld(iz, deg, half);
        maxErr = Math.max(maxErr, Math.abs((dens[densIndex(ix, iy, iz, M)] ?? 0) - fn(x, y, z)));
      }
    }
  }
  return maxErr;
}

function benchCase(caseDef, deg, half, probes) {
  const fn = compileExpr(caseDef.latex).bind({});

  const t0 = performance.now();
  const gauss = fitChebyshev3D(fn, half, deg, { skipMono: true, skipL2: true });
  const gaussMs = performance.now() - t0;

  const t1 = performance.now();
  const lobatto = fitChebyshevLobatto3D(fn, half, deg, { skipL2: true });
  const lobattoMs = performance.now() - t1;

  let progSamples = 0;
  const t2 = performance.now();
  fitChebyshevLobattoProgressive(fn, half, deg, (s) => {
    progSamples += s.newSamples;
  });
  const progMs = performance.now() - t2;
  const fullSamples = (deg + 1) ** 3;

  return {
    case: caseDef.id,
    deg,
    gauss: {
      sampleMs: gauss.timing.sampleMs,
      chebMs: gauss.timing.chebMs,
      totalMs: gaussMs,
      gridMaxErr: gridMaxErrGauss(gauss.cheb, deg, half, fn),
      probeRelL2: probeRelL2Cheb(gauss.cheb, deg, half, fn, probes),
    },
    lobatto: {
      sampleMs: lobatto.timing.sampleMs,
      chebMs: lobatto.timing.chebMs,
      totalMs: lobattoMs,
      gridMaxErr: gridMaxErrLobatto(lobatto.cheb, deg, half, fn),
      probeRelL2: probeRelL2Lobatto(lobatto.cheb, deg, half, fn, probes),
    },
    progressive: {
      totalMs: progMs,
      samplesEvaluated: progSamples,
      fullSamples,
      sampleSavingsPct: Math.round((1 - progSamples / fullSamples) * 1000) / 10,
    },
    ratio: {
      probeRelL2:
        probeRelL2Lobatto(lobatto.cheb, deg, half, fn, probes) /
        (probeRelL2Cheb(gauss.cheb, deg, half, fn, probes) + 1e-15),
    },
  };
}

const { degs, probes } = parseArgs();
const half = 1;
const results = [];
const started = new Date().toISOString();

console.log(`Lobatto vs Gauss accuracy · half=${half} · degs=[${degs.join(", ")}] · probes=${probes}³\n`);
console.log(
  "case".padEnd(14),
  "deg".padStart(4),
  "gauss L2".padStart(10),
  "lob L2".padStart(10),
  "ratio".padStart(8),
  "gauss grid".padStart(11),
  "lob grid".padStart(11),
  "prog save".padStart(10),
);

for (const caseDef of CASES) {
  for (const deg of degs) {
    process.stdout.write(`  ${caseDef.id} deg=${deg}...`);
    const row = benchCase(caseDef, deg, half, probes);
    results.push(row);
    console.log(
      `\r${row.case.padEnd(14)} ${String(row.deg).padStart(4)} ${row.gauss.probeRelL2.toExponential(2).padStart(10)} ${row.lobatto.probeRelL2.toExponential(2).padStart(10)} ${row.ratio.probeRelL2.toFixed(2).padStart(8)} ${row.gauss.gridMaxErr.toExponential(2).padStart(11)} ${row.lobatto.gridMaxErr.toExponential(2).padStart(11)} ${(row.progressive.sampleSavingsPct + "%").padStart(10)}`,
    );
  }
}

const payload = {
  started,
  finished: new Date().toISOString(),
  half,
  degs,
  probes,
  cases: CASES.map((c) => c.id),
  results,
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(payload, null, 2));
console.log(`\nWrote ${outPath}`);
