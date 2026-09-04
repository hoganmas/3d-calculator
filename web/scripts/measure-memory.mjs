#!/usr/bin/env node
/**
 * Measures JS heap / DOM node / event listener counts over a sustained
 * animation-playing session, to catch the kind of slow non-DOM,
 * non-listener leak that a quick manual check would miss.
 *
 * See docs/known-issues.md for the confirmed leak this is tracking. This
 * script does not assert pass/fail — the leak isn't fixed yet — it just
 * reports the trend so a fix can be verified against it later.
 *
 * Spawns its own dev server on a free port, so it doesn't require one
 * already running.
 *
 * Usage: npx tsx scripts/measure-memory.mjs [--quick] [--duration-sec 600] [--interval-sec 10]
 *   --quick   1-minute run (sanity check the harness itself works)
 */
import { spawn } from "node:child_process";
import { createConnection } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function parseArgs() {
  const quick = process.argv.includes("--quick");
  const arg = (name, fallback) => {
    const pref = `--${name}=`;
    const found = process.argv.find((a) => a.startsWith(pref));
    return found ? Number(found.slice(pref.length)) : fallback;
  };
  return {
    durationSec: arg("duration-sec", quick ? 60 : 600),
    intervalSec: arg("interval-sec", quick ? 10 : 10),
  };
}

async function pickPort() {
  for (let port = 5190; port < 5250; port++) {
    const free = await new Promise((resolve) => {
      const socket = createConnection({ port, host: "127.0.0.1" });
      socket.once("connect", () => {
        socket.destroy();
        resolve(false);
      });
      socket.once("error", () => resolve(true));
    });
    if (free) return port;
  }
  throw new Error("No free port found for the dev server");
}

function waitForServer(url, timeoutMs = 45_000) {
  const { hostname, port } = new URL(url);
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = createConnection({ host: hostname, port: Number(port) }, () => {
        req.end();
        resolve();
      });
      req.on("error", () => {
        if (Date.now() - start > timeoutMs) {
          reject(new Error(`Dev server did not start at ${url}`));
          return;
        }
        setTimeout(tick, 250);
      });
    };
    tick();
  });
}

function startDevServer(port) {
  const viteBin = join(root, "node_modules", "vite", "bin", "vite.js");
  const child = spawn(
    process.execPath,
    [viteBin, "--port", String(port), "--host", "127.0.0.1", "--strictPort"],
    { cwd: root, stdio: ["ignore", "pipe", "pipe"] },
  );
  const logs = [];
  child.stdout?.on("data", (b) => logs.push(b.toString()));
  child.stderr?.on("data", (b) => logs.push(b.toString()));
  child.logs = logs;
  return child;
}

function metricsToObj(metrics) {
  const o = {};
  for (const m of metrics) o[m.name] = m.value;
  return o;
}

function linearRegressionSlope(xs, ys) {
  const n = xs.length;
  const xMean = xs.reduce((a, b) => a + b, 0) / n;
  const yMean = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - xMean) * (ys[i] - yMean);
    den += (xs[i] - xMean) ** 2;
  }
  return den > 0 ? num / den : 0;
}

async function main() {
  const { durationSec, intervalSec } = parseArgs();
  const port = await pickPort();
  const url = `https://127.0.0.1:${port}/`;
  const server = startDevServer(port);
  let browser;

  try {
    await waitForServer(url);

    browser = await chromium.launch({ args: ["--js-flags=--expose-gc"] });
    const page = await browser.newPage({
      viewport: { width: 390, height: 844 }, // mobile-width layout, matching the reported repro
      ignoreHTTPSErrors: true, // vite dev serves a self-signed cert
      deviceScaleFactor: 2,
    });

    const pageErrors = [];
    page.on("pageerror", (e) => pageErrors.push(String(e)));
    page.on("console", (msg) => {
      if (msg.type() === "error") pageErrors.push(`[console] ${msg.text()}`);
    });

    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Performance.enable");

    await page.goto(url, { waitUntil: "load", timeout: 30_000 });
    await page.waitForTimeout(3000);

    // Default expression set already has an animating parameter; make sure
    // it's actually playing rather than assuming boot state.
    await page.evaluate(() => {
      const btn = document.querySelector(".expr-param-play");
      if (btn && btn.getAttribute("title") !== "Pause animation") btn.click();
    });

    async function forceGc() {
      try {
        await page.evaluate(() => {
          if (window.gc) {
            window.gc();
            window.gc();
          }
        });
      } catch {
        /* page mid-navigation; ignore */
      }
    }

    let resizeToggle = false;
    async function simulateUsage() {
      // Nudge viewport width (rotation/keyboard-open-ish resize) and cycle
      // the sidebar panel — exercises the resize + mount/unmount paths a
      // long real session hits repeatedly, on top of continuous animation.
      resizeToggle = !resizeToggle;
      await page.setViewportSize({ width: resizeToggle ? 390 : 400, height: 844 });
      const clickCollapse = () =>
        page.evaluate(() => {
          const btn = document.getElementById("collapsePanel") || document.getElementById("togglePanel");
          btn?.click();
        });
      await clickCollapse();
      await page.waitForTimeout(400);
      await clickCollapse();
    }

    const rows = [];
    const start = Date.now();

    async function sample() {
      await forceGc();
      const { metrics } = await cdp.send("Performance.getMetrics");
      const m = metricsToObj(metrics);
      const dom = await page.evaluate(() => ({
        exprRowCount: document.querySelectorAll(".expr-row").length,
        sliderVal: document.querySelector(".expr-param-slider")?.value ?? null,
      }));
      const row = {
        elapsedSec: Math.round((Date.now() - start) / 1000),
        jsHeapUsedMB: m.JSHeapUsedSize / 1e6,
        nodes: m.Nodes,
        jsEventListeners: m.JSEventListeners,
        ...dom,
      };
      rows.push(row);
      console.log(
        `t=${String(row.elapsedSec).padStart(4)}s  heap=${row.jsHeapUsedMB.toFixed(2)}MB  nodes=${row.nodes}  listeners=${row.jsEventListeners}  slider=${row.sliderVal}`,
      );
    }

    await sample();
    let lastUsageSim = Date.now();
    while (Date.now() - start < durationSec * 1000) {
      await page.waitForTimeout(intervalSec * 1000);
      if (Date.now() - lastUsageSim > 30_000) {
        await simulateUsage();
        lastUsageSim = Date.now();
      }
      await sample();
    }

    const heapSlopeMBPerMin = linearRegressionSlope(
      rows.map((r) => r.elapsedSec / 60),
      rows.map((r) => r.jsHeapUsedMB),
    );

    console.log("\n--- summary ---");
    console.log(`samples: ${rows.length}, duration: ${durationSec}s`);
    console.log(
      `heap: ${rows[0].jsHeapUsedMB.toFixed(2)}MB -> ${rows.at(-1).jsHeapUsedMB.toFixed(2)}MB` +
        ` (Δ ${(rows.at(-1).jsHeapUsedMB - rows[0].jsHeapUsedMB).toFixed(2)}MB)`,
    );
    console.log(`estimated heap growth rate: ${heapSlopeMBPerMin.toFixed(3)} MB/min`);
    console.log(`nodes: ${rows[0].nodes} -> ${rows.at(-1).nodes}`);
    console.log(`listeners: ${rows[0].jsEventListeners} -> ${rows.at(-1).jsEventListeners}`);
    if (pageErrors.length) {
      console.log(`\n${pageErrors.length} page error(s):`);
      for (const e of pageErrors.slice(0, 10)) console.log(" -", e);
    }
    console.log(
      heapSlopeMBPerMin > 0.3
        ? `\nHeap is growing ~${heapSlopeMBPerMin.toFixed(2)} MB/min with no plateau after forced GC — see docs/known-issues.md.`
        : "\nNo significant sustained heap growth detected in this run.",
    );
  } finally {
    server.kill();
    await browser?.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
