#!/usr/bin/env node
/**
 * Renders public/og-image.png (1200×630) with logo, copy, and scene captures.
 * Run from web/: npm run build && node scripts/render-og-image.mjs
 */
import { spawn } from "node:child_process";
import { createConnection } from "node:net";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildCompositeHtml } from "./og/composite.mjs";
import {
  captureScene,
  launchOgBrowser,
  prepareCapturePage,
  screenshotComposite,
} from "./og/capture.mjs";
import { DEFAULT_OG_SCENES } from "./og/defaultScenes.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outPath = join(root, "public/og-image.png");
const viteBin = join(root, "node_modules", "vite", "bin", "vite.js");

async function pickPort() {
  for (let port = 4174; port < 4200; port++) {
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
  throw new Error("No free preview port found");
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
          reject(new Error(`Preview server did not start at ${url}`));
          return;
        }
        setTimeout(tick, 250);
      });
    };
    tick();
  });
}

function startPreview(port) {
  const logs = [];
  const child = spawn(process.execPath, [viteBin, "preview", "--port", String(port), "--host", "127.0.0.1", "--strictPort"], {
    cwd: root,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout?.on("data", (buf) => logs.push(buf.toString()));
  child.stderr?.on("data", (buf) => logs.push(buf.toString()));
  child.logs = logs;
  return child;
}

const port = await pickPort();
const previewUrl = `http://127.0.0.1:${port}/`;
const preview = startPreview(port);
try {
  await waitForServer(previewUrl);
  const logoSvg = readFileSync(join(root, "public/logo.svg"), "utf8");
  const browser = await launchOgBrowser();
  try {
    const page = await prepareCapturePage(browser, previewUrl, 16);
    const captured = [];
    for (const scene of DEFAULT_OG_SCENES) {
      const png = await captureScene(page, scene);
      captured.push({ ...scene, png: png.toString("base64") });
      console.log(`Captured: ${scene.label}`);
    }
    const html = buildCompositeHtml(captured, logoSvg);
    const png = await screenshotComposite(browser, html);
    writeFileSync(outPath, png);
    console.log("Wrote public/og-image.png");
  } finally {
    await browser.close();
  }
} catch (err) {
  if (preview.logs?.length) console.error(preview.logs.join(""));
  throw err;
} finally {
  preview.kill("SIGTERM");
}
