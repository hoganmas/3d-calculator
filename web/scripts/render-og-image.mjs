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
import { renderOgComposite } from "./og/renderShareOg.mjs";
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
  const png = await renderOgComposite({ siteUrl: previewUrl, scenes: DEFAULT_OG_SCENES, logoSvg });
  writeFileSync(outPath, png);
  console.log("Wrote public/og-image.png");
} catch (err) {
  if (preview.logs?.length) console.error(preview.logs.join(""));
  throw err;
} finally {
  preview.kill("SIGTERM");
}
