#!/usr/bin/env node
/**
 * Renders public/og-image.png (1200×630) with logo, copy, and two scene captures.
 * Run from web/: npm run build && node scripts/render-og-image.mjs
 */
import { spawn } from "node:child_process";
import { createConnection } from "node:net";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outPath = join(root, "public/og-image.png");
const viteBin = join(root, "node_modules", "vite", "bin", "vite.js");

const SCENES = [
  {
    latex: String.raw`z=-\cos\left(x\right)\sin\left(2y\right)`,
    label: "z = −cos(x) sin(2y)",
    palette: 0,
    camera: { position: [6.8, 6.2, 4.8], target: [0, 0, 0] },
    settleMs: 2000,
  },
  {
    latex: String.raw`\left(0,z,-y\right)`,
    label: "(0, z, −y)",
    palette: 2,
    camera: { position: [7.2, 1.2, 5.8], target: [0, 0, 0] },
    settleMs: 3500,
  },
  {
    latex: String.raw`e^{-2.5r}\abs\left(2z^{2}-x^{2}-y^{2}\right)`,
    label: "e^{−2.5r}|2z² − x² − y²|",
    palette: 1,
    camera: { position: [6.4, 5.8, 5.2], target: [0, 0, 0] },
    settleMs: 3000,
  },
];

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

async function preparePage(browser, previewUrl) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto(`${previewUrl}?ogCapture=1`, { waitUntil: "domcontentloaded", timeout: 90_000 });
  await page.waitForFunction(() => window.__laplacianOgCapture, { timeout: 60_000 });
  await page.waitForFunction(() => !document.getElementById("splash"), { timeout: 60_000 });
  await page.addStyleTag({
    content: `
      #panel, #panelResize, .viewport-toolbar, #hud, #kfLoadBar, #splash { display: none !important; }
      html { --panel-progress: 1 !important; }
      #app { display: block !important; }
      #viewport {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
      }
      html, body { overflow: hidden !important; }
    `,
  });
  await page.evaluate(() => {
    document.documentElement.dataset.panelCollapsed = "true";
    window.__laplacianOgCapture?.resetCamera();
  });
  await page.waitForSelector("#viewport canvas", { timeout: 60_000 });
  return page;
}

async function captureScene(page, scene) {
  await page.evaluate(async ({ latex, palette, camera, settleMs }) => {
    const api = window.__laplacianOgCapture;
    if (!api) throw new Error("ogCapture API missing");
    await api.load(latex, { palette });
    api.setCamera(camera.position, camera.target);
    await api.waitFrame(settleMs);
  }, scene);
  const canvas = page.locator("#viewport canvas.clip-gpu").first();
  const target = (await canvas.count()) > 0 ? canvas : page.locator("#viewport canvas").first();
  return target.screenshot({ type: "png" });
}

function buildCompositeHtml(scenes, logoSvg) {
  const logoData = `data:image/svg+xml;base64,${Buffer.from(logoSvg).toString("base64")}`;
  const cards = scenes
    .map(
      (scene) => `
    <figure class="card">
      <div class="shot"><img src="data:image/png;base64,${scene.png}" alt="" /></div>
      <figcaption>${scene.label}</figcaption>
    </figure>`,
    )
    .join("");

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: 1200px; height: 630px; overflow: hidden;
    background: #1a1228;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: #f5eef8;
  }
  .wrap {
    width: 100%; height: 100%;
    display: flex;
    flex-direction: column;
    gap: 28px;
    padding: 36px 44px 32px;
  }
  .brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    text-align: center;
  }
  .logo-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 22px;
  }
  .logo-row img { width: 88px; height: 88px; flex: 0 0 auto; }
  .logo-row h1 {
    font-size: 72px;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
  }
  .tagline {
    font-size: 30px;
    color: #c9b8d9;
    line-height: 1.2;
    letter-spacing: -0.01em;
  }
  .shots {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    align-items: stretch;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
    min-height: 0;
  }
  .shot {
    flex: 1;
    min-height: 0;
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid #3d2d52;
    background: #0f0a18;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .shot img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    object-position: center;
    display: block;
  }
  figcaption {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    color: #9a86b0;
    text-align: center;
    line-height: 1.3;
    flex: 0 0 auto;
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="brand">
      <div class="logo-row">
        <img src="${logoData}" alt="" />
        <h1>laplaci</h1>
      </div>
      <p class="tagline">3D Graphing Calculator</p>
    </header>
    <div class="shots">${cards}</div>
  </div>
</body></html>`;
}

const port = await pickPort();
const previewUrl = `http://127.0.0.1:${port}/`;
const preview = startPreview(port);
try {
  await waitForServer(previewUrl);
  const logoSvg = readFileSync(join(root, "public/logo.svg"), "utf8");
  const browser = await chromium.launch({
    args: ["--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=metal"],
  });
  try {
    const page = await preparePage(browser, previewUrl);
    const captured = [];
    for (const scene of SCENES) {
      const png = await captureScene(page, scene);
      captured.push({ ...scene, png: png.toString("base64") });
      console.log(`Captured: ${scene.label}`);
    }
    const compositePage = await browser.newPage({ viewport: { width: 1200, height: 630 } });
    await compositePage.setContent(buildCompositeHtml(captured, logoSvg), { waitUntil: "load" });
    await compositePage.screenshot({ path: outPath, type: "png", clip: { x: 0, y: 0, width: 1200, height: 630 } });
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
