import chromium from "@sparticuz/chromium";
import { chromium as playwright } from "playwright-core";

export async function launchOgBrowser() {
  if (process.env.VERCEL) {
    const executablePath = await chromium.executablePath();
    if (executablePath) {
      const libDir = executablePath.slice(0, executablePath.lastIndexOf("/"));
      process.env.LD_LIBRARY_PATH = [libDir, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":");
    }
    return playwright.launch({
      args: chromium.args,
      executablePath,
      headless: chromium.headless,
    });
  }

  // --no-sandbox/--disable-dev-shm-usage: headless Chromium's sandboxed
  // renderer process commonly can't spawn inside a container/CI sandbox
  // (surfaces as "Failed to create browser context" on the second newPage).
  const sandboxArgs = ["--no-sandbox", "--disable-dev-shm-usage"];
  try {
    const { chromium: local } = await import("playwright");
    return local.launch({
      args: [...sandboxArgs, "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=metal"],
    });
  } catch {
    return playwright.launch({ headless: true, args: sandboxArgs });
  }
}

const HIDE_CHROME_CSS = `
  /* .mobile-expr-footer: multi-panel captures use per-cell viewports as
     narrow as ~400px (see renderShareOg.mjs), which trips the app's own
     (max-width: 800px) responsive breakpoint (panelLayout.ts) into showing
     its mobile expression carousel instead of the desktop #panel sidebar. */
  #panel, #panelResize, .viewport-toolbar, #hud, #kfLoadBar, #splash,
  .mobile-expr-footer { display: none !important; }
  html { --panel-progress: 1 !important; }
  #app { display: block !important; }
  #viewport {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
  }
  html, body { overflow: hidden !important; }
`;

// A random per-capture phase offset for the lava background's uTime
// (see web/src/render/background.ts), so repeated/similar shares don't get
// an identical-looking background. Injected here, not in app code, since
// it's purely a capture-tool concern — real interactive sessions should
// never see a jumped clock.
function randomTimeOffsetMs() {
  return Math.random() * 300_000;
}

export async function prepareCapturePage(browser, siteUrl, ogDeg) {
  // ignoreHTTPSErrors: local dev serves over HTTPS with a self-signed cert
  // (@vitejs/plugin-basic-ssl, so LAN/device testing works); a real deploy
  // has a valid cert, so this is a no-op there.
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, ignoreHTTPSErrors: true });
  await page.addInitScript((offsetMs) => {
    const realNow = performance.now.bind(performance);
    performance.now = () => realNow() + offsetMs;
  }, randomTimeOffsetMs());
  const q = new URLSearchParams({ ogCapture: "1" });
  if (ogDeg != null) q.set("ogDeg", String(ogDeg));
  await page.goto(`${siteUrl.replace(/\/$/, "")}/?${q}`, {
    waitUntil: "domcontentloaded",
    timeout: 90_000,
  });
  await page.waitForFunction(() => window.__laplacianOgCapture, { timeout: 60_000 });
  await page.waitForFunction(() => !document.getElementById("splash"), { timeout: 60_000 });
  await page.addStyleTag({ content: HIDE_CHROME_CSS });
  await page.evaluate(() => {
    document.documentElement.dataset.panelCollapsed = "true";
    window.__laplacianOgCapture?.resetCamera();
  });
  await page.waitForSelector("#viewport canvas", { timeout: 60_000 });
  return page;
}

export async function captureScene(page, scene) {
  // Capture at the panel's exact on-composite pixel size (see
  // renderShareOg.mjs) so composite.mjs can lay it in edge-to-edge with no
  // scaling/cropping — also lets the WebGPU canvas render at native res
  // for that slot instead of a fixed size that gets downscaled/cropped.
  if (scene.viewport) {
    await page.setViewportSize(scene.viewport);
    await page.evaluate(() => window.__laplacianOgCapture?.resetCamera());
  }
  await page.evaluate(async ({ latex, palette, paramRows, camera, settleMs }) => {
    const api = window.__laplacianOgCapture;
    if (!api) throw new Error("ogCapture API missing");
    await api.load(latex, { palette, paramRows });
    if (camera?.position) api.setCamera(camera.position, camera.target ?? [0, 0, 0]);
    else api.resetCamera();
    await api.waitFrame(settleMs ?? 2500);
  }, scene);
  const canvas = page.locator("#viewport canvas.clip-gpu").first();
  const target = (await canvas.count()) > 0 ? canvas : page.locator("#viewport canvas").first();
  return target.screenshot({ type: "png" });
}

/**
 * Capture the whole scene (every row loaded together, exactly as a viewer
 * would see it) in one shot — used for real per-share OG images instead of
 * isolating each expression into its own panel (see captureScene), so
 * multi-expression scenes render as the actual composition, not N unrelated
 * thumbnails, and cross-row dependencies (e.g. an animated parameter) just
 * work without needing to be carried along separately.
 */
export async function captureFullScene(page, { rows, camera, settleMs, isometric = true } = {}) {
  await page.evaluate(
    async ({ rows, camera, settleMs, isometric }) => {
      const api = window.__laplacianOgCapture;
      if (!api) throw new Error("ogCapture API missing");
      await api.loadExpressions(rows);
      if (camera?.position) api.setCamera(camera.position, camera.target ?? [0, 0, 0]);
      else api.resetCamera();
      api.setIsometric(isometric);
      await api.waitFrame(settleMs ?? 2500);
    },
    { rows, camera, settleMs, isometric },
  );
  const canvas = page.locator("#viewport canvas.clip-gpu").first();
  const target = (await canvas.count()) > 0 ? canvas : page.locator("#viewport canvas").first();
  return target.screenshot({ type: "png" });
}

export async function screenshotComposite(browser, html) {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
  await page.setContent(html, { waitUntil: "load" });
  return page.screenshot({ type: "png", clip: { x: 0, y: 0, width: 1200, height: 630 } });
}
