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

  try {
    const { chromium: local } = await import("playwright");
    return local.launch({
      args: ["--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=metal"],
    });
  } catch {
    return playwright.launch({ headless: true });
  }
}

const HIDE_CHROME_CSS = `
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
`;

export async function prepareCapturePage(browser, siteUrl, ogDeg = 16) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const q = new URLSearchParams({ ogCapture: "1", ogDeg: String(ogDeg) });
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
  await page.evaluate(async ({ latex, palette, camera, settleMs }) => {
    const api = window.__laplacianOgCapture;
    if (!api) throw new Error("ogCapture API missing");
    await api.load(latex, { palette });
    if (camera?.position) api.setCamera(camera.position, camera.target ?? [0, 0, 0]);
    else api.resetCamera();
    await api.waitFrame(settleMs ?? 2500);
  }, scene);
  const canvas = page.locator("#viewport canvas.clip-gpu").first();
  const target = (await canvas.count()) > 0 ? canvas : page.locator("#viewport canvas").first();
  return target.screenshot({ type: "png" });
}

export async function screenshotComposite(browser, html) {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
  await page.setContent(html, { waitUntil: "load" });
  return page.screenshot({ type: "png", clip: { x: 0, y: 0, width: 1200, height: 630 } });
}
