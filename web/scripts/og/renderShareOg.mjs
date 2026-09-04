import { buildCompositeHtml, buildSingleShotHtml } from "./composite.mjs";
import {
  captureFullScene,
  captureScene,
  launchOgBrowser,
  prepareCapturePage,
  screenshotComposite,
} from "./capture.mjs";

const DEFAULT_CAMERA = { position: [6.8, 6.2, 4.8], target: [0, 0, 0] };

// Same view direction as DEFAULT_CAMERA, but closer (bigger subject in
// frame — overflow past the edges is fine) and panned right so the scene
// sits clear of the now-large, vertically-centered logo overlay. Position
// and target are shifted by the same world-space delta (the camera's local
// "left" direction) — a pure lateral pan, not a rotation.
const SHARE_CAMERA = { position: [6.65, 2.54, 3.46], target: [1.75, -1.92, 0] };

// Must match composite.mjs's canvas size — the single-shot capture renders
// at exactly this size natively, no client-side scaling/cropping.
const COMPOSITE_W = 1200;
const COMPOSITE_H = 630;
const PANEL_GAP = 3;

/**
 * Render an OG composite PNG for a set of expression scenes — shared by the
 * dynamic per-share render (renderShareOgPng, below) and the static
 * build-time public/og-image.png generator (render-og-image.mjs), so both
 * stay in sync on layout/viewport sizing.
 * `ogDeg` is an explicit fit-degree override for local testing/fast
 * iteration — omit it (the default) to render at forced max quality, which
 * is what ogCapture.ts's installOgCapture() does regardless.
 * @param {{ siteUrl: string, scenes: { latex: string, palette?: number, label: string, paramRows?: object[], camera?: object, settleMs?: number }[], logoSvg: string, ogDeg?: number }} opts
 */
export async function renderOgComposite({ siteUrl, scenes, logoSvg, ogDeg }) {
  const browser = await launchOgBrowser();
  try {
    const page = await prepareCapturePage(browser, siteUrl, ogDeg);
    const cols = Math.min(3, Math.max(1, scenes.length));
    const cellW = Math.round((COMPOSITE_W - (cols - 1) * PANEL_GAP) / cols);
    const captured = [];
    for (const scene of scenes) {
      const png = await captureScene(page, {
        latex: scene.latex,
        palette: scene.palette ?? 0,
        paramRows: scene.paramRows,
        camera: scene.camera ?? DEFAULT_CAMERA,
        settleMs: scene.settleMs ?? (scene.latex.includes("\\left(") ? 3500 : 2500),
        viewport: { width: cellW, height: COMPOSITE_H },
      });
      captured.push({ ...scene, png: png.toString("base64") });
    }
    // Close the capture page before compositing: the app's render loop
    // (requestAnimationFrame, WebGPU/WebGL) keeps running on it otherwise,
    // and that ongoing load can starve the browser during the composite
    // screenshot for no reason once we no longer need this page.
    await page.close();
    const html = buildCompositeHtml(captured, logoSvg);
    // The `await` here is load-bearing, not redundant: without it, `return`
    // hands back the pending promise and lets the `finally` block's
    // `browser.close()` run immediately (before compositing finishes) —
    // closing the browser out from under the still-in-flight screenshot.
    return await screenshotComposite(browser, html);
  } finally {
    await browser.close();
  }
}

/**
 * Render a share OG PNG for a decoded share payload's full row set — every
 * row loaded together in one scene, exactly as a viewer opening the link
 * would see it (not split into isolated per-expression panels; see
 * renderOgComposite for that, used only by the static marketing gallery).
 * `ogDeg` is an explicit fit-degree override for local testing/fast
 * iteration — omit it (the default) to render at forced max quality, which
 * is what ogCapture.ts's installOgCapture() does regardless.
 * @param {{ siteUrl: string, rows: object[], logoSvg: string, ogDeg?: number }} opts
 */
export async function renderShareOgPng({ siteUrl, rows, logoSvg, ogDeg }) {
  const browser = await launchOgBrowser();
  try {
    const page = await prepareCapturePage(browser, siteUrl, ogDeg);
    const png = await captureFullScene(page, {
      rows,
      camera: SHARE_CAMERA,
      settleMs: 3500,
    });
    await page.close();
    const html = buildSingleShotHtml(png.toString("base64"), logoSvg);
    return await screenshotComposite(browser, html);
  } finally {
    await browser.close();
  }
}
