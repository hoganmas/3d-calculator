import { buildCompositeHtml } from "./composite.mjs";
import {
  captureScene,
  launchOgBrowser,
  prepareCapturePage,
  screenshotComposite,
} from "./capture.mjs";

const DEFAULT_CAMERA = { position: [6.8, 6.2, 4.8], target: [0, 0, 0] };

/**
 * Render a share OG PNG for the given expression panels.
 * @param {{ siteUrl: string, panels: { latex: string, palette?: number, label: string }[], logoSvg: string, ogDeg?: number }} opts
 */
export async function renderShareOgPng({ siteUrl, panels, logoSvg, ogDeg = 16 }) {
  const browser = await launchOgBrowser();
  try {
    const page = await prepareCapturePage(browser, siteUrl, ogDeg);
    const captured = [];
    for (const panel of panels) {
      const png = await captureScene(page, {
        latex: panel.latex,
        palette: panel.palette ?? 0,
        camera: DEFAULT_CAMERA,
        settleMs: panel.latex.includes("\\left(") ? 3500 : 2500,
      });
      captured.push({ ...panel, png: png.toString("base64") });
    }
    const html = buildCompositeHtml(captured, logoSvg);
    return screenshotComposite(browser, html);
  } finally {
    await browser.close();
  }
}
