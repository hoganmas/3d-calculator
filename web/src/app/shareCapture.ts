/**
 * Client-side OG image capture, run at share-creation time.
 *
 * Rather than spinning up headless Chromium per request (api/og.ts's
 * fallback), capture the scene the sharer's own browser already rendered:
 * temporarily switch to the same branded framing the server capture uses
 * (isometric, grid hidden, camera panned for the logo), grab the canvas,
 * composite the logo on top, upload it, then restore the live view. Best
 * effort — any failure here just means api/og.ts falls back to rendering
 * it server-side on first request, so this never blocks the actual share.
 *
 * Not imported statically by exprShare.ts (which stays Node-test-safe) —
 * loaded via dynamic import from shareExpressionLink() instead.
 */
import { camera, controls, isIsometric, setIsometric } from "./scene.js";
import { state } from "./state.js";

// Keep in sync with scripts/og/renderShareOg.mjs's SHARE_CAMERA — same
// view direction as the default framing, closer in (overflow past the
// edges is fine) and panned right so the subject clears the logo.
const SHARE_CAMERA_POSITION: [number, number, number] = [6.65, 2.54, 3.46];
const SHARE_CAMERA_TARGET: [number, number, number] = [1.75, -1.92, 0];

const OUTPUT_W = 1200;
const OUTPUT_H = 630;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function findCanvas(): HTMLCanvasElement {
  const el =
    document.querySelector<HTMLCanvasElement>("#viewport canvas.clip-gpu") ??
    document.querySelector<HTMLCanvasElement>("#viewport canvas");
  if (!el) throw new Error("shareCapture: no viewport canvas found");
  return el;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`shareCapture: failed to load ${src}`));
    img.src = src;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("shareCapture: toBlob returned null"));
    }, "image/png");
  });
}

/** Draw `img` into `ctx` filling the target box, cropping to fit (CSS object-fit: cover). */
function drawCover(ctx: CanvasRenderingContext2D, img: HTMLImageElement, w: number, h: number) {
  const scale = Math.max(w / img.naturalWidth, h / img.naturalHeight);
  const dw = img.naturalWidth * scale;
  const dh = img.naturalHeight * scale;
  ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
}

function payloadFromShareUrl(shareUrl: string): string | null {
  const path = new URL(shareUrl).pathname;
  const prefix = "/s/";
  if (!path.startsWith(prefix)) return null;
  const payload = path.slice(prefix.length);
  return payload || null;
}

export async function captureAndUploadOgImage(shareUrl: string): Promise<void> {
  const payload = payloadFromShareUrl(shareUrl);
  if (!payload) return;

  const prevPosition = camera.position.clone();
  const prevTarget = controls.target.clone();
  const prevGridAxes = state.showGridAxes;
  const prevIsometric = isIsometric();

  try {
    state.showGridAxes = false;
    camera.up.set(0, 0, 1);
    camera.position.set(...SHARE_CAMERA_POSITION);
    controls.target.set(...SHARE_CAMERA_TARGET);
    controls.update();
    setIsometric(true);
    state.clipDirty = true;

    // Let the render loop actually draw the new framing before capturing —
    // there's no "settle" needed (the scene is already baked, unlike a
    // fresh headless load), just enough frames for the camera/grid change
    // to land on screen.
    for (let i = 0; i < 4; i++) await nextFrame();
    await wait(120);

    const shotBlob = await canvasToBlob(findCanvas());
    const shotUrl = URL.createObjectURL(shotBlob);
    let shotImg: HTMLImageElement;
    try {
      shotImg = await loadImage(shotUrl);
    } finally {
      URL.revokeObjectURL(shotUrl);
    }

    const out = document.createElement("canvas");
    out.width = OUTPUT_W;
    out.height = OUTPUT_H;
    const ctx = out.getContext("2d");
    if (!ctx) throw new Error("shareCapture: 2D context unavailable");
    ctx.fillStyle = "#1a1228";
    ctx.fillRect(0, 0, OUTPUT_W, OUTPUT_H);
    drawCover(ctx, shotImg, OUTPUT_W, OUTPUT_H);

    // Matches scripts/og/composite.mjs's buildSingleShotHtml layout.
    const logoImg = await loadImage("/logo.svg");
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 10;
    ctx.shadowOffsetY = 2;
    const iconSize = 130;
    const iconX = 84;
    const iconY = (OUTPUT_H - iconSize) / 2;
    ctx.drawImage(logoImg, iconX, iconY, iconSize, iconSize);
    ctx.font = "700 76px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText("laplaci", iconX + iconSize + 22, OUTPUT_H / 2);
    ctx.restore();

    const finalBlob = await canvasToBlob(out);
    await fetch(`/api/upload-og?e=${encodeURIComponent(payload)}`, {
      method: "POST",
      headers: { "Content-Type": "image/png" },
      body: finalBlob,
    });
  } finally {
    camera.position.copy(prevPosition);
    controls.target.copy(prevTarget);
    controls.update();
    if (isIsometric() !== prevIsometric) setIsometric(prevIsometric);
    state.showGridAxes = prevGridAxes;
    state.clipDirty = true;
  }
}
