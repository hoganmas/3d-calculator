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
 * The live camera/canvas are the only ones this app has, so the branded
 * framing genuinely is drawn into the on-screen viewport for a beat — but a
 * frozen snapshot of the *previous* frame (`buildFreezeOverlay`) sits on top
 * of the canvases for that whole window, so the sharer never actually sees
 * the swap: their own view is what's on screen throughout, and it's lifted
 * only once the real camera is back and repainted underneath it. The camera
 * hijack itself is also scoped to just the screenshot (`captureShareShot`),
 * not the slower compositing/sign/upload work after it.
 *
 * Not imported statically by exprShare.ts (which stays Node-test-safe) —
 * loaded via dynamic import from shareExpressionLink() instead.
 */
import { camera, controls, setIsometric } from "./scene.js";
import { state } from "./state.js";
import { els } from "./dom.js";

// Keep in sync with scripts/og/renderShareOg.mjs's SHARE_CAMERA — same
// view direction as the default framing, closer in (overflow past the
// edges is fine) and panned right so the subject clears the logo.
const SHARE_CAMERA_POSITION: [number, number, number] = [6.65, 2.54, 3.46];
const SHARE_CAMERA_TARGET: [number, number, number] = [1.75, -1.92, 0];

const OUTPUT_W = 1200;
const OUTPUT_H = 630;

// Under plain `vite dev` (no `vercel dev`), /api/* has no handler — a POST
// body over ~64KB (Node's stream highWaterMark) to a route that 404s without
// draining it hangs the connection forever instead of erroring. A deployed
// backend actually reads the body, so this never fires there; it's here so a
// misbehaving/unreachable endpoint in any environment can't hang the share
// button — this whole capture/upload is meant to be best-effort.
const API_TIMEOUT_MS = 8000;

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

/** Visible (non-`display:none`) canvases stacked inside the viewport, in paint order. */
function findViewportCanvasLayers(viewport: HTMLElement): HTMLCanvasElement[] {
  return Array.from(viewport.querySelectorAll<HTMLCanvasElement>("canvas"))
    .map((c) => ({ c, z: Number.parseInt(getComputedStyle(c).zIndex, 10) || 0 }))
    .filter(({ c }) => getComputedStyle(c).display !== "none")
    .sort((a, b) => a.z - b.z)
    .map(({ c }) => c);
}

/** True for "transparent" and any rgba(...) with alpha 0. */
function isFullyTransparent(color: string): boolean {
  if (!color || color === "transparent") return true;
  const m = color.match(/rgba?\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*(?:,\s*([\d.]+))?\s*\)/);
  return m ? parseFloat(m[1] ?? "1") === 0 : false;
}

/**
 * The color actually showing through wherever the live canvases are
 * transparent (both three.js renderers clear with alpha 0, and the WebGPU
 * clip-gpu canvas blends with `alphaMode: "premultiplied"`) — walks up from
 * `viewport` to find the nearest ancestor with an opaque background.
 */
function resolveVisibleBackground(el: HTMLElement): string {
  for (let node: HTMLElement | null = el; node; node = node.parentElement) {
    const bg = getComputedStyle(node).backgroundColor;
    if (!isFullyTransparent(bg)) return bg;
  }
  return "#1a1228"; // matches the OG image's own background fallback below
}

/**
 * Snapshot every visible canvas layer into one composited freeze-frame
 * canvas, stacked on top of them at the same size — so the sharer keeps
 * looking at their own last-rendered frame while the layers underneath get
 * driven to the branded OG framing and back.
 */
function buildFreezeOverlay(viewport: HTMLElement): HTMLCanvasElement | null {
  const layers = findViewportCanvasLayers(viewport);
  if (layers.length === 0) return null;

  const w = Math.max(...layers.map((c) => c.width));
  const h = Math.max(...layers.map((c) => c.height));
  if (!w || !h) return null;

  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  out.className = "og-capture-freeze";
  const ctx = out.getContext("2d");
  if (!ctx) return null;
  // Both three.js renderers clear with alpha 0, and the WebGPU canvas blends
  // with alphaMode "premultiplied" — so the layers we're about to draw carry
  // real transparency in their background/empty regions. A blank 2D canvas
  // is transparent by default too, so without this fill those regions would
  // stay transparent here as well, letting the *live* canvases underneath
  // (already mid-swing to the OG framing) show straight through — visible
  // immediately, since the swing starts the instant this function returns.
  ctx.fillStyle = resolveVisibleBackground(viewport);
  ctx.fillRect(0, 0, w, h);
  for (const layer of layers) {
    try {
      ctx.drawImage(layer, 0, 0, w, h);
    } catch {
      // Best effort — a blank layer just leaves a harmless gap in the
      // freeze frame, not a broken capture.
    }
  }
  viewport.appendChild(out);
  return out;
}

/**
 * Swing the live camera to the branded OG framing, screenshot it, then swing
 * back — scoped tightly to just the screenshot so the camera hijack (and the
 * freeze overlay hiding it) lasts as briefly as possible.
 */
async function captureShareShot(): Promise<Blob> {
  const prevPosition = camera.position.clone();
  const prevTarget = controls.target.clone();
  const prevGridAxes = state.showGridAxes;
  // The exact previous fov, not just an isometric on/off flag — restoring by
  // re-running setIsometric() would rescale prevPosition a second time (it's
  // meant to convert a *current* framing to a new fov, not reproduce a
  // ground-truth position we already have), landing the camera somewhere
  // else entirely once the freeze overlay lifts.
  const prevFov = camera.fov;
  const prevDamping = controls.enableDamping;
  const freeze = buildFreezeOverlay(els.viewport);

  try {
    // loop.ts's render loop calls controls.update() every frame regardless
    // of what this function does, and OrbitControls' damping doesn't treat
    // a directly-set position/target as authoritative: it keeps applying
    // whatever orbit/pan momentum was left over from the user's last drag
    // on top of it. Flush that momentum (disable damping, which applies the
    // remainder in one step and zeroes it, then update()) before overriding
    // the camera ourselves — otherwise that leftover momentum keeps nudging
    // the camera on every frame afterward, including post-restore once the
    // freeze overlay is already gone, which looks like the view "settling"
    // on its own. Kept off for the rest of the capture too: the freeze
    // overlay blocks pointer input, so no new momentum can accumulate, and
    // this keeps the eventual restore an exact, undamped snap back.
    controls.enableDamping = false;
    controls.update();

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
    // to land on screen. Hidden behind the freeze overlay the whole time.
    for (let i = 0; i < 4; i++) await nextFrame();
    await wait(120);

    return await canvasToBlob(findCanvas());
  } finally {
    camera.position.copy(prevPosition);
    controls.target.copy(prevTarget);
    camera.fov = prevFov;
    camera.updateProjectionMatrix();
    controls.update();
    controls.enableDamping = prevDamping;
    state.showGridAxes = prevGridAxes;
    state.clipDirty = true;

    if (freeze) {
      // Give the restored framing a few frames to actually repaint
      // underneath before lifting the freeze — otherwise removing it
      // flashes the still-stale OG framing for a tick.
      for (let i = 0; i < 3; i++) await nextFrame();
      freeze.remove();
    }
  }
}

export async function captureAndUploadOgImage(shareUrl: string): Promise<void> {
  const payload = payloadFromShareUrl(shareUrl);
  if (!payload) return;

  const shotBlob = await captureShareShot();
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

  // Best effort: an unsigned upload still has a chance if the server is
  // running fail-open (no OG_SIGNING_SECRET configured); if signing is
  // enforced, upload-og rejects a missing/invalid token on its own.
  let token = "";
  try {
    const signRes = await fetch(`/api/og-sign?e=${encodeURIComponent(payload)}`, {
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });
    if (signRes.ok) {
      const signed = (await signRes.json()) as { token?: string | null };
      if (signed.token) token = signed.token;
    }
  } catch {
    // ignore — fall through with no token.
  }

  const uploadUrl = `/api/upload-og?e=${encodeURIComponent(payload)}${
    token ? `&token=${encodeURIComponent(token)}` : ""
  }`;
  await fetch(uploadUrl, {
    method: "POST",
    headers: { "Content-Type": "image/png" },
    body: finalBlob,
    signal: AbortSignal.timeout(API_TIMEOUT_MS),
  });
}
