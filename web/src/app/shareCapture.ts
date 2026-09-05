/**
 * Client-side OG image capture, run at share-creation time.
 *
 * Rather than spinning up headless Chromium per request (api/og.ts's
 * fallback), capture the scene the sharer's own browser already rendered:
 * switch to the same branded framing the server capture uses (isometric,
 * grid hidden, camera panned for the logo), grab a screenshot, composite
 * the logo on top, and upload it. Best effort — any failure here just means
 * api/og.ts falls back to rendering it server-side on first request, so
 * this never blocks the actual share.
 *
 * Two ways to get that screenshot, tried in order:
 *
 * - `captureShareShotOffscreenGpu` (EXPERIMENTAL): when the WebGPU clip path
 *   is active, render the OG framing straight into a private, never-shown
 *   canvas on the same GPU device — the live camera/controls/canvas are
 *   never touched at all, so there's nothing to freeze or restore. See its
 *   own doc comment for how it borrows the live render pipeline safely.
 * - `captureShareShot` (fallback): this app has only one on-screen camera
 *   and canvas, so the branded framing is genuinely drawn into the visible
 *   viewport for a beat — a frozen snapshot of the *previous* frame
 *   (`buildFreezeOverlay`) sits on top of the canvases for that whole
 *   window so the sharer never actually sees the swap, lifted only once the
 *   real camera is back and repainted underneath it.
 *
 * Not imported statically by exprShare.ts (which stays Node-test-safe) —
 * loaded via dynamic import from shareExpressionLink() instead.
 */
import * as THREE from "three";
import { camera, controls, setIsometric, DEFAULT_FOV, ISO_FOV, fovDistanceScale } from "./scene.js";
import { state } from "./state.js";
import { els } from "./dom.js";
import { gpu } from "../render/webgpu/gpuState.js";
import {
  renderClipFrameGpu,
  resizeClipGpuCanvas,
  isClipBakeGpuReady,
  hasUploadedVolume,
} from "../render/webgpu/march.js";
import { useGpuClipPath, clipUniforms } from "./webglFallback.js";

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

let offscreenCanvas: HTMLCanvasElement | null = null;

function getOffscreenCanvas(): HTMLCanvasElement {
  if (!offscreenCanvas) {
    offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = OUTPUT_W;
    offscreenCanvas.height = OUTPUT_H;
  }
  return offscreenCanvas;
}

/**
 * EXPERIMENTAL: render the OG framing into a private, never-displayed
 * canvas using the same WebGPU device the live view already uses — so the
 * live camera/controls/on-screen canvas are never touched.
 *
 * `renderClipFrameGpu` already takes a `camera` argument rather than
 * reading a shared one, so a throwaway camera is all that's needed there.
 * The actual obstacle is that everything else it touches — the WebGPU
 * context/canvas and every intermediate render-target texture — lives as
 * flat fields on the `gpu` module singleton (gpuState.ts), not as
 * parameters. Rather than refactor that pipeline to accept a target
 * struct (real surgery on rendering code this session can't visually
 * verify), this snapshots the singleton, temporarily points its
 * canvas/context/texture fields at a private target, renders once, and
 * restores the snapshot — the same save/mutate/restore shape already used
 * for the camera in `captureShareShot`, just applied to `gpu`'s fields.
 * The device, pipelines, and uploaded scene buffers (`volumeBuf`/
 * `colorBuf`) are left untouched throughout: they're safely shareable
 * across render targets, only the per-canvas textures need swapping.
 *
 * Everything from the swap to the restore is synchronous (no `await`) —
 * that's what guarantees this can never interleave with the live render
 * loop's own per-frame call to the same function, since JS won't yield to
 * that rAF-driven call in between. Reading the rendered pixels back out
 * (`canvasToBlob`, unavoidably async) happens only *after* `gpu` is fully
 * restored, since the offscreen canvas keeps its rendered bitmap
 * regardless of whether `gpu` still points at it.
 *
 * Returns null (falls back to `captureShareShot`) when the GPU path isn't
 * active, or anything here throws.
 */
async function captureShareShotOffscreenGpu(): Promise<Blob | null> {
  if (!useGpuClipPath() || !hasUploadedVolume() || !isClipBakeGpuReady()) return null;

  const canvas = getOffscreenCanvas();
  const prevGridAxes = state.showGridAxes;
  const prev = { ...gpu };
  let ok = false;

  try {
    // Grid/axis overlays are driven by this flag directly inside
    // renderClipFrameGpu (not a param) — same toggle captureShareShot uses.
    state.showGridAxes = false;

    // The only `gpu` fields sized to a specific canvas. Nulling the
    // textures (rather than just leaving stale live-sized ones in place)
    // forces the pipeline's own `ensure*Targets` allocators to size them
    // for this canvas instead — those allocators are the existing,
    // untouched machinery that already does this on every live resize.
    gpu.canvas = canvas;
    gpu.ctx = null;
    // acquireMarchGpuHandles() (inside renderClipFrameGpu, before it ever
    // gets to its own resizeClipGpuCanvas call) requires gpu.ctx to already
    // be non-null — it won't lazily create one. Configure it up front.
    resizeClipGpuCanvas(OUTPUT_W, OUTPUT_H);
    if (!gpu.ctx) return null;
    gpu.occlIsoTex = null;
    gpu.occlSurfTex = null;
    gpu.isoCoarseColorTex = null;
    gpu.isoCoarseOcclTex = null;
    gpu.isoCoarseNormalTex = null;
    gpu.isoCoarseDepthTex = null;
    gpu.isoMidColorTex = null;
    gpu.isoMidOcclTex = null;
    gpu.isoMidNormalTex = null;
    gpu.isoMidDepthTex = null;
    gpu.depthTex = null;
    gpu.normalTex = null;
    gpu.sceneColorTex = null;
    gpu.volColorTex = null;
    gpu.volMidColorTex = null;

    // A private camera, never the shared live one — same branded framing
    // math captureShareShot uses (SHARE_CAMERA_POSITION/TARGET, scaled for
    // the isometric FOV), just computed directly instead of animated from
    // whatever the live camera currently happens to be at.
    const ogCamera = new THREE.PerspectiveCamera(ISO_FOV, OUTPUT_W / OUTPUT_H, 0.05, 2000);
    ogCamera.up.set(0, 0, 1);
    const target = new THREE.Vector3(...SHARE_CAMERA_TARGET);
    const isoScale = fovDistanceScale(DEFAULT_FOV, ISO_FOV);
    ogCamera.position.set(...SHARE_CAMERA_POSITION).sub(target).multiplyScalar(isoScale).add(target);
    ogCamera.lookAt(target);
    ogCamera.updateProjectionMatrix();
    ogCamera.updateMatrixWorld(true);

    // The iso-refine ladder (coarse → mid → fine) draws all three tiers
    // fresh within a single call — no multi-frame warm-up needed, and no
    // reason to respect the interactive-performance downscale slider for a
    // one-off shot, so isoFineDownscale is forced to 1 (finest) here.
    // ndcOffsetX/Y stay 0: that's for framing around a floating side panel,
    // which doesn't exist for this fixed, panel-free output.
    ok = renderClipFrameGpu({
      camera: ogCamera,
      half: clipUniforms.uHalf.value,
      fbW: OUTPUT_W,
      fbH: OUTPUT_H,
      volFbW: OUTPUT_W,
      volFbH: OUTPUT_H,
      displayW: OUTPUT_W,
      displayH: OUTPUT_H,
      isoFineDownscale: 1,
      scale: clipUniforms.uScale.value,
      steps: clipUniforms.uSteps.value | 0,
      isoSteps: clipUniforms.uIsoSteps.value | 0,
      ndcOffsetX: 0,
      ndcOffsetY: 0,
    });
  } catch {
    ok = false;
  } finally {
    Object.assign(gpu, prev);
    state.showGridAxes = prevGridAxes;
  }

  if (!ok) return null;
  try {
    return await canvasToBlob(canvas);
  } catch {
    return null;
  }
}

export async function captureAndUploadOgImage(shareUrl: string): Promise<void> {
  const payload = payloadFromShareUrl(shareUrl);
  if (!payload) return;

  const shotBlob = (await captureShareShotOffscreenGpu()) ?? (await captureShareShot());
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
