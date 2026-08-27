/**
 * WebGPU volume march: IDCT dens grids + iso manifolds + multi-layer Beer.
 */
import type { Camera, PerspectiveCamera } from "three";
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix } from "../camera.js";
import { MAX_GRAD_STOPS } from "../../model/expressions.js";
import {
  gpu,
  MAX_DENS_LAYERS,
  labelVertScratch,
  resetPipelinesOnDeviceLost,
  DEFAULT_DENS_RGB,
  DEFAULT_DENS_RGB2,
  DEFAULT_ISO_RGB,
  DEFAULT_ISO_RGB2,
} from "./gpuState.js";
import {
  packDrawParamsIso,
  packDrawParamsBeer,
  packSsaoParams,
  writeLayerColors,
} from "./uniforms.js";
import {
  uploadSceneColors,
  uploadSceneVolumes,
  setConstraintKeyframeBlends,
  patchConstraintKeyframeFrame,
  hasUploadedVolume,
  ensureVolumeBuf,
} from "./sceneUpload.js";
import { ensurePipelinesForDegree as buildPipelines } from "./pipelines.js";

export {
  MAX_DENS_LAYERS,
  uploadSceneColors,
  uploadSceneVolumes,
  setConstraintKeyframeBlends,
  patchConstraintKeyframeFrame,
  hasUploadedVolume,
};

export interface ClipGpuTheme {
  gridMajor?: number;
  gridMinor?: number;
  boxEdgeRgb?: number[];
  axisXRgb?: number[];
  axisYRgb?: number[];
  axisZRgb?: number[];
  labelStroke?: string;
}

export interface ClipGpuProfile {
  idctMs: number;
  marchMs: number;
  marchFbW: number;
  marchFbH: number;
  presentWallMs: number;
  presentIntervalMs: number;
  lastPresentAt: number;
  method: string;
  gridM: number;
  timestamps: boolean;
  isoInterp: string;
}

export interface RenderClipFrameGpuParams {
  camera: PerspectiveCamera;
  half: number;
  fbW: number;
  fbH: number;
  scale: number;
  steps: number;
  ndcOffsetX?: number;
  displayW?: number;
  displayH?: number;
}

export function getIsoInterpHermite(): boolean {
  return gpu.isoInterpHermite;
}

/** @returns true if the mode changed (iso pipeline must rebuild) */
export function setIsoInterpHermite(on: boolean): boolean {
  const next = !!on;
  if (next === gpu.isoInterpHermite) return false;
  gpu.isoInterpHermite = next;
  gpu.isoPipeline = null;
  return true;
}

function pushGridVert(
  dst: Float32Array, i: number, x: number, y: number, z: number,
  r: number, g: number, b: number, a: number,
): void {
  const o = i * 8;
  dst[o] = x; dst[o + 1] = y; dst[o + 2] = z; dst[o + 3] = 0;
  dst[o + 4] = r; dst[o + 5] = g; dst[o + 6] = b; dst[o + 7] = a;
}

function pushGridLine(
  dst: Float32Array, i: number,
  ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  r: number, g: number, b: number, a: number,
): number {
  pushGridVert(dst, i, ax, ay, az, r, g, b, a);
  pushGridVert(dst, i + 1, bx, by, bz, r, g, b, a);
  return i + 2;
}

let themeGridMajor = 0x6b5a82;
let themeGridMinor = 0x3d2f55;
let themeBoxRgb = [0.35, 0.29, 0.45];
let themeAxisXRgb = [0.9, 0.35, 0.38];
let themeAxisYRgb = [0.35, 0.75, 0.48];
let themeAxisZRgb = [0.79, 0.66, 0.91];
let themeLabelStroke = "rgba(26, 18, 40, 0.58)";

export function applyClipGpuTheme(colors: ClipGpuTheme): void {
  if (colors.gridMajor) themeGridMajor = colors.gridMajor;
  if (colors.gridMinor) themeGridMinor = colors.gridMinor;
  if (colors.boxEdgeRgb) themeBoxRgb = colors.boxEdgeRgb;
  if (colors.axisXRgb) themeAxisXRgb = colors.axisXRgb;
  if (colors.axisYRgb) themeAxisYRgb = colors.axisYRgb;
  if (colors.axisZRgb) themeAxisZRgb = colors.axisZRgb;
  if (colors.labelStroke) themeLabelStroke = colors.labelStroke;
  gpu.gridHalf = -1;
  gpu.labelAtlasDirty = true;
}

function hexToRgb(hex: number): number[] {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
}

export function syncClipGpuWorldGrid(half: number): void {
  const h = Math.max(0.5, half);
  if (!gpu.device) {
    gpu.gridHalf = h;
    return;
  }
  if (gpu.gridVertexBuf && Math.abs(gpu.gridHalf - h) < 1e-9) return;
  gpu.gridHalf = h;

  const extent = Math.ceil(h + 0.5);
  const size = extent * 2;
  const divisions = Math.max(2, size);
  const step = size / divisions;
  const lo = -size / 2;
  const hi = size / 2;
  const [majR, majG, majB] = hexToRgb(themeGridMajor);
  const [minR, minG, minB] = hexToRgb(themeGridMinor);
  const maxVerts = 3 * (divisions + 1) * 2 * 2 + 6 + 24 + 16;
  const data = new Float32Array(maxVerts * 8);
  let n = 0;

  const emitPlane = (axis: "xz" | "xy" | "yz", alpha: number) => {
    for (let i = 0; i <= divisions; i++) {
      const t = lo + i * step;
      const major = i === 0 || i === divisions || Math.abs(t) < 1e-6;
      const r = major ? majR : minR;
      const g = major ? majG : minG;
      const b = major ? majB : minB;
      const a = alpha;
      if (axis === "xz") {
        n = pushGridLine(data, n, lo, 0, t, hi, 0, t, r, g, b, a);
        n = pushGridLine(data, n, t, 0, lo, t, 0, hi, r, g, b, a);
      } else if (axis === "xy") {
        n = pushGridLine(data, n, lo, t, 0, hi, t, 0, r, g, b, a);
        n = pushGridLine(data, n, t, lo, 0, t, hi, 0, r, g, b, a);
      } else {
        n = pushGridLine(data, n, 0, lo, t, 0, hi, t, r, g, b, a);
        n = pushGridLine(data, n, 0, t, lo, 0, t, hi, r, g, b, a);
      }
    }
  };
  emitPlane("xz", 0.55);
  emitPlane("xy", 0.35);
  emitPlane("yz", 0.35);

  // RGB axes
  const axisLen = extent + 0.25;
  n = pushGridLine(data, n, 0, 0, 0, axisLen, 0, 0, themeAxisXRgb[0], themeAxisXRgb[1], themeAxisXRgb[2], 0.95);
  n = pushGridLine(data, n, 0, 0, 0, 0, axisLen, 0, themeAxisYRgb[0], themeAxisYRgb[1], themeAxisYRgb[2], 0.95);
  n = pushGridLine(data, n, 0, 0, 0, 0, 0, axisLen, themeAxisZRgb[0], themeAxisZRgb[1], themeAxisZRgb[2], 0.95);

  // Fit-box wireframe
  const bh = h;
  const br = themeBoxRgb[0]; const bg = themeBoxRgb[1]; const bb = themeBoxRgb[2]; const ba = 0.85;
  const corners = [
    [-bh, -bh, -bh], [bh, -bh, -bh], [bh, bh, -bh], [-bh, bh, -bh],
    [-bh, -bh, bh], [bh, -bh, bh], [bh, bh, bh], [-bh, bh, bh],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  for (const [i0, i1] of edges) {
    const a = corners[i0]; const b = corners[i1];
    n = pushGridLine(data, n, a[0], a[1], a[2], b[0], b[1], b[2], br, bg, bb, ba);
  }

  gpu.gridVertexCount = n;
  const bytes = n * 8 * 4;
  if (!gpu.gridVertexBuf || gpu.gridVertexCapacity < bytes) {
    if (gpu.gridVertexBuf) { try { gpu.gridVertexBuf.destroy(); } catch (_) {} }
    gpu.gridVertexCapacity = Math.max(bytes, 4096);
    gpu.gridVertexBuf = gpu.device.createBuffer({
      size: gpu.gridVertexCapacity,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }
  gpu.device.queue.writeBuffer(gpu.gridVertexBuf, 0, data.subarray(0, n * 8));
}

function rgb01Css(rgb: number[]): string {
  const r = Math.round(Math.min(1, Math.max(0, rgb[0])) * 255);
  const g = Math.round(Math.min(1, Math.max(0, rgb[1])) * 255);
  const b = Math.round(Math.min(1, Math.max(0, rgb[2])) * 255);
  return `rgb(${r},${g},${b})`;
}

/** Bake x/y/z glyphs into a 3-cell atlas (theme-colored). */
function ensureAxisLabelAtlas() {
  if (!gpu.device) return;
  if (!gpu.labelAtlasDirty && gpu.labelAtlasTex && gpu.labelAtlasSamp) return;

  const cell = 128;
  const canvasEl = document.createElement("canvas");
  canvasEl.width = cell * 3;
  canvasEl.height = cell;
  const c2d = canvasEl.getContext("2d");
  if (!c2d) return;
  c2d.clearRect(0, 0, canvasEl.width, canvasEl.height);
  c2d.font = "600 72px 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif";
  c2d.textAlign = "center";
  c2d.textBaseline = "middle";
  c2d.lineWidth = 3;
  c2d.strokeStyle = themeLabelStroke || "rgba(26, 18, 40, 0.58)";
  const glyphs = [
    { ch: "x", rgb: themeAxisXRgb },
    { ch: "y", rgb: themeAxisYRgb },
    { ch: "z", rgb: themeAxisZRgb },
  ];
  for (let i = 0; i < 3; i++) {
    const cx = cell * i + cell / 2;
    const cy = cell / 2 + 1;
    c2d.strokeText(glyphs[i].ch, cx, cy);
    c2d.fillStyle = rgb01Css(glyphs[i].rgb);
    c2d.fillText(glyphs[i].ch, cx, cy);
  }

  if (!gpu.labelAtlasTex) {
    gpu.labelAtlasTex = gpu.device.createTexture({
      size: [canvasEl.width, canvasEl.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  gpu.device.queue.copyExternalImageToTexture(
    { source: canvasEl },
    { texture: gpu.labelAtlasTex },
    [canvasEl.width, canvasEl.height],
  );
  if (!gpu.labelAtlasSamp) {
    gpu.labelAtlasSamp = gpu.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
  }
  if (!gpu.labelVertexBuf) {
    gpu.labelVertexBuf = gpu.device.createBuffer({
      size: labelVertScratch.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }
  gpu.labelAtlasDirty = false;
}

/**
 * Camera-facing x/y/z billboards at axis tips (world size ~0.42).
 * @param {{ matrixWorld: { elements: ArrayLike<number> } }} camera
 * @param {number} half
 */
function uploadAxisLabelBillboards(camera: Camera, half: number): number {
  ensureAxisLabelAtlas();
  if (!gpu.device || !gpu.labelVertexBuf) return 0;

  const h = Math.max(0.5, half);
  const tip = Math.ceil(h + 0.5) + 0.25;
  const e = camera.matrixWorld.elements;
  const rx = e[0]; const ry = e[1]; const rz = e[2];
  const ux = e[4]; const uy = e[5]; const uz = e[6];
  const hs = 0.21;

  const centers = [
    [tip, 0, 0, 0],
    [0, tip, 0, 1],
    [0, 0, tip, 2],
  ];
  // UV: WebGPU texture (0,0) = top-left of atlas row.
  const corners = [
    [-1, -1], [1, -1], [1, 1],
    [-1, -1], [1, 1], [-1, 1],
  ];
  let vi = 0;
  for (const [cx, cy, cz, cell] of centers) {
    const u0 = cell / 3;
    const u1 = (cell + 1) / 3;
    for (const [sx, sy] of corners) {
      const o = vi * 6;
      labelVertScratch[o] = cx + rx * sx * hs + ux * sy * hs;
      labelVertScratch[o + 1] = cy + ry * sx * hs + uy * sy * hs;
      labelVertScratch[o + 2] = cz + rz * sx * hs + uz * sy * hs;
      labelVertScratch[o + 3] = 0;
      // sy=-1 → v=1 (bottom), sy=+1 → v=0 (top) to match canvas Y-down
      labelVertScratch[o + 4] = sx < 0 ? u0 : u1;
      labelVertScratch[o + 5] = sy < 0 ? 1 : 0;
      vi++;
    }
  }
  gpu.device.queue.writeBuffer(gpu.labelVertexBuf, 0, labelVertScratch);
  return vi;
}

function packGridParams(
  viewProj: ArrayLike<number>, ro: number[], half: number,
  M: Float64Array | Float32Array | number[], fbW: number, fbH: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(160);
  const f32 = new Float32Array(buf);
  // Column-major mat4
  for (let i = 0; i < 16; i++) f32[i] = viewProj[i];
  f32[16] = ro[0]; f32[17] = ro[1]; f32[18] = ro[2]; f32[19] = half;
  f32[20] = M[0]; f32[21] = M[1]; f32[22] = M[2]; f32[23] = 0;
  f32[24] = M[3]; f32[25] = M[4]; f32[26] = M[5]; f32[27] = 0;
  f32[28] = M[6]; f32[29] = M[7]; f32[30] = M[8]; f32[31] = 0;
  f32[32] = fbW; f32[33] = fbH; f32[34] = 0; f32[35] = 0;
  return buf;
}

export function isClipBakeGpuReady(): boolean {
  return Boolean(
    gpu.device && gpu.isoPipeline && gpu.beerPipeline && gpu.fxaaPipeline && gpu.ssaoPipeline && gpu.gridPipeline && gpu.labelPipeline,
  );
}
export function isClipMarchReady(): boolean {
  return Boolean(
    isClipBakeGpuReady() && gpu.ctx && gpu.sceneM > 1 &&
    (gpu.densLayerCount > 0 || gpu.sceneConstraints.length > 0),
  );
}

function noteGpuPresent(submitWallAt: number): void {
  const now = performance.now();
  gpu.profilePresentWallMs = gpu.profilePresentWallMs * 0.85 + (now - submitWallAt) * 0.15;
  if (gpu.lastPresentAt > 0) gpu.profilePresentIntervalMs = gpu.profilePresentIntervalMs * 0.85 + (now - gpu.lastPresentAt) * 0.15;
  else gpu.profilePresentIntervalMs = now - submitWallAt;
  gpu.lastPresentAt = now;
}

export function getClipGpuProfile(): ClipGpuProfile {
  return {
    idctMs: gpu.profileBakeMs,
    marchMs: gpu.profileMarchMs,
    marchFbW: gpu.profileMarchFbW,
    marchFbH: gpu.profileMarchFbH,
    presentWallMs: gpu.profilePresentWallMs,
    presentIntervalMs: gpu.profilePresentIntervalMs,
    lastPresentAt: gpu.lastPresentAt,
    method: gpu.profileMethod,
    gridM: gpu.profileGridM,
    timestamps: gpu.timestampsSupported,
    isoInterp: gpu.isoInterpHermite ? "hermite" : "trilinear",
  };
}

export function resetClipGpuProfile(): void {
  gpu.profileBakeMs = 0;
  gpu.profileMarchMs = 0;
  gpu.profileMarchFbW = 0;
  gpu.profileMarchFbH = 0;
}

function attachClipGpuCanvas(viewportEl: HTMLElement): HTMLCanvasElement {
  if (gpu.canvas) return gpu.canvas;
  gpu.canvas = document.createElement("canvas");
  gpu.canvas.className = "clip-gpu";
  gpu.canvas.style.cssText =
    "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:1;display:none;";
  viewportEl.appendChild(gpu.canvas);
  return gpu.canvas;
}
export function setClipGpuCanvasVisible(visible: boolean): void {
  if (!gpu.canvas) return;
  gpu.canvas.style.display = visible ? "block" : "none";
}

function ensureOcclTex(w: number, h: number): void {
  if (gpu.occlTex && gpu.occlW === w && gpu.occlH === h) return;
  if (gpu.occlTex) { try { gpu.occlTex.destroy(); } catch { /* */ } }
  if (!gpu.device) return;
  gpu.occlTex = gpu.device.createTexture({
    size: [w, h],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.occlW = w;
  gpu.occlH = h;
}

function ensureDepthTex(w: number, h: number): void {
  if (gpu.depthTex && gpu.depthW === w && gpu.depthH === h) return;
  if (gpu.depthTex) { try { gpu.depthTex.destroy(); } catch { /* */ } }
  if (!gpu.device) return;
  gpu.depthTex = gpu.device.createTexture({
    size: [w, h],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  gpu.depthW = w;
  gpu.depthH = h;
}

function ensureNormalTex(w: number, h: number): void {
  if (gpu.normalTex && gpu.normalW === w && gpu.normalH === h) return;
  if (gpu.normalTex) { try { gpu.normalTex.destroy(); } catch { /* */ } }
  if (!gpu.device) return;
  gpu.normalTex = gpu.device.createTexture({
    size: [w, h],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.normalW = w;
  gpu.normalH = h;
}

function ensureSceneColorTex(w: number, h: number): void {
  if (gpu.sceneColorTex && gpu.sceneColorW === w && gpu.sceneColorH === h) return;
  if (gpu.sceneColorTex) { try { gpu.sceneColorTex.destroy(); } catch { /* */ } }
  if (!gpu.device) return;
  gpu.sceneColorTex = gpu.device.createTexture({
    size: [w, h],
    format: gpu.canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.sceneColorW = w;
  gpu.sceneColorH = h;
}

function ensureSceneColorAoTex(w: number, h: number): void {
  if (gpu.sceneColorAoTex && gpu.sceneColorAoW === w && gpu.sceneColorAoH === h) return;
  if (gpu.sceneColorAoTex) { try { gpu.sceneColorAoTex.destroy(); } catch { /* */ } }
  if (!gpu.device) return;
  gpu.sceneColorAoTex = gpu.device.createTexture({
    size: [w, h],
    format: gpu.canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  gpu.sceneColorAoW = w;
  gpu.sceneColorAoH = h;
}

export function resizeClipGpuCanvas(pixelW: number, pixelH: number): { w: number; h: number } {
  if (!gpu.canvas || !gpu.device) return { w: pixelW | 0, h: pixelH | 0 };
  if (!gpu.ctx) {
    gpu.ctx = gpu.canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!gpu.ctx) return { w: pixelW | 0, h: pixelH | 0 };
  }
  const w = Math.max(1, pixelW | 0);
  const h = Math.max(1, pixelH | 0);
  const changed = gpu.canvas.width !== w || gpu.canvas.height !== h;
  if (changed) {
    gpu.canvas.width = w;
    gpu.canvas.height = h;
  }
  gpu.canvas.style.width = "100%";
  gpu.canvas.style.height = "100%";
  if (changed || !gpu.canvas._clipConfigured) {
    gpu.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    gpu.ctx.configure({
      device: gpu.device,
      format: gpu.canvasFormat,
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    gpu.canvas._clipConfigured = true;
  }
  // March targets are sized independently in renderClipFrameGpu.
  return { w: gpu.canvas.width, h: gpu.canvas.height };
}

function ensureMarchTargets(w: number, h: number): void {
  ensureOcclTex(w, h);
  ensureDepthTex(w, h);
  ensureNormalTex(w, h);
  ensureSceneColorTex(w, h);
  ensureSceneColorAoTex(w, h);
}



/** Clear the WebGPU overlay to transparent (no density / iso). */
export function clearClipGpuFrame(fbW: number, fbH: number): boolean {
  if (!gpu.device || !gpu.ctx || !gpu.canvas) return false;
  const { w, h } = resizeClipGpuCanvas(fbW, fbH);
  const view = gpu.ctx.getCurrentTexture().createView();
  const enc = gpu.device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.end();
  gpu.device.queue.submit([enc.finish()]);
  gpu.profileMarchFbW = w;
  gpu.profileMarchFbH = h;
  gpu.profileMethod = "gpu-clear";
  return true;
}

export async function ensurePipelinesForDegree(deg: number) {
  const result = await buildPipelines(deg);
  if (result && result.gridRebuildHalf != null) {
    syncClipGpuWorldGrid(result.gridRebuildHalf);
  }
  return result !== false;
}

export async function initClipBakeGpu(viewportEl: HTMLElement | null | undefined): Promise<boolean> {
  if (isClipBakeGpuReady()) return true;
  if (gpu.initFailed) return false;
  if (gpu.initPromise) return gpu.initPromise;
  gpu.initPromise = (async () => {
    try {
      if (!navigator.gpu) { gpu.initFailed = true; return false; }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) { gpu.initFailed = true; return false; }
      gpu.timestampsSupported = adapter.features.has("timestamp-query");
      const requiredFeatures: GPUFeatureName[] = gpu.timestampsSupported ? ["timestamp-query"] : [];
      gpu.device = await adapter.requestDevice({ requiredFeatures });
      gpu.device.lost.then(() => {
        gpu.device = null;
        resetPipelinesOnDeviceLost();
        gpu.initFailed = true;
      });
      if (gpu.timestampsSupported) {
        gpu.stampQuerySet = gpu.device.createQuerySet({ type: "timestamp", count: 2 });
        gpu.stampResolveBuf = gpu.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        gpu.stampReadBuf = gpu.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }
      gpu.drawParamBuf = gpu.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.drawParamBufBeer = gpu.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.fxaaParamBuf = gpu.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.ssaoParamBuf = gpu.device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.gridParamBuf = gpu.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gpu.fxaaSampler = gpu.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      });
      gpu.colorBuf = gpu.device.createBuffer({
        size: MAX_DENS_LAYERS * MAX_GRAD_STOPS * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      writeLayerColors(gpu.device, gpu.colorBuf, [[DEFAULT_DENS_RGB, DEFAULT_DENS_RGB2] as number[][]]);
      ensureVolumeBuf(8 * 8 * 8);
      await ensurePipelinesForDegree(4);
      if (viewportEl) attachClipGpuCanvas(viewportEl);
      if (gpu.canvas) gpu.ctx = gpu.canvas.getContext("webgpu") as GPUCanvasContext | null;
      return isClipBakeGpuReady();
    } catch (e) {
      console.warn("[clipBakeGpu] init failed", e);
      gpu.initFailed = true;
      gpu.device = null;
      resetPipelinesOnDeviceLost();
      return false;
    }
  })();
  return gpu.initPromise;
}

function scheduleStampReadback(): void {
  if (!gpu.timestampsSupported || !gpu.stampReadBuf || gpu.stampReadPending) return;
  gpu.stampReadPending = true;
  const readBuf = gpu.stampReadBuf;
  readBuf.mapAsync(GPUMapMode.READ).then(() => {
    const stamps = new BigInt64Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    gpu.stampReadPending = false;
    if (stamps[1] > stamps[0]) {
      gpu.profileMarchMs = gpu.profileMarchMs * 0.7 + Number(stamps[1] - stamps[0]) / 1e6 * 0.3;
    }
  }).catch(() => { gpu.stampReadPending = false; });
}

function darken(c: number[], t: number): number[] {
  return [c[0] * t, c[1] * t, c[2] * t];
}

export function renderClipFrameGpu({
  camera, half, fbW, fbH, scale, steps, ndcOffsetX = 0, displayW = 0, displayH = 0,
}: RenderClipFrameGpuParams): boolean {
  if (
    !isClipBakeGpuReady() || !gpu.ctx || !gpu.volumeBuf || !gpu.colorBuf ||
    !gpu.fxaaParamBuf || !gpu.ssaoParamBuf || !gpu.fxaaSampler || !gpu.gridParamBuf
  ) {
    return false;
  }
  if (gpu.densLayerCount < 1 && gpu.sceneConstraints.length < 1) return false;

  const device = gpu.device!;
  const ctx = gpu.ctx;
  const volumeBuf = gpu.volumeBuf;
  const colorBuf = gpu.colorBuf;
  const fxaaParamBuf = gpu.fxaaParamBuf;
  const ssaoParamBuf = gpu.ssaoParamBuf;
  const fxaaSampler = gpu.fxaaSampler;
  const gridParamBuf = gpu.gridParamBuf;
  const isoPipeline = gpu.isoPipeline!;
  const beerPipeline = gpu.beerPipeline!;
  const ssaoPipeline = gpu.ssaoPipeline!;
  const fxaaPipeline = gpu.fxaaPipeline!;
  const gridPipeline = gpu.gridPipeline!;
  const drawParamBuf = gpu.drawParamBuf!;
  const drawParamBufBeer = gpu.drawParamBufBeer!;

  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const Mat = offsetDirMatrix(ndcToDirMatrix(camera, sx, sy), ndcOffsetX);
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;
  const Mgrid = gpu.sceneM;

  const marchW = Math.max(1, fbW | 0);
  const marchH = Math.max(1, fbH | 0);
  const outW = Math.max(1, (displayW | 0) || marchW);
  const outH = Math.max(1, (displayH | 0) || marchH);
  resizeClipGpuCanvas(outW, outH);
  ensureMarchTargets(marchW, marchH);
  syncClipGpuWorldGrid(h);

  const sceneColorTex = gpu.sceneColorTex;
  const sceneColorAoTex = gpu.sceneColorAoTex;
  const occlTex = gpu.occlTex;
  const depthTex = gpu.depthTex;
  const normalTex = gpu.normalTex;
  if (!sceneColorTex || !sceneColorAoTex || !occlTex || !depthTex || !normalTex) return false;

  gpu.profileMarchFbW = marchW;
  gpu.profileMarchFbH = marchH;
  gpu.profileMethod = "gpu-iso+ssao+beer+grid+fxaa";
  gpu.profileGridM = Mgrid;

  if (gpu.scenePacked) device.queue.writeBuffer(volumeBuf, 0, gpu.scenePacked);
  writeLayerColors(device, colorBuf, gpu.densGradStops);

  let sceneView = sceneColorTex.createView();
  const swapView = ctx.getCurrentTexture().createView();
  const occlView = occlTex.createView();
  const depthView = depthTex.createView();
  const normalView = normalTex.createView();

  // Clear scene color + occl (far = 1) + normals + depth (far = 1)
  {
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: sceneView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: occlView,
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: normalView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  for (const c of gpu.sceneConstraints) {
    const stride = c.frameStride || 0;
    const base0 = c.base + (c.i0 | 0) * stride;
    const base1 = c.base + (c.i1 | 0) * stride;
    const blendT = Number.isFinite(c.t) ? c.t : 0;
    const c0 = c.color || DEFAULT_ISO_RGB;
    const c1 = c.color2 || DEFAULT_ISO_RGB2;
    const stops = c.colors || [c0, c1];
    device.queue.writeBuffer(
      drawParamBuf,
      0,
      packDrawParamsIso(
        marchW, marchH, Mgrid, steps, h, scale, c.isoLevel, base0, ro, Mat,
        c0, c1,
        base1, blendT, stops,
      ),
    );
    const bg = device.createBindGroup({
      layout: isoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: drawParamBuf } },
        { binding: 1, resource: { buffer: volumeBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: sceneView, loadOp: "load", storeOp: "store" },
        { view: occlView, loadOp: "load", storeOp: "store" },
        { view: normalView, loadOp: "load", storeOp: "store" },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(isoPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // Iso-only SSAO → ping-pong color (skip if no manifolds this frame).
  if (gpu.sceneConstraints.length > 0) {
    const aoView = sceneColorAoTex.createView();
    device.queue.writeBuffer(
      ssaoParamBuf,
      0,
      packSsaoParams(
        marchW, marchH, h,
        /* radius */ Math.max(0.2, h * 0.18),
        /* strength */ 0.85,
        /* bias */ 0.03,
        ro, Mat,
      ),
    );
    const bg = device.createBindGroup({
      layout: ssaoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ssaoParamBuf } },
        { binding: 1, resource: sceneView },
        { binding: 2, resource: occlView },
        { binding: 3, resource: normalView },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: aoView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(ssaoPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
    sceneView = aoView;
  }

  if (gpu.densLayerCount > 0 && gpu.densPacked) {
    device.queue.writeBuffer(
      drawParamBufBeer,
      0,
      packDrawParamsBeer(marchW, marchH, Mgrid, steps, h, scale, gpu.densBase, gpu.densLayerCount, ro, Mat),
    );
    const bg = device.createBindGroup({
      layout: beerPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: drawParamBufBeer } },
        { binding: 1, resource: { buffer: volumeBuf } },
        { binding: 2, resource: occlView },
        { binding: 3, resource: { buffer: colorBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: sceneView, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(beerPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // FXAA → swapchain (upsamples march-res color to display resolution).
  {
    const inv = new Float32Array([1 / marchW, 1 / marchH, 0, 0]);
    device.queue.writeBuffer(fxaaParamBuf, 0, inv);
    const bg = device.createBindGroup({
      layout: fxaaPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: fxaaParamBuf } },
        { binding: 1, resource: sceneView },
        { binding: 2, resource: fxaaSampler },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: swapView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(fxaaPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // World grid / axes / labels at display res; soft-occlude via march gpu.occlTex.
  if (gpu.gridVertexBuf && gpu.gridVertexCount > 0 && gpu.gridPipeline) {
    camera.updateMatrixWorld(true);
    const viewProj = new Float32Array(16);
    const e = camera.projectionMatrix.elements;
    const v = camera.matrixWorldInverse.elements;
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        viewProj[c * 4 + r] =
          e[0 * 4 + r] * v[c * 4 + 0] +
          e[1 * 4 + r] * v[c * 4 + 1] +
          e[2 * 4 + r] * v[c * 4 + 2] +
          e[3 * 4 + r] * v[c * 4 + 3];
      }
    }
    device.queue.writeBuffer(
      gridParamBuf,
      0,
      packGridParams(viewProj, ro, h, Mat, outW, outH),
    );
    const labelVertCount = gpu.labelPipeline
      ? uploadAxisLabelBillboards(camera, h)
      : 0;
    const bg = device.createBindGroup({
      layout: gridPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gridParamBuf } },
        { binding: 1, resource: occlView },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: swapView, loadOp: "load", storeOp: "store" },
      ],
    });
    pass.setPipeline(gridPipeline);
    pass.setBindGroup(0, bg);
    pass.setVertexBuffer(0, gpu.gridVertexBuf);
    pass.draw(gpu.gridVertexCount);

    if (gpu.labelPipeline && gpu.labelVertexBuf && gpu.labelAtlasTex && gpu.labelAtlasSamp && labelVertCount > 0) {
      const labelPipeline = gpu.labelPipeline;
      const labelBg = device.createBindGroup({
        layout: labelPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gridParamBuf } },
          { binding: 1, resource: gpu.labelAtlasTex.createView() },
          { binding: 2, resource: gpu.labelAtlasSamp },
          { binding: 3, resource: occlView },
        ],
      });
      pass.setPipeline(labelPipeline);
      pass.setBindGroup(0, labelBg);
      pass.setVertexBuffer(0, gpu.labelVertexBuf);
      pass.draw(labelVertCount);
    }

    pass.end();
    device.queue.submit([enc.finish()]);
  }

  const submitWallAt = performance.now();
  void device.queue.onSubmittedWorkDone().then(() => {
    noteGpuPresent(submitWallAt);
    scheduleStampReadback();
  });
  return true;
}
