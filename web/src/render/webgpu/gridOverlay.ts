import type { Camera } from "three";
import type { DirMatrix } from "../camera.js";
import { gpu, labelVertScratch } from "./gpuState.js";
import type { ClipGpuTheme } from "./marchTypes.js";

type GridPlane = "xz" | "xy" | "yz";

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

function hexToRgb(hex: number): [number, number, number] {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
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

  const emitPlane = (axis: GridPlane, alpha: number) => {
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
  emitPlane("xy", 0.55);
  emitPlane("xz", 0.35);
  emitPlane("yz", 0.35);

  const axisLen = extent + 0.25;
  n = pushGridLine(data, n, 0, 0, 0, axisLen, 0, 0, themeAxisXRgb[0], themeAxisXRgb[1], themeAxisXRgb[2], 0.95);
  n = pushGridLine(data, n, 0, 0, 0, 0, axisLen, 0, themeAxisYRgb[0], themeAxisYRgb[1], themeAxisYRgb[2], 0.95);
  n = pushGridLine(data, n, 0, 0, 0, 0, 0, axisLen, themeAxisZRgb[0], themeAxisZRgb[1], themeAxisZRgb[2], 0.95);

  const bh = h;
  const br = themeBoxRgb[0]; const bg = themeBoxRgb[1]; const bb = themeBoxRgb[2]; const ba = 0.85;
  const corners: [number, number, number][] = [
    [-bh, -bh, -bh], [bh, -bh, -bh], [bh, bh, -bh], [-bh, bh, -bh],
    [-bh, -bh, bh], [bh, -bh, bh], [bh, bh, bh], [-bh, bh, bh],
  ];
  const edges: [number, number][] = [
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
    if (gpu.gridVertexBuf) { try { gpu.gridVertexBuf.destroy(); } catch { /* */ } }
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

function ensureAxisLabelAtlas(): void {
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

/** Camera-facing x/y/z billboards at axis tips (world size ~0.42). */
export function uploadAxisLabelBillboards(camera: Camera, half: number): number {
  ensureAxisLabelAtlas();
  if (!gpu.device || !gpu.labelVertexBuf) return 0;

  const h = Math.max(0.5, half);
  const tip = Math.ceil(h + 0.5) + 0.25;
  const e = camera.matrixWorld.elements;
  const rx = e[0]; const ry = e[1]; const rz = e[2];
  const ux = e[4]; const uy = e[5]; const uz = e[6];
  const hs = 0.21;

  const centers: [number, number, number, number][] = [
    [tip, 0, 0, 0],
    [0, tip, 0, 1],
    [0, 0, tip, 2],
  ];
  const corners: [number, number][] = [
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
      labelVertScratch[o + 4] = sx < 0 ? u0 : u1;
      labelVertScratch[o + 5] = sy < 0 ? 1 : 0;
      vi++;
    }
  }
  gpu.device.queue.writeBuffer(gpu.labelVertexBuf, 0, labelVertScratch);
  return vi;
}

export function packGridParams(
  viewProj: ArrayLike<number>, ro: [number, number, number], half: number,
  M: DirMatrix, fbW: number, fbH: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(160);
  const f32 = new Float32Array(buf);
  for (let i = 0; i < 16; i++) f32[i] = viewProj[i];
  f32[16] = ro[0]; f32[17] = ro[1]; f32[18] = ro[2]; f32[19] = half;
  f32[20] = M[0]; f32[21] = M[1]; f32[22] = M[2]; f32[23] = 0;
  f32[24] = M[3]; f32[25] = M[4]; f32[26] = M[5]; f32[27] = 0;
  f32[28] = M[6]; f32[29] = M[7]; f32[30] = M[8]; f32[31] = 0;
  f32[32] = fbW; f32[33] = fbH; f32[34] = 0; f32[35] = 0;
  return buf;
}
