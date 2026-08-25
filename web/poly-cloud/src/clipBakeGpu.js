/**
 * WebGPU clip-grid path (golden): tile-parallel middle-out Babbage dens bake +
 * Beer–Lambert fullscreen march. See research/poly/notes/clip-space-babbage.md.
 *
 * CPU/WebGL fallback: main.js via bakeClipGridFibers + DataTexture.
 * Path C is a separate legacy LOS mode — research/poly/notes/path-c.md.
 */

import {
  ndcToDirMatrix,
  perspectiveDirScale,
  viewFiberWindow,
  MAX_DEG,
} from "./clipGrid.js";

/** Absolute caps (pipeline arrays specialize to active fit deg ≤ this). */
export const MAX_N = MAX_DEG + 1;
export const MAX_1D_N = 3 * MAX_DEG + 1;
export const MAX_COEFFS = MAX_N * MAX_N * MAX_N;
/** f32-friendly default tile (CPU Babbage uses 256 in f64). */
export const GPU_BABBAGE_TILE = 128;
const WG_SIZE = 64;

/**
 * Tile width for GPU Babbage. High D=3N makes f32 Δ^k / Newton blow up when the
 * coarse step h is large (Δ^k ~ h^k). Shrink tiles with degree so coarseStep
 * yields h=1, or skip Newton entirely (exact dens) at the highest degrees.
 */
export function gpuBabbageTile(deg, max1d) {
  const D = max1d | 0;
  const d = deg | 0;
  // N≥6 (D≥18): exact per pixel — order-18+ Newton in f32 is not usable.
  if (d >= 6) return Math.max(1, D);
  // N=5 (D=15): h=1 Babbage (tile = D+1 → span = D → coarseStep returns 1).
  if (d >= 5) return D + 1;
  return Math.max(D + 1, GPU_BABBAGE_TILE);
}

/** Per-degree sizes — WGSL locals/workgroup memory match fit deg (no max-N tax). */
export function degSizes(deg) {
  const d = Math.min(MAX_DEG, Math.max(1, deg | 0));
  const maxN = d + 1;
  const max1d = 3 * d;
  const max1dN = max1d + 1;
  return {
    deg: d,
    maxN,
    max1d,
    max1dN,
    maxCoeffs: maxN * maxN * maxN,
    seedDiffN: max1dN * max1dN,
  };
}

function makeBakeWgsl(sz) {
  const MAX_N = sz.maxN;
  const MAX_1D_N = sz.max1dN;
  const MAX_COEFFS = sz.maxCoeffs;
  const SEED_DIFF_N = sz.seedDiffN;
  return /* wgsl */ `
const MAX_N: u32 = ${MAX_N}u;
const MAX_1D_N: u32 = ${MAX_1D_N}u;
const MAX_COEFFS: u32 = ${MAX_COEFFS}u;
const WG_SIZE: u32 = ${WG_SIZE}u;
const SEED_DIFF_N: u32 = ${SEED_DIFF_N}u;
const CAP: f32 = 8.0;

struct Params {
  width: u32,
  height: u32,
  deg: u32,
  max1d: u32,
  tile: u32,
  xGrid0: i32,
  nTilesX: u32,
  _pad0: u32,
  half: f32,
  tMid: f32,
  tHw: f32,
  _pad1: f32,
  ro: vec3f,
  _pad2: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coeffs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outAtlas: array<f32>;

var<workgroup> seedSamples: array<f32, SEED_DIFF_N>;
var<workgroup> diffTable: array<f32, SEED_DIFF_N>;

fn coeffAt(idx: u32) -> f32 {
  if (idx >= MAX_COEFFS) { return 0.0; }
  return coeffs[idx];
}

fn evalMonomial3D(deg: u32, p: vec3f) -> f32 {
  let n = deg + 1u;
  var s = 0.0;
  var xp = 1.0;
  for (var i: u32 = 0u; i < MAX_N; i++) {
    if (i >= n) { break; }
    var yp = 1.0;
    for (var j: u32 = 0u; j < MAX_N; j++) {
      if (j >= n) { break; }
      var zp = 1.0;
      for (var k: u32 = 0u; k < MAX_N; k++) {
        if (k >= n) { break; }
        s += coeffAt(i + j * n + k * n * n) * xp * yp * zp;
        zp *= p.z;
      }
      yp *= p.y;
    }
    xp *= p.x;
  }
  return s;
}

fn rayDir(px: i32, py: i32) -> vec3f {
  let width = params.width;
  let height = params.height;
  let ndcX = -1.0 + (2.0 / f32(width)) * (f32(px) + 0.5);
  let ndcY = -1.0 + (2.0 / f32(height)) * (f32(py) + 0.5);
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  return vec3f(
    dot(params.m0.xyz, xy1),
    dot(params.m1.xyz, xy1),
    dot(params.m2.xyz, xy1),
  );
}

fn chebRoot(j: u32, M: u32) -> f32 {
  return cos(3.141592653589793 * (f32(j) + 0.5) / f32(M));
}

fn coarseStep(span: u32, D: u32) -> u32 {
  if (span <= D) { return 1u; }
  let hNeed = max(1u, (span + D - 1u) / D);
  var h = 1u;
  while (h < hNeed) { h = h * 2u; }
  return h;
}

fn writePixel(px: u32, py: u32, dens: ptr<function, array<f32, MAX_1D_N>>, nAlpha: u32) {
  let width = params.width;
  let height = params.height;
  let half = params.half;
  let ro = params.ro;
  let rd = rayDir(i32(px), i32(py));
  for (var j: u32 = 0u; j < MAX_1D_N; j++) {
    if (j >= nAlpha) { break; }
    let t = params.tMid + params.tHw * chebRoot(j, nAlpha);
    let p = ro + rd * t;
    let inside = abs(p.x) <= half && abs(p.y) <= half && abs(p.z) <= half;
    var v = select(0.0, (*dens)[j], inside);
    if (v != v) { v = 0.0; }
    v = clamp(v, -CAP, CAP);
    outAtlas[(j * height + py) * width + px] = v;
  }
}

fn exactDensAt(px: i32, py: i32, dens: ptr<function, array<f32, MAX_1D_N>>, deg: u32, nAlpha: u32) {
  let rd = rayDir(px, py);
  let ro = params.ro;
  for (var j: u32 = 0u; j < MAX_1D_N; j++) {
    if (j >= nAlpha) {
      (*dens)[j] = 0.0;
      continue;
    }
    let t = params.tMid + params.tHw * chebRoot(j, nAlpha);
    let p = ro + rd * t;
    (*dens)[j] = evalMonomial3D(deg, p);
  }
}

fn evalNewtonForward(base: u32, D: u32, t: f32) -> f32 {
  var acc = diffTable[base];
  var binom = 1.0;
  for (var k: u32 = 1u; k <= D; k++) {
    binom = binom * (t - f32(k - 1u)) / f32(k);
    acc += binom * diffTable[base + k];
  }
  return acc;
}

@compute @workgroup_size(${WG_SIZE}, 1, 1)
fn bakeBabbageMain(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid3: vec3u,
) {
  let lid = lid3.x;
  let tx = wid.x;
  let py = wid.y;
  let width = params.width;
  let height = params.height;
  if (py >= height || tx >= params.nTilesX) { return; }

  let deg = params.deg;
  let max1d = params.max1d;
  let D = max1d;
  let nAlpha = max1d + 1u;
  let tile = params.tile;

  let xBase = max(0i, params.xGrid0 + i32(tx * tile));
  let xEnd = min(i32(width), params.xGrid0 + i32((tx + 1u) * tile));
  if (xBase >= xEnd) { return; }

  let span = u32(xEnd - xBase - 1);

  // Narrow / single-pixel tiles: exact eval (no Babbage).
  if (span < D) {
    var densExact: array<f32, MAX_1D_N>;
    for (var px: u32 = u32(xBase) + lid; px < u32(xEnd); px += WG_SIZE) {
      exactDensAt(i32(px), i32(py), &densExact, deg, nAlpha);
      writePixel(px, py, &densExact, nAlpha);
    }
    return;
  }

  let h = coarseStep(span, D);
  let cover = D * h;
  // Must match Math.floor((span-cover)/2). WGSL i32 division truncates toward
  // zero (not floor); arithmetic >> 1 is signed floor-halve.
  var xOrigin = xBase + ((i32(span) - i32(cover)) >> 1);
  // Keep coarse seeds inside the atlas. Middle-out often places xOrigin < 0 on
  // the leftmost tile (and past width on the right); those out-of-frame NDC
  // monomial samples explode in f32 and poison the whole tile Δ table.
  if (xOrigin < 0) {
    xOrigin = 0;
  }
  if (xOrigin + i32(cover) >= i32(width)) {
    xOrigin = max(0i, i32(width) - i32(cover));
  }

  // Seed dens at middle-out coarse lattice → workgroup memory.
  if (lid <= D) {
    var densSeed: array<f32, MAX_1D_N>;
    exactDensAt(xOrigin + i32(lid * h), i32(py), &densSeed, deg, nAlpha);
    for (var j: u32 = 0u; j < MAX_1D_N; j++) {
      if (j >= nAlpha) { break; }
      seedSamples[j * MAX_1D_N + lid] = densSeed[j];
    }
  }
  workgroupBarrier();

  // Build forward-difference tables (one channel per thread when possible).
  if (lid < nAlpha) {
    let base = lid * MAX_1D_N;
    for (var i: u32 = 0u; i <= D; i++) {
      diffTable[base + i] = seedSamples[base + i];
    }
    for (var k: u32 = 1u; k <= D; k++) {
      var i: u32 = D;
      loop {
        if (i < k) { break; }
        diffTable[base + i] -= diffTable[base + i - 1u];
        if (i == 0u) { break; }
        i -= 1u;
      }
    }
  }
  workgroupBarrier();

  var dens: array<f32, MAX_1D_N>;
  for (var px: u32 = u32(xBase) + lid; px < u32(xEnd); px += WG_SIZE) {
    let t = (f32(i32(px) - xOrigin)) / f32(h);
    for (var j: u32 = 0u; j < MAX_1D_N; j++) {
      if (j >= nAlpha) {
        dens[j] = 0.0;
        continue;
      }
      dens[j] = evalNewtonForward(j * MAX_1D_N, D, t);
    }
    writePixel(px, py, &dens, nAlpha);
  }
}
`;
}

function makeMarchWgsl(sz) {
  const MAX_1D_N = sz.max1dN;
  return /* wgsl */ `
const MAX_1D_N: u32 = ${MAX_1D_N}u;

struct DrawParams {
  gridW: u32,
  gridH: u32,
  fbW: u32,
  fbH: u32,
  nAlpha: u32,
  max1d: u32,
  steps: u32,
  _p0: u32,
  half: f32,
  scale: f32,
  tMid: f32,
  tHw: f32,
  ro: vec3f,
  _p3: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  absorb: vec4f,
  emit: vec4f,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> atlas: array<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  // Fullscreen triangle
  var p = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}

fn gammaAt(px: i32, py: i32, k: u32) -> f32 {
  let w = draw.gridW;
  let h = draw.gridH;
  let x = clamp(px, 0, i32(w) - 1);
  let y = clamp(py, 0, i32(h) - 1);
  return atlas[(k * h + u32(y)) * w + u32(x)];
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  // WebGPU framebuffer origin is top-left; bake atlas py=0 is NDC y=-1 (bottom).
  let fbW = f32(draw.fbW);
  let fbH = f32(draw.fbH);
  let ndcX = -1.0 + 2.0 * in.pos.x / fbW;
  let ndcY = 1.0 - 2.0 * in.pos.y / fbH;

  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(
    dot(draw.m0.xyz, xy1),
    dot(draw.m1.xyz, xy1),
    dot(draw.m2.xyz, xy1),
  );
  let ro = draw.ro;
  let half = draw.half;

  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB);
  let tmax = max(tA, tB);
  var tEnter = max(max(tmin.x, tmin.y), tmin.z);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  tEnter = max(tEnter, 0.0);
  if (!(tExit > tEnter + 1e-6)) {
    return vec4f(0.0);
  }

  let tMid = draw.tMid;
  let tHw = draw.tHw;
  if (tHw < 1e-8) { return vec4f(0.0); }

  let fx = (ndcX + 1.0) * 0.5 * f32(draw.gridW) - 0.5;
  let fy = (ndcY + 1.0) * 0.5 * f32(draw.gridH) - 0.5;
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let x1 = x0 + 1;
  let y1 = y0 + 1;

  var densSamp: array<f32, MAX_1D_N>;
  let nA = draw.nAlpha;
  for (var k: u32 = 0u; k < MAX_1D_N; k++) {
    if (k >= nA) {
      densSamp[k] = 0.0;
      continue;
    }
    let g00 = gammaAt(x0, y0, k);
    let g10 = gammaAt(x1, y0, k);
    let g01 = gammaAt(x0, y1, k);
    let g11 = gammaAt(x1, y1, k);
    densSamp[k] = mix(mix(g00, g10, tx), mix(g01, g11, tx), ty);
  }

  var steps = draw.steps;
  if (steps < 8u) { steps = 8u; }
  if (steps > 96u) { steps = 96u; }
  let dt = (tExit - tEnter) / f32(steps);
  let ds = length(rd) * dt;

  var rgb = vec3f(0.0);
  var T = 1.0;
  var s = tEnter + 0.5 * dt;
  let absorbCol = draw.absorb.xyz;
  let emitCol = draw.emit.xyz;

  for (var i: u32 = 0u; i < 96u; i++) {
    if (i >= steps) { break; }
    if (T < 0.002) { break; }

    let u = (s - tMid) / tHw;
    var dval = densSamp[0];
    if (nA > 1u) {
      let invM = 1.0 / f32(nA);
      let uFirst = cos(3.141592653589793 * 0.5 * invM);
      let uLast = cos(3.141592653589793 * (f32(nA) - 0.5) * invM);
      if (u >= uFirst) {
        dval = densSamp[0];
      } else if (u <= uLast) {
        dval = densSamp[nA - 1u];
      } else {
        var found = false;
        for (var j: u32 = 0u; j < MAX_1D_N; j++) {
          if (j + 1u >= nA) { break; }
          let u0 = cos(3.141592653589793 * (f32(j) + 0.5) * invM);
          let u1 = cos(3.141592653589793 * (f32(j) + 1.5) * invM);
          if (u <= u0 && u >= u1) {
            let tt = (u0 - u) / max(u0 - u1, 1e-12);
            dval = mix(densSamp[j], densSamp[j + 1u], clamp(tt, 0.0, 1.0));
            found = true;
            break;
          }
        }
        if (!found) { dval = densSamp[nA - 1u]; }
      }
    }

    let p = ro + rd * s;
    if (abs(p.x) > half || abs(p.y) > half || abs(p.z) > half) {
      dval = 0.0;
    }
    if (dval != dval) { dval = 0.0; }
    dval = clamp(dval, -4.0, 8.0);

    var sigma = max(0.0, draw.scale * dval);
    sigma = min(sigma, 40.0);
    let absorb = exp(-sigma * ds);
    let opacity = 1.0 - absorb;
    rgb += T * opacity * (emitCol * sigma + absorbCol * 0.15);
    T *= absorb;
    s += dt;
  }

  let a = 1.0 - T;
  if (a < 0.001) { return vec4f(0.0); }
  // Canvas alphaMode is premultiplied.
  return vec4f(rgb * a, a);
}
`;
}

/** @type {GPUDevice | null} */
let device = null;
/** Active pipeline specialization degree (null until built). */
let pipelineDeg = null;
/** @type {GPUComputePipeline | null} */
let bakePipeline = null;
/** @type {GPURenderPipeline | null} */
let marchPipeline = null;
/** @type {GPUBuffer | null} */
let coeffBuf = null;
/** @type {GPUBuffer | null} */
let bakeParamBuf = null;
/** @type {GPUBuffer | null} */
let drawParamBuf = null;
/** @type {GPUBuffer | null} */
let atlasFront = null; // march reads
/** @type {GPUBuffer | null} */
let atlasBack = null; // bake writes
let atlasCapacity = 0;
/** @type {GPUBindGroup | null} */
let marchBindGroup = null;

/** @type {HTMLCanvasElement | null} */
let canvas = null;
/** @type {GPUCanvasContext | null} */
let ctx = null;
let canvasFormat = "bgra8unorm";
let initPromise = null;
let initFailed = false;

/** Last resident bake metadata (matches atlasFront). */
let resident = null;

/** Pack Bake Params (must match BAKE_WGSL Params; size 112, buffer 128). */
function packBakeParams(width, height, deg, max1d, tile, xGrid0, nTilesX, half, tMid, tHw, ro, M) {
  const buf = new ArrayBuffer(128);
  const u32 = new Uint32Array(buf, 0, 8);
  u32[0] = width;
  u32[1] = height;
  u32[2] = deg;
  u32[3] = max1d;
  u32[4] = tile;
  u32[6] = nTilesX;
  u32[7] = 0;
  new Int32Array(buf, 0, 8)[5] = xGrid0 | 0;
  const f32 = new Float32Array(buf, 32);
  f32[0] = half;
  f32[1] = tMid;
  f32[2] = tHw;
  f32[3] = 0;
  f32[4] = ro[0];
  f32[5] = ro[1];
  f32[6] = ro[2];
  f32[7] = 0;
  f32[8] = M[0];
  f32[9] = M[1];
  f32[10] = M[2];
  f32[11] = 0;
  f32[12] = M[3];
  f32[13] = M[4];
  f32[14] = M[5];
  f32[15] = 0;
  f32[16] = M[6];
  f32[17] = M[7];
  f32[18] = M[8];
  f32[19] = 0;
  return buf;
}

function packDrawParams(state, fbW, fbH, scale, steps, absorb, emit) {
  // Must match DrawParams in MARCH_WGSL (WebGPU uniform layout). Size ≥ 144, use 256.
  const buf = new ArrayBuffer(256);
  const u32 = new Uint32Array(buf, 0, 8);
  u32[0] = state.width;
  u32[1] = state.height;
  u32[2] = fbW;
  u32[3] = fbH;
  u32[4] = state.nAlpha;
  u32[5] = state.max1d;
  u32[6] = steps;
  u32[7] = 0;
  const f32 = new Float32Array(buf, 32);
  f32[0] = state.half;
  f32[1] = scale;
  f32[2] = state.tMid ?? 0;
  f32[3] = state.tHw ?? 1;
  f32[4] = state.ro[0];
  f32[5] = state.ro[1];
  f32[6] = state.ro[2];
  f32[7] = 0;
  const M = state.M;
  f32[8] = M[0];
  f32[9] = M[1];
  f32[10] = M[2];
  f32[11] = 0;
  f32[12] = M[3];
  f32[13] = M[4];
  f32[14] = M[5];
  f32[15] = 0;
  f32[16] = M[6];
  f32[17] = M[7];
  f32[18] = M[8];
  f32[19] = 0;
  f32[20] = absorb[0];
  f32[21] = absorb[1];
  f32[22] = absorb[2];
  f32[23] = 1;
  f32[24] = emit[0];
  f32[25] = emit[1];
  f32[26] = emit[2];
  f32[27] = 1;
  return buf;
}

export function isClipBakeGpuReady() {
  return Boolean(device && bakePipeline && marchPipeline);
}

export function hasResidentAtlas() {
  return Boolean(resident && atlasFront && marchBindGroup);
}

/** True when WebGPU clip pipelines + canvas are ready for a per-frame bake+march. */
export function isClipMarchReady() {
  return Boolean(device && bakePipeline && marchPipeline && ctx);
}

export function clipBakeGpuStatus() {
  if (isClipBakeGpuReady()) return "ready";
  if (initFailed) return "unavailable";
  if (initPromise) return "init";
  return "idle";
}

/**
 * Create/attach the WebGPU canvas inside the viewport (under HUD, over Three).
 */
export function attachClipGpuCanvas(viewportEl) {
  if (canvas) return canvas;
  canvas = document.createElement("canvas");
  canvas.className = "clip-gpu";
  canvas.style.cssText =
    "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:1;display:none;";
  viewportEl.appendChild(canvas);
  return canvas;
}

export function setClipGpuCanvasVisible(visible) {
  if (!canvas) return;
  canvas.style.display = visible ? "block" : "none";
}

export function resizeClipGpuCanvas(pixelW, pixelH) {
  if (!canvas || !device) return false;
  if (!ctx) {
    ctx = canvas.getContext("webgpu");
    if (!ctx) return false;
  }
  const w = Math.max(1, pixelW | 0);
  const h = Math.max(1, pixelH | 0);
  // Assigning canvas.width/height clears the drawing buffer — only do it on change.
  const needResize = canvas.width !== w || canvas.height !== h || !canvas._clipConfigured;
  if (!needResize) return false;
  canvas.width = w;
  canvas.height = h;
  ctx.configure({
    device,
    format: canvasFormat,
    alphaMode: "premultiplied",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  canvas._clipConfigured = true;
  return true; // caller must redraw
}

export async function initClipBakeGpu(viewportEl) {
  if (isClipBakeGpuReady()) return true;
  if (initFailed) return false;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      if (!navigator.gpu) {
        initFailed = true;
        return false;
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        initFailed = true;
        return false;
      }
      device = await adapter.requestDevice();
      device.lost.then(() => {
        device = null;
        bakePipeline = null;
        marchPipeline = null;
        pipelineDeg = null;
        initFailed = true;
      });

      coeffBuf = device.createBuffer({
        size: MAX_COEFFS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      bakeParamBuf = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      drawParamBuf = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // Specialize to a default degree; rebake/upload rebuilds when fit deg changes.
      await ensurePipelinesForDegree(4);

      if (viewportEl) attachClipGpuCanvas(viewportEl);
      if (canvas) {
        ctx = canvas.getContext("webgpu");
      }
      return true;
    } catch (e) {
      console.warn("[clipBakeGpu] init failed", e);
      initFailed = true;
      device = null;
      bakePipeline = null;
      marchPipeline = null;
      pipelineDeg = null;
      return false;
    }
  })();

  return initPromise;
}

/**
 * Rebuild bake/march pipelines for the active fit degree so WGSL locals and
 * workgroup arrays match N (no idle max-deg tax).
 */
export async function ensurePipelinesForDegree(deg) {
  if (!device) return false;
  const sz = degSizes(deg);
  if (pipelineDeg === sz.deg && bakePipeline && marchPipeline) return true;

  const bakeMod = device.createShaderModule({ code: makeBakeWgsl(sz) });
  device.pushErrorScope("validation");
  const nextBake = device.createComputePipeline({
    layout: "auto",
    compute: { module: bakeMod, entryPoint: "bakeBabbageMain" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`bake pipeline deg=${sz.deg}: ${err.message}`);
  }

  const marchMod = device.createShaderModule({ code: makeMarchWgsl(sz) });
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  device.pushErrorScope("validation");
  const nextMarch = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: marchMod, entryPoint: "vsMain" },
    fragment: {
      module: marchMod,
      entryPoint: "fsMain",
      targets: [
        {
          format: canvasFormat,
          blend: {
            color: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`march pipeline deg=${sz.deg}: ${err.message}`);
  }

  bakePipeline = nextBake;
  marchPipeline = nextMarch;
  pipelineDeg = sz.deg;
  marchBindGroup = null;
  bakeBindGroup = null;
  uploadedCoeffDeg = -1;
  uploadedCoeffRef = null;
  if (atlasFront) bindMarchToFront();
  return true;
}

function storageBufferBudget() {
  if (!device) return 64 * 1024 * 1024;
  const a = device.limits.maxStorageBufferBindingSize || 1 << 27;
  const b = device.limits.maxBufferSize || a;
  // Leave headroom; huge uploads are slow and some backends flake near the limit.
  return Math.min(a, b, 96 * 1024 * 1024);
}

/**
 * Grow the dens atlas scratch buffer (bake writes, march reads same frame).
 */
function ensureAtlasPair(byteSize) {
  const aligned = Math.max(256, Math.ceil(byteSize / 256) * 256);
  if (atlasFront && atlasCapacity >= aligned) return;

  if (aligned > storageBufferBudget()) {
    throw new Error(
      `atlas ${aligned}B exceeds GPU storage budget ${storageBufferBudget()}B`,
    );
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const next = device.createBuffer({ size: aligned, usage });
  const oldFront = atlasFront;
  const oldBack = atlasBack;
  atlasFront = next;
  atlasBack = null;
  atlasCapacity = aligned;
  invalidateAtlasBindGroups();
  const retire = [oldFront, oldBack].filter(Boolean);
  if (retire.length) {
    void device.queue.onSubmittedWorkDone().then(() => {
      for (const b of retire) {
        try {
          b.destroy();
        } catch (_) {
          /* ignore */
        }
      }
    });
  }
}

/** @type {GPUBindGroup | null} */
let bakeBindGroup = null;
/** Cached monomial upload (skip rewrite when fit unchanged). */
let uploadedCoeffDeg = -1;
let uploadedCoeffRef = null;

function bindMarchToFront() {
  if (!marchPipeline || !atlasFront) return;
  marchBindGroup = device.createBindGroup({
    layout: marchPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: drawParamBuf } },
      { binding: 1, resource: { buffer: atlasFront } },
    ],
  });
}

function bindBakeToFront() {
  if (!bakePipeline || !atlasFront) return null;
  if (bakeBindGroup) return bakeBindGroup;
  bakeBindGroup = device.createBindGroup({
    layout: bakePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bakeParamBuf } },
      { binding: 1, resource: { buffer: coeffBuf } },
      { binding: 2, resource: { buffer: atlasFront } },
    ],
  });
  return bakeBindGroup;
}

function invalidateAtlasBindGroups() {
  marchBindGroup = null;
  bakeBindGroup = null;
}

/**
 * Per-frame GPU path: tile Babbage dens into the atlas, then march — one submit,
 * no await. Pipelines must already match `deg` ({@link ensurePipelinesForDegree}).
 */
export function renderClipFrameGpu({
  worldMono,
  deg,
  camera,
  width,
  height,
  half,
  fbW,
  fbH,
  scale,
  steps,
  absorb = [0.15, 0.25, 0.45],
  emit = [0.55, 0.75, 1.0],
}) {
  if (!device || !bakePipeline || !marchPipeline || !ctx || !worldMono) return false;
  const sz = degSizes(deg);
  if (pipelineDeg !== sz.deg) return false;

  const d = sz.deg;
  const max1d = sz.max1d;
  const nAlpha = sz.max1dN;
  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const M = ndcToDirMatrix(camera, sx, sy);
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;
  const { tMid, tHw } = viewFiberWindow(ro, h, M);

  const tile = gpuBabbageTile(d, max1d);
  const nTilesX = Math.max(1, Math.ceil(width / tile));
  const xGrid0 = -Math.floor((nTilesX * tile - width) / 2);

  const outBytes = width * height * nAlpha * 4;
  const prevAtlas = atlasFront;
  ensureAtlasPair(outBytes);
  if (!atlasFront) return false;
  if (atlasFront !== prevAtlas) invalidateAtlasBindGroups();

  if (uploadedCoeffRef !== worldMono || uploadedCoeffDeg !== d) {
    const coeffs = new Float32Array(sz.maxCoeffs);
    const n = d + 1;
    for (let i = 0; i <= d; i++) {
      for (let j = 0; j <= d; j++) {
        for (let k = 0; k <= d; k++) {
          const idx = i + j * n + k * n * n;
          coeffs[idx] = worldMono[idx] || 0;
        }
      }
    }
    device.queue.writeBuffer(coeffBuf, 0, coeffs);
    uploadedCoeffRef = worldMono;
    uploadedCoeffDeg = d;
  }

  const state = {
    width,
    height,
    nAlpha,
    max1d,
    deg: d,
    M,
    ro,
    half: h,
    tMid,
    tHw,
    sx,
    sy,
    tile,
    method: tile <= max1d ? "gpu-exact-dens" : "gpu-babbage-newton-t-mid",
    gpuResident: true,
  };

  device.queue.writeBuffer(
    bakeParamBuf,
    0,
    packBakeParams(width, height, d, max1d, tile, xGrid0, nTilesX, h, tMid, tHw, ro, M),
  );
  device.queue.writeBuffer(
    drawParamBuf,
    0,
    packDrawParams(state, fbW, fbH, scale, steps, absorb, emit),
  );

  const bakeBind = bindBakeToFront();
  if (!marchBindGroup) bindMarchToFront();
  if (!bakeBind || !marchBindGroup) return false;

  resizeClipGpuCanvas(fbW, fbH);
  const enc = device.createCommandEncoder();

  {
    const pass = enc.beginComputePass();
    pass.setPipeline(bakePipeline);
    pass.setBindGroup(0, bakeBind);
    pass.dispatchWorkgroups(nTilesX, height, 1);
    pass.end();
  }

  {
    const view = ctx.getCurrentTexture().createView();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(marchPipeline);
    pass.setBindGroup(0, marchBindGroup);
    pass.draw(3);
    pass.end();
  }

  device.queue.submit([enc.finish()]);
  resident = state;
  return true;
}

/**
 * March-only (legacy). Prefer {@link renderClipFrameGpu} for the live path.
 */
export function renderClipGridGpu({
  fbW,
  fbH,
  scale,
  steps,
  absorb = [0.15, 0.25, 0.45],
  emit = [0.55, 0.75, 1.0],
}) {
  if (!device || !marchPipeline || !ctx || !resident || !atlasFront || !marchBindGroup) {
    return false;
  }

  resizeClipGpuCanvas(fbW, fbH);
  device.queue.writeBuffer(
    drawParamBuf,
    0,
    packDrawParams(resident, fbW, fbH, scale, steps, absorb, emit),
  );

  const enc = device.createCommandEncoder();
  const view = ctx.getCurrentTexture().createView();
  const pass = enc.beginRenderPass({
    colorAttachments: [
      {
        view,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(marchPipeline);
  pass.setBindGroup(0, marchBindGroup);
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
  return true;
}
