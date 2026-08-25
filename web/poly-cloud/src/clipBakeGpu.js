/**
 * WebGPU clip-grid path: bake γ atlas in a storage buffer (no CPU readback),
 * then fullscreen-march directly from that buffer onto a canvas.
 *
 * CPU/WebGL fallback remains in main.js via bakeClipGridFibers + DataTexture.
 */

import { ndcToDirMatrix, perspectiveDirScale, MAX_DEG } from "./clipGrid.js";

const MAX_N = MAX_DEG + 1;
const MAX_1D = 3 * MAX_DEG;
const MAX_1D_N = MAX_1D + 1;
const MAX_COEFFS = MAX_N * MAX_N * MAX_N;

const BAKE_WGSL = /* wgsl */ `
const MAX_DEG: u32 = ${MAX_DEG}u;
const MAX_N: u32 = ${MAX_N}u;
const MAX_1D_N: u32 = ${MAX_1D_N}u;
const MAX_COEFFS: u32 = ${MAX_COEFFS}u;

struct Params {
  width: u32,
  height: u32,
  deg: u32,
  max1d: u32,
  half: f32,
  _p0: f32,
  _p1: f32,
  _p2: f32,
  ro: vec3f,
  _p3: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coeffs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outAtlas: array<f32>;

fn coeffAt(idx: u32) -> f32 {
  if (idx >= MAX_COEFFS) { return 0.0; }
  return coeffs[idx];
}

fn mulLinear(poly: ptr<function, array<f32, MAX_1D_N>>, a0: f32, a1: f32, max1d: u32, outp: ptr<function, array<f32, MAX_1D_N>>) {
  for (var t: u32 = 0u; t < MAX_1D_N; t++) { (*outp)[t] = 0.0; }
  for (var t: u32 = 0u; t <= max1d; t++) {
    let v = (*poly)[t];
    if (v == 0.0) { continue; }
    (*outp)[t] += v * a0;
    if (t + 1u <= max1d) { (*outp)[t + 1u] += v * a1; }
  }
}

fn copyPoly(src: ptr<function, array<f32, MAX_1D_N>>, dst: ptr<function, array<f32, MAX_1D_N>>) {
  for (var t: u32 = 0u; t < MAX_1D_N; t++) { (*dst)[t] = (*src)[t]; }
}

fn clearPoly(p: ptr<function, array<f32, MAX_1D_N>>) {
  for (var t: u32 = 0u; t < MAX_1D_N; t++) { (*p)[t] = 0.0; }
}

@compute @workgroup_size(8, 8, 1)
fn bakeMain(@builtin(global_invocation_id) gid: vec3u) {
  let width = params.width;
  let height = params.height;
  let px = gid.x;
  let py = gid.y;
  if (px >= width || py >= height) { return; }

  let deg = params.deg;
  let max1d = params.max1d;
  let n = deg + 1u;
  let nAlpha = max1d + 1u;
  let half = params.half;
  let ro = params.ro;

  let ndcX = -1.0 + (2.0 / f32(width)) * (f32(px) + 0.5);
  let ndcY = -1.0 + (2.0 / f32(height)) * (f32(py) + 0.5);
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(
    dot(params.m0.xyz, xy1),
    dot(params.m1.xyz, xy1),
    dot(params.m2.xyz, xy1),
  );

  var t0 = -1e30;
  var t1 = 1e30;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB);
  let tmax = max(tA, tB);
  t0 = max(max(tmin.x, tmin.y), tmin.z);
  t1 = min(min(tmax.x, tmax.y), tmax.z);
  t0 = max(t0, 0.0);

  for (var k: u32 = 0u; k < nAlpha; k++) {
    outAtlas[(k * height + py) * width + px] = 0.0;
  }
  if (!(t1 > t0 + 1e-6)) { return; }

  let tMid = 0.5 * (t0 + t1);
  let tHw = 0.5 * (t1 - t0);
  if (tHw <= 1e-10) { return; }

  let P0 = ro + rd * tMid;
  let Du = rd * tHw;

  var zPow: array<f32, ${MAX_N * MAX_1D_N}>;
  var pk: array<f32, MAX_1D_N>;
  var tmp: array<f32, MAX_1D_N>;
  clearPoly(&pk);
  pk[0] = 1.0;
  for (var k: u32 = 0u; k <= deg; k++) {
    for (var m: u32 = 0u; m < MAX_1D_N; m++) {
      zPow[k * MAX_1D_N + m] = pk[m];
    }
    if (k == deg) { break; }
    mulLinear(&pk, P0.z, Du.z, max1d, &tmp);
    copyPoly(&tmp, &pk);
  }

  var gamma: array<f32, MAX_1D_N>;
  var si: array<f32, MAX_1D_N>;
  var row: array<f32, MAX_1D_N>;
  clearPoly(&gamma);

  for (var i: i32 = i32(deg); i >= 0; i--) {
    clearPoly(&si);
    for (var j: i32 = i32(deg); j >= 0; j--) {
      clearPoly(&row);
      let iu = u32(i);
      let ju = u32(j);
      for (var k: u32 = 0u; k <= deg; k++) {
        let c = coeffAt(iu + ju * n + k * n * n);
        if (abs(c) < 1e-20) { continue; }
        for (var m: u32 = 0u; m < MAX_1D_N; m++) {
          row[m] += c * zPow[k * MAX_1D_N + m];
        }
      }
      if (j < i32(deg)) {
        mulLinear(&si, P0.y, Du.y, max1d, &tmp);
        copyPoly(&tmp, &si);
      }
      for (var m: u32 = 0u; m < MAX_1D_N; m++) { si[m] += row[m]; }
    }
    if (i < i32(deg)) {
      mulLinear(&gamma, P0.x, Du.x, max1d, &tmp);
      copyPoly(&tmp, &gamma);
    }
    for (var m: u32 = 0u; m < MAX_1D_N; m++) { gamma[m] += si[m]; }
  }

  for (var k: u32 = 0u; k <= max1d; k++) {
    outAtlas[(k * height + py) * width + px] = gamma[k];
  }
}
`;

const MARCH_WGSL = /* wgsl */ `
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
  _p1: f32,
  _p2: f32,
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

  let tMid = 0.5 * (tEnter + tExit);
  let tHw = 0.5 * (tExit - tEnter);
  if (tHw < 1e-8) { return vec4f(0.0); }

  let fx = (ndcX + 1.0) * 0.5 * f32(draw.gridW) - 0.5;
  let fy = (ndcY + 1.0) * 0.5 * f32(draw.gridH) - 0.5;
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let x1 = x0 + 1;
  let y1 = y0 + 1;

  var gamma: array<f32, MAX_1D_N>;
  let M = draw.max1d;
  for (var k: u32 = 0u; k < MAX_1D_N; k++) {
    if (k > M) {
      gamma[k] = 0.0;
      continue;
    }
    let g00 = gammaAt(x0, y0, k);
    let g10 = gammaAt(x1, y0, k);
    let g01 = gammaAt(x0, y1, k);
    let g11 = gammaAt(x1, y1, k);
    gamma[k] = mix(mix(g00, g10, tx), mix(g01, g11, tx), ty);
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
    var dens = 0.0;
    for (var k: i32 = 24; k >= 0; k--) {
      if (u32(k) > M) { continue; }
      dens = dens * u + gamma[u32(k)];
    }

    let sigma = max(0.0, draw.scale * dens);
    let absorb = exp(-sigma * ds);
    let opacity = 1.0 - absorb;
    rgb += T * opacity * (emitCol * sigma + absorbCol * 0.15);
    T *= absorb;
    s += dt;
  }

  let a = 1.0 - T;
  if (a < 0.001) { return vec4f(0.0); }
  // Match WebGL path: rgb is already emission accumulation; blend uses One / 1-srcA.
  return vec4f(rgb, a);
}
`;

/** @type {GPUDevice | null} */
let device = null;
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

function packBakeParams(width, height, deg, max1d, half, ro, M) {
  const u32 = new Uint32Array(4);
  u32[0] = width;
  u32[1] = height;
  u32[2] = deg;
  u32[3] = max1d;
  const f32 = new Float32Array(20);
  f32[0] = half;
  f32[4] = ro[0];
  f32[5] = ro[1];
  f32[6] = ro[2];
  f32[8] = M[0];
  f32[9] = M[1];
  f32[10] = M[2];
  f32[12] = M[3];
  f32[13] = M[4];
  f32[14] = M[5];
  f32[16] = M[6];
  f32[17] = M[7];
  f32[18] = M[8];
  const buf = new ArrayBuffer(96);
  new Uint32Array(buf, 0, 4).set(u32);
  new Float32Array(buf, 16).set(f32);
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
  f32[2] = 0;
  f32[3] = 0;
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
  return Boolean(resident && atlasFront);
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
  if (!canvas || !device) return;
  if (!ctx) {
    ctx = canvas.getContext("webgpu");
    if (!ctx) return;
  }
  const w = Math.max(1, pixelW | 0);
  const h = Math.max(1, pixelH | 0);
  const needConfigure = canvas.width !== w || canvas.height !== h || !canvas._clipConfigured;
  canvas.width = w;
  canvas.height = h;
  if (needConfigure) {
    ctx.configure({
      device,
      format: canvasFormat,
      // Transparent so Three's box helper / clear show through.
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    canvas._clipConfigured = true;
  }
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
        initFailed = true;
      });

      const bakeMod = device.createShaderModule({ code: BAKE_WGSL });
      device.pushErrorScope("validation");
      bakePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: bakeMod, entryPoint: "bakeMain" },
      });
      {
        const err = await device.popErrorScope();
        if (err) throw new Error(`bake pipeline: ${err.message}`);
      }

      const marchMod = device.createShaderModule({ code: MARCH_WGSL });
      canvasFormat = navigator.gpu.getPreferredCanvasFormat();
      device.pushErrorScope("validation");
      marchPipeline = device.createRenderPipeline({
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
        if (err) throw new Error(`march pipeline: ${err.message}`);
      }
      coeffBuf = device.createBuffer({
        size: MAX_COEFFS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      bakeParamBuf = device.createBuffer({
        size: 96,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      drawParamBuf = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

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
      return false;
    }
  })();

  return initPromise;
}

function ensureAtlasPair(byteSize) {
  if (atlasFront && atlasBack && atlasCapacity >= byteSize) return;
  atlasFront?.destroy();
  atlasBack?.destroy();
  atlasCapacity = byteSize;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  atlasFront = device.createBuffer({ size: byteSize, usage });
  atlasBack = device.createBuffer({ size: byteSize, usage });
  marchBindGroup = null;
}

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

/**
 * Bake into the back buffer, then swap → front. March always reads front, so a
 * concurrent frame never samples a half-written atlas (that caused line flicker).
 */
export async function bakeClipGridFibersGpu(worldMono, deg, camera, width, height, half) {
  const ok = await initClipBakeGpu();
  if (!ok || !device || !bakePipeline) return null;

  const d = Math.min(MAX_DEG, Math.max(1, deg | 0));
  const max1d = 3 * d;
  const nAlpha = max1d + 1;
  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const M = ndcToDirMatrix(camera, sx, sy);
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;

  const outBytes = width * height * nAlpha * 4;
  ensureAtlasPair(outBytes);

  const coeffs = new Float32Array(MAX_COEFFS);
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
  device.queue.writeBuffer(
    bakeParamBuf,
    0,
    packBakeParams(width, height, d, max1d, h, ro, M),
  );

  const bindGroup = device.createBindGroup({
    layout: bakePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bakeParamBuf } },
      { binding: 1, resource: { buffer: coeffBuf } },
      { binding: 2, resource: { buffer: atlasBack } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(bakePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Publish: swap so march sees the completed atlas atomically w.r.t. JS frames.
  const tmp = atlasFront;
  atlasFront = atlasBack;
  atlasBack = tmp;
  bindMarchToFront();

  resident = {
    gpuResident: true,
    width,
    height,
    nAlpha,
    max1d,
    deg: d,
    M,
    ro,
    half: h,
    sx,
    sy,
  };
  return resident;
}

/**
 * March resident (front) atlas to the WebGPU canvas.
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
