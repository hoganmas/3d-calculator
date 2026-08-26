/**
 * WebGPU clip-grid path (golden): Chebyshev IDCT volume + Beer march.
 * Bake is view-independent (fit / coeff change). See
 * research/poly/notes/cheb-idct-volume.md.
 */

import { ndcToDirMatrix, perspectiveDirScale, MAX_DEG } from "./clipGrid.js";
import { idctCheb3D } from "./chebIdct.js";

const MAX_N = MAX_DEG + 1;
export const MAX_COEFFS = MAX_N * MAX_N * MAX_N;

function volumeGridM(deg) {
  // Exact Chebyshev interpolant nodes (M = N+1); buffer grows with upload.
  return Math.max(2, (deg | 0) + 1);
}

function makeMarchWgsl() {
  return /* wgsl */ `
struct DrawParams {
  fbW: u32,
  fbH: u32,
  gridM: u32,
  steps: u32,
  half: f32,
  scale: f32,
  _pad0: f32,
  _pad1: f32,
  ro: vec3f,
  _p1: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  absorb: vec4f,
  emit: vec4f,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;

struct VSOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}

fn densAt(ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(draw.gridM);
  let x = clamp(ix, 0, M - 1);
  let y = clamp(iy, 0, M - 1);
  let z = clamp(iz, 0, M - 1);
  return volume[u32(x) + u32(y) * draw.gridM + u32(z) * draw.gridM * draw.gridM];
}

/** ξ ∈ [-1,1] → continuous Chebyshev-root index. */
fn chebIndex(xi: f32) -> f32 {
  let x = clamp(xi, -1.0, 1.0);
  return f32(draw.gridM) / 3.141592653589793 * acos(x) - 0.5;
}

fn sampleVolume(p: vec3f) -> f32 {
  let half = draw.half;
  let xi = p / half;
  if (abs(xi.x) > 1.0 || abs(xi.y) > 1.0 || abs(xi.z) > 1.0) {
    return 0.0;
  }
  let fx = chebIndex(xi.x);
  let fy = chebIndex(xi.y);
  let fz = chebIndex(xi.z);
  let x0 = i32(floor(fx));
  let y0 = i32(floor(fy));
  let z0 = i32(floor(fz));
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let tz = clamp(fz - f32(z0), 0.0, 1.0);
  let c000 = densAt(x0, y0, z0);
  let c100 = densAt(x0 + 1, y0, z0);
  let c010 = densAt(x0, y0 + 1, z0);
  let c110 = densAt(x0 + 1, y0 + 1, z0);
  let c001 = densAt(x0, y0, z0 + 1);
  let c101 = densAt(x0 + 1, y0, z0 + 1);
  let c011 = densAt(x0, y0 + 1, z0 + 1);
  let c111 = densAt(x0 + 1, y0 + 1, z0 + 1);
  let c00 = mix(c000, c100, tx);
  let c10 = mix(c010, c110, tx);
  let c01 = mix(c001, c101, tx);
  let c11 = mix(c011, c111, tx);
  let c0 = mix(c00, c10, ty);
  let c1 = mix(c01, c11, ty);
  return mix(c0, c1, tz);
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
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

    let p = ro + rd * s;
    var dval = sampleVolume(p);
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
  return vec4f(rgb * a, a);
}
`;
}

/** @type {GPUDevice | null} */
let device = null;
/** @type {GPUCanvasContext | null} */
let ctx = null;
/** @type {HTMLCanvasElement | null} */
let canvas = null;
let canvasFormat = "bgra8unorm";
/** @type {GPURenderPipeline | null} */
let marchPipeline = null;
/** @type {GPUBindGroup | null} */
let marchBindGroup = null;
/** @type {GPUBuffer | null} */
let drawParamBuf = null;
/** @type {GPUBuffer | null} */
let volumeBuf = null;
let volumeCapacity = 0;
let volumeM = 0;
let uploadedVolumeRef = null;

let initFailed = false;
/** @type {Promise<boolean> | null} */
let initPromise = null;

let timestampsSupported = false;
/** @type {GPUQuerySet | null} */
let stampQuerySet = null;
/** @type {GPUBuffer | null} */
let stampResolveBuf = null;
/** @type {GPUBuffer | null} */
let stampReadBuf = null;
let stampReadPending = false;

let profileBakeMs = 0;
let profileMarchMs = 0;
let profileMarchFbW = 0;
let profileMarchFbH = 0;
let profilePresentWallMs = 0;
let profilePresentIntervalMs = 0;
let lastPresentAt = 0;
let profileMethod = "";
let profileGridM = 0;

function packDrawParams(fbW, fbH, gridM, steps, half, scale, ro, M, absorb, emit) {
  const buf = new ArrayBuffer(256);
  const u32 = new Uint32Array(buf, 0, 4);
  u32[0] = fbW;
  u32[1] = fbH;
  u32[2] = gridM;
  u32[3] = steps;
  const f32 = new Float32Array(buf, 16);
  f32[0] = half;
  f32[1] = scale;
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
  f32[20] = absorb[0];
  f32[21] = absorb[1];
  f32[22] = absorb[2];
  f32[24] = emit[0];
  f32[25] = emit[1];
  f32[26] = emit[2];
  return buf;
}

export function isClipBakeGpuReady() {
  return Boolean(device && marchPipeline);
}

export function isClipMarchReady() {
  return Boolean(device && marchPipeline && ctx && volumeBuf && volumeM > 0);
}

function noteGpuPresent(submitWallAt) {
  const now = performance.now();
  profilePresentWallMs = profilePresentWallMs * 0.85 + (now - submitWallAt) * 0.15;
  if (lastPresentAt > 0) {
    profilePresentIntervalMs =
      profilePresentIntervalMs * 0.85 + (now - lastPresentAt) * 0.15;
  } else {
    profilePresentIntervalMs = now - submitWallAt;
  }
  lastPresentAt = now;
}

export function getClipGpuProfile() {
  return {
    seedMs: profileBakeMs,
    fillMs: 0,
    marchMs: profileMarchMs,
    marchFbW: profileMarchFbW,
    marchFbH: profileMarchFbH,
    presentWallMs: profilePresentWallMs,
    presentIntervalMs: profilePresentIntervalMs,
    lastPresentAt,
    method: profileMethod,
    tile: profileGridM,
    nTilesX: 1,
    timestamps: timestampsSupported,
  };
}

export function resetClipGpuProfile() {
  profileBakeMs = 0;
  profileMarchMs = 0;
  profileMarchFbW = 0;
  profileMarchFbH = 0;
}

function attachClipGpuCanvas(viewportEl) {
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
  return true;
}

function ensureVolumeBuf(floatCount) {
  const aligned = Math.max(256, Math.ceil((floatCount * 4) / 256) * 256);
  if (volumeBuf && volumeCapacity >= aligned) return;
  const old = volumeBuf;
  volumeBuf = device.createBuffer({
    size: aligned,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  volumeCapacity = aligned;
  marchBindGroup = null;
  if (old) {
    void device.queue.onSubmittedWorkDone().then(() => {
      try {
        old.destroy();
      } catch (_) {
        /* ignore */
      }
    });
  }
}

function bindMarch() {
  if (!marchPipeline || !volumeBuf || !drawParamBuf) return;
  marchBindGroup = device.createBindGroup({
    layout: marchPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: drawParamBuf } },
      { binding: 1, resource: { buffer: volumeBuf } },
    ],
  });
}

/**
 * CPU separable IDCT → upload dens volume. Call on fit / coeff change only.
 * @returns {{ M: number, bakeMs: number, dens: Float32Array }}
 */
export function uploadChebVolume(cheb, deg) {
  if (!device || !cheb) return null;
  const t0 = performance.now();
  const M = volumeGridM(deg);
  const { dens } = idctCheb3D(cheb, deg, M);
  ensureVolumeBuf(dens.length);
  device.queue.writeBuffer(volumeBuf, 0, dens);
  volumeM = M;
  uploadedVolumeRef = cheb;
  profileBakeMs = profileBakeMs * 0.5 + (performance.now() - t0) * 0.5;
  profileGridM = M;
  profileMethod = "cpu-idct-volume";
  marchBindGroup = null;
  bindMarch();
  return { M, bakeMs: performance.now() - t0, dens };
}

/** True if current GPU volume matches this cheb buffer reference. */
export function hasUploadedVolume(cheb) {
  return uploadedVolumeRef === cheb && volumeM > 0;
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
      timestampsSupported = adapter.features.has("timestamp-query");
      const requiredFeatures = [];
      if (timestampsSupported) requiredFeatures.push("timestamp-query");
      device = await adapter.requestDevice({ requiredFeatures });
      device.lost.then(() => {
        device = null;
        marchPipeline = null;
        initFailed = true;
      });

      if (timestampsSupported) {
        stampQuerySet = device.createQuerySet({ type: "timestamp", count: 2 });
        stampResolveBuf = device.createBuffer({
          size: 2 * 8,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        stampReadBuf = device.createBuffer({
          size: 2 * 8,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }

      drawParamBuf = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      ensureVolumeBuf(8 * 8 * 8);

      await ensurePipelinesForDegree(4);

      if (viewportEl) attachClipGpuCanvas(viewportEl);
      if (canvas) ctx = canvas.getContext("webgpu");
      return true;
    } catch (e) {
      console.warn("[clipBakeGpu] init failed", e);
      initFailed = true;
      device = null;
      marchPipeline = null;
      return false;
    }
  })();

  return initPromise;
}

export async function ensurePipelinesForDegree(_deg) {
  if (!device) return false;
  if (marchPipeline) return true;

  const marchMod = device.createShaderModule({ code: makeMarchWgsl() });
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
    if (err) throw new Error(`march pipeline: ${err.message}`);
  }
  marchPipeline = nextMarch;
  marchBindGroup = null;
  if (volumeBuf) bindMarch();
  return true;
}

function scheduleStampReadback() {
  if (!timestampsSupported || !stampReadBuf || stampReadPending) return;
  stampReadPending = true;
  stampReadBuf
    .mapAsync(GPUMapMode.READ)
    .then(() => {
      const stamps = new BigInt64Array(stampReadBuf.getMappedRange().slice(0));
      stampReadBuf.unmap();
      stampReadPending = false;
      const ns = (a, b) => Number(stamps[b] - stamps[a]) / 1e6;
      if (stamps[1] > stamps[0]) profileMarchMs = profileMarchMs * 0.7 + ns(0, 1) * 0.3;
    })
    .catch(() => {
      stampReadPending = false;
    });
}

/**
 * Per-frame march only (volume must already be uploaded).
 */
export function renderClipFrameGpu({
  camera,
  half,
  fbW,
  fbH,
  scale,
  steps,
  absorb = [0.15, 0.25, 0.45],
  emit = [0.55, 0.75, 1.0],
}) {
  if (!device || !marchPipeline || !ctx || !volumeBuf || volumeM < 2) return false;

  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const M = ndcToDirMatrix(camera, sx, sy);
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;

  if (!marchBindGroup) bindMarch();
  if (!marchBindGroup) return false;

  resizeClipGpuCanvas(fbW, fbH);
  const marchW = canvas?.width ?? fbW;
  const marchH = canvas?.height ?? fbH;
  profileMarchFbW = marchW;
  profileMarchFbH = marchH;
  profileMethod = "gpu-idct-volume";
  profileGridM = volumeM;

  device.queue.writeBuffer(
    drawParamBuf,
    0,
    packDrawParams(marchW, marchH, volumeM, steps, h, scale, ro, M, absorb, emit),
  );

  const enc = device.createCommandEncoder();
  const useStamps = timestampsSupported && stampQuerySet;
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
      ...(useStamps
        ? {
            timestampWrites: {
              querySet: stampQuerySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          }
        : {}),
    });
    pass.setPipeline(marchPipeline);
    pass.setBindGroup(0, marchBindGroup);
    pass.draw(3);
    pass.end();
  }

  if (useStamps && stampResolveBuf && stampReadBuf && !stampReadPending) {
    enc.resolveQuerySet(stampQuerySet, 0, 2, stampResolveBuf, 0);
    enc.copyBufferToBuffer(stampResolveBuf, 0, stampReadBuf, 0, 16);
  }

  const submitWallAt = performance.now();
  device.queue.submit([enc.finish()]);
  void device.queue.onSubmittedWorkDone().then(() => {
    noteGpuPresent(submitWallAt);
    if (useStamps) scheduleStampReadback();
  });
  return true;
}
