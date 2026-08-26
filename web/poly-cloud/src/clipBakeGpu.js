/**
 * WebGPU volume march: IDCT dens grids + iso manifolds + multi-layer Beer.
 */
import { ndcToDirMatrix, perspectiveDirScale } from "./clipGrid.js";

export const MAX_DENS_LAYERS = 8;

/** Classic DrawParams layout (matches 872b141). volBase in the _p1 slot. */
function makeIsoWgsl() {
  return /* wgsl */ `
struct DrawParams {
  fbW: u32,
  fbH: u32,
  gridM: u32,
  steps: u32,
  half: f32,
  scale: f32,
  isoLevel: f32,
  shadeMode: u32,
  ro: vec3f,
  volBase: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  absorb: vec4f,
  emit: vec4f,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;

struct VSOut { @builtin(position) pos: vec4f, }

struct FSOut {
  @location(0) color: vec4f,
  @location(1) occl: vec4f,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}

fn densAt(ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(draw.gridM);
  let base = u32(draw.volBase);
  let x = clamp(ix, 0, M - 1);
  let y = clamp(iy, 0, M - 1);
  let z = clamp(iz, 0, M - 1);
  return volume[base + u32(x) + u32(y) * draw.gridM + u32(z) * draw.gridM * draw.gridM];
}

fn densAtBase(base: u32, ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(draw.gridM);
  let x = clamp(ix, 0, M - 1);
  let y = clamp(iy, 0, M - 1);
  let z = clamp(iz, 0, M - 1);
  return volume[base + u32(x) + u32(y) * draw.gridM + u32(z) * draw.gridM * draw.gridM];
}

fn chebIndex(xi: f32) -> f32 {
  let x = clamp(xi, -1.0, 1.0);
  return f32(draw.gridM) / 3.141592653589793 * acos(x) - 0.5;
}

fn sampleFieldBase(base: u32, p: vec3f) -> f32 {
  let half = draw.half;
  let xi = p / half;
  if (abs(xi.x) > 1.0 || abs(xi.y) > 1.0 || abs(xi.z) > 1.0) { return 0.0; }
  let fx = chebIndex(xi.x); let fy = chebIndex(xi.y); let fz = chebIndex(xi.z);
  let x0 = i32(floor(fx)); let y0 = i32(floor(fy)); let z0 = i32(floor(fz));
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let tz = clamp(fz - f32(z0), 0.0, 1.0);
  let c00 = mix(densAtBase(base, x0, y0, z0), densAtBase(base, x0 + 1, y0, z0), tx);
  let c10 = mix(densAtBase(base, x0, y0 + 1, z0), densAtBase(base, x0 + 1, y0 + 1, z0), tx);
  let c01 = mix(densAtBase(base, x0, y0, z0 + 1), densAtBase(base, x0 + 1, y0, z0 + 1), tx);
  let c11 = mix(densAtBase(base, x0, y0 + 1, z0 + 1), densAtBase(base, x0 + 1, y0 + 1, z0 + 1), tx);
  return mix(mix(c00, c10, ty), mix(c01, c11, ty), tz);
}

fn sampleVolume(p: vec3f) -> f32 {
  return sampleFieldBase(u32(draw.volBase), p);
}

fn fieldAt(p: vec3f) -> f32 {
  var d = sampleVolume(p);
  if (d != d) { d = 0.0; }
  return d - draw.isoLevel;
}

/** Analytic ∇f from Chebyshev-diff IDCT slabs (∂/∂ξ,∂/∂η,∂/∂ζ), then / half → world. */
fn fieldGrad(p: vec3f) -> vec3f {
  let half = draw.half;
  let volN = draw.gridM * draw.gridM * draw.gridM;
  let b0 = u32(draw.volBase);
  let gxi = sampleFieldBase(b0 + volN, p);
  let geta = sampleFieldBase(b0 + 2u * volN, p);
  let gzeta = sampleFieldBase(b0 + 3u * volN, p);
  let invH = select(0.0, 1.0 / half, abs(half) > 1e-12);
  return vec3f(gxi, geta, gzeta) * invH;
}

fn shadeIso(p: vec3f, rd: vec3f) -> vec4f {
  var g = fieldGrad(p);
  let gl = length(g);
  var n = select(vec3f(0.0, 1.0, 0.0), g / gl, gl > 1e-8);
  if (dot(n, -rd) < 0.0) { n = -n; }
  let L = normalize(vec3f(0.35, 0.85, 0.45));
  let H = normalize(L - normalize(rd));
  let ndotl = max(dot(n, L), 0.0);
  let spec = pow(max(dot(n, H), 0.0), 32.0);
  let ambient = 0.18;
  let lambert = ambient + (1.0 - ambient) * ndotl;
  let base = mix(draw.absorb.xyz, draw.emit.xyz, 0.65);
  let rgb = base * lambert + vec3f(spec) * 0.35;
  return vec4f(rgb, 1.0);
}

fn marchIso(ro: vec3f, rd: vec3f, tEnter: f32, tExit: f32) -> FSOut {
  var out: FSOut;
  out.color = vec4f(0.0);
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);

  var steps = draw.steps;
  if (steps < 16u) { steps = 16u; }
  if (steps > 192u) { steps = 192u; }
  let dt = (tExit - tEnter) / f32(steps);
  var s0 = tEnter;
  var f0 = fieldAt(ro + rd * s0);
  let far = max(tExit, draw.half * 4.0);

  for (var i: u32 = 0u; i < 192u; i++) {
    if (i >= steps) { break; }
    let s1 = min(s0 + dt, tExit);
    let f1 = fieldAt(ro + rd * s1);
    if (f0 * f1 < 0.0) {
      var lo = s0; var hi = s1; var flo = f0;
      for (var b: u32 = 0u; b < 12u; b++) {
        let mid = 0.5 * (lo + hi);
        let fm = fieldAt(ro + rd * mid);
        if (flo * fm <= 0.0) { hi = mid; } else { lo = mid; flo = fm; }
      }
      let hit = 0.5 * (lo + hi);
      out.color = shadeIso(ro + rd * hit, rd);
      out.occl = vec4f(clamp(hit / far, 0.0, 0.999), 0.0, 0.0, 1.0);
      return out;
    }
    s0 = s1; f0 = f1;
    if (s0 >= tExit - 1e-6) { break; }
  }
  return out;
}

@fragment
fn fsMain(in: VSOut) -> FSOut {
  var out: FSOut;
  out.color = vec4f(0.0);
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);

  let fbW = f32(draw.fbW); let fbH = f32(draw.fbH);
  let ndcX = -1.0 + 2.0 * in.pos.x / fbW;
  let ndcY = 1.0 - 2.0 * in.pos.y / fbH;
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(draw.m0.xyz, xy1), dot(draw.m1.xyz, xy1), dot(draw.m2.xyz, xy1));
  let ro = draw.ro; let half = draw.half;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB); let tmax = max(tA, tB);
  var tEnter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  if (!(tExit > tEnter + 1e-6)) { return out; }
  return marchIso(ro, rd, tEnter, tExit);
}
`;
}

function makeBeerMultiWgsl() {
  return /* wgsl */ `
struct DrawParams {
  fbW: u32,
  fbH: u32,
  gridM: u32,
  steps: u32,
  half: f32,
  scale: f32,
  densBase: f32,
  layerCount: u32,
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
@group(0) @binding(2) var occlTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> layerColors: array<vec4f>;

struct VSOut { @builtin(position) pos: vec4f, }

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}

fn densAtBase(base: u32, ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(draw.gridM);
  let x = clamp(ix, 0, M - 1); let y = clamp(iy, 0, M - 1); let z = clamp(iz, 0, M - 1);
  return volume[base + u32(x) + u32(y) * draw.gridM + u32(z) * draw.gridM * draw.gridM];
}
fn chebIndex(xi: f32) -> f32 {
  let x = clamp(xi, -1.0, 1.0);
  return f32(draw.gridM) / 3.141592653589793 * acos(x) - 0.5;
}
fn sampleLayer(base: u32, p: vec3f) -> f32 {
  let half = draw.half; let xi = p / half;
  if (abs(xi.x) > 1.0 || abs(xi.y) > 1.0 || abs(xi.z) > 1.0) { return 0.0; }
  let fx = chebIndex(xi.x); let fy = chebIndex(xi.y); let fz = chebIndex(xi.z);
  let x0 = i32(floor(fx)); let y0 = i32(floor(fy)); let z0 = i32(floor(fz));
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let tz = clamp(fz - f32(z0), 0.0, 1.0);
  let c000 = densAtBase(base, x0, y0, z0); let c100 = densAtBase(base, x0 + 1, y0, z0);
  let c010 = densAtBase(base, x0, y0 + 1, z0); let c110 = densAtBase(base, x0 + 1, y0 + 1, z0);
  let c001 = densAtBase(base, x0, y0, z0 + 1); let c101 = densAtBase(base, x0 + 1, y0, z0 + 1);
  let c011 = densAtBase(base, x0, y0 + 1, z0 + 1); let c111 = densAtBase(base, x0 + 1, y0 + 1, z0 + 1);
  return mix(mix(mix(c000, c100, tx), mix(c010, c110, tx), ty),
              mix(mix(c001, c101, tx), mix(c011, c111, tx), ty), tz);
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let fbW = f32(draw.fbW); let fbH = f32(draw.fbH);
  let ndcX = -1.0 + 2.0 * in.pos.x / fbW;
  let ndcY = 1.0 - 2.0 * in.pos.y / fbH;
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(draw.m0.xyz, xy1), dot(draw.m1.xyz, xy1), dot(draw.m2.xyz, xy1));
  let ro = draw.ro; let half = draw.half;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB); let tmax = max(tA, tB);
  var tEnter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  var tExit = min(min(tmax.x, tmax.y), tmax.z);
  if (!(tExit > tEnter + 1e-6)) { return vec4f(0.0); }

  let far = max(tExit, half * 4.0);
  let px = u32(clamp(floor(in.pos.x), 0.0, fbW - 1.0));
  let py = u32(clamp(floor(in.pos.y), 0.0, fbH - 1.0));
  let dSurf = textureLoad(occlTex, vec2u(px, py), 0).r;
  if (dSurf < 0.999) { tExit = min(tExit, dSurf * far); }
  if (!(tExit > tEnter + 1e-6)) { return vec4f(0.0); }

  var steps = draw.steps;
  if (steps < 8u) { steps = 8u; }
  if (steps > 96u) { steps = 96u; }
  let dt = (tExit - tEnter) / f32(steps);
  let ds = length(rd) * dt;
  let volN = draw.gridM * draw.gridM * draw.gridM;
  let densBase = u32(draw.densBase);
  let nLay = min(draw.layerCount, ${MAX_DENS_LAYERS}u);

  var rgb = vec3f(0.0); var T = 1.0; var s = tEnter + 0.5 * dt;
  for (var i: u32 = 0u; i < 96u; i++) {
    if (i >= steps) { break; }
    if (T < 0.002) { break; }
    let p = ro + rd * s;
    var sigma = 0.0; var emitAcc = vec3f(0.0);
    for (var L: u32 = 0u; L < ${MAX_DENS_LAYERS}u; L++) {
      if (L >= nLay) { break; }
      var dval = sampleLayer(densBase + L * volN, p);
      if (dval != dval) { dval = 0.0; }
      dval = clamp(dval, -4.0, 8.0);
      let sig = min(max(0.0, draw.scale * dval), 40.0);
      if (sig > 1e-8) {
        let col = layerColors[L].xyz;
        sigma += sig;
        emitAcc += col * sig;
      }
    }
    if (sigma > 1e-8) {
      let absorb = exp(-sigma * ds);
      let opacity = 1.0 - absorb;
      let col = emitAcc / sigma;
      // Match single-layer look: emit * sigma + small absorb tint.
      rgb += T * opacity * (col * sigma + col * 0.15);
      T *= absorb;
    }
    s += dt;
  }
  let a = 1.0 - T;
  if (a < 0.001) { return vec4f(0.0); }
  return vec4f(rgb * a, a);
}
`;
}

/**
 * Compact FXAA 3.11-style post (luma edge detect + blend).
 * Input is premultiplied overlay; we un-premultiply for luma, then re-premultiply.
 */
function makeFxaaWgsl() {
  return /* wgsl */ `
struct FxaaParams {
  invRes: vec2f,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: FxaaParams;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var srcSamp: sampler;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  // WebGPU: texel (0,0) is top-left; clip +Y is up — flip V so the overlay matches Three.
  o.uv = vec2f(p[vi].x * 0.5 + 0.5, -p[vi].y * 0.5 + 0.5);
  return o;
}

fn luma(rgb: vec3f) -> f32 {
  return dot(rgb, vec3f(0.299, 0.587, 0.114));
}

fn samplePremul(uv: vec2f) -> vec4f {
  return textureSampleLevel(srcTex, srcSamp, uv, 0.0);
}

fn unpremul(c: vec4f) -> vec3f {
  return c.xyz / max(c.a, 1e-4);
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let rcp = u.invRes;
  let uv = in.uv;

  let rgbM = samplePremul(uv);
  // Skip empty pixels — keep overlay holes crisp for Three underneath.
  if (rgbM.a < 0.001) { return vec4f(0.0); }

  let rgbNW = samplePremul(uv + vec2f(-rcp.x, -rcp.y));
  let rgbNE = samplePremul(uv + vec2f( rcp.x, -rcp.y));
  let rgbSW = samplePremul(uv + vec2f(-rcp.x,  rcp.y));
  let rgbSE = samplePremul(uv + vec2f( rcp.x,  rcp.y));

  // Edge detect on straight alpha + luma (silhouettes are mostly alpha edges).
  let lM = luma(unpremul(rgbM)) * rgbM.a + rgbM.a;
  let lNW = luma(unpremul(rgbNW)) * rgbNW.a + rgbNW.a;
  let lNE = luma(unpremul(rgbNE)) * rgbNE.a + rgbNE.a;
  let lSW = luma(unpremul(rgbSW)) * rgbSW.a + rgbSW.a;
  let lSE = luma(unpremul(rgbSE)) * rgbSE.a + rgbSE.a;

  let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  let range = lMax - lMin;
  // No edge → passthrough.
  if (range < max(0.0312, lMax * 0.125)) {
    return rgbM;
  }

  var dir = vec2f(
    -((lNW + lNE) - (lSW + lSE)),
    ((lNW + lSW) - (lNE + lSE)),
  );
  let dirReduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
  let rcpDir = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDir, vec2f(-8.0), vec2f(8.0)) * rcp;

  let rgbA = 0.5 * (
    samplePremul(uv + dir * (1.0 / 3.0 - 0.5)) +
    samplePremul(uv + dir * (2.0 / 3.0 - 0.5))
  );
  let rgbB = rgbA * 0.5 + 0.25 * (
    samplePremul(uv + dir * -0.5) +
    samplePremul(uv + dir * 0.5)
  );

  let lB = luma(unpremul(rgbB)) * rgbB.a + rgbB.a;
  if (lB < lMin || lB > lMax) {
    return rgbA;
  }
  return rgbB;
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

let isoPipeline = null;
let beerPipeline = null;
let fxaaPipeline = null;

let drawParamBuf = null;
let drawParamBufBeer = null;
let fxaaParamBuf = null;
let volumeBuf = null;
let volumeCapacity = 0;
let colorBuf = null;

let occlTex = null;
let occlW = 0;
let occlH = 0;
/** Intermediate color (iso+beer) before FXAA → swapchain. */
let sceneColorTex = null;
let sceneColorW = 0;
let sceneColorH = 0;
let fxaaSampler = null;

/** @type {{ color: number[], isoLevel: number, base: number }[]} */
let sceneConstraints = [];
let densPacked = false;
/** @type {number[][]} */
let densColors = [];
let densLayerCount = 0;
let densBase = 0;
let sceneM = 0;
let sceneEpoch = 0;
/** @type {Float32Array | null} */
let scenePacked = null;

let initFailed = false;
let initPromise = null;
let timestampsSupported = false;
let stampQuerySet = null;
let stampResolveBuf = null;
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

const PIPELINE_EPOCH = 11;
let builtEpoch = -1;

/** Classic 256-byte pack; volBase lives in the f32 pad after ro (index 11). */
function packDrawParamsIso(fbW, fbH, gridM, steps, half, scale, isoLevel, volBase, ro, M, absorb, emit) {
  const buf = new ArrayBuffer(256);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = fbW | 0; u32[1] = fbH | 0; u32[2] = gridM | 0; u32[3] = steps | 0;
  f32[4] = half; f32[5] = scale; f32[6] = isoLevel; u32[7] = 1;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[11] = volBase; // exact integer bases fit in f32 for our volume sizes
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
  f32[24] = absorb[0]; f32[25] = absorb[1]; f32[26] = absorb[2];
  f32[28] = emit[0]; f32[29] = emit[1]; f32[30] = emit[2];
  return buf;
}

/**
 * Beer uniforms — densBase in the isoLevel slot (f32), layerCount in shadeMode slot.
 * Colors live in a separate storage buffer (avoids uniform array packing issues).
 */
function packDrawParamsBeer(fbW, fbH, gridM, steps, half, scale, densBaseOff, layerCount, ro, M) {
  const buf = new ArrayBuffer(256);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = fbW | 0; u32[1] = fbH | 0; u32[2] = gridM | 0; u32[3] = steps | 0;
  f32[4] = half; f32[5] = scale; f32[6] = densBaseOff; u32[7] = layerCount | 0;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
  f32[24] = 0.15; f32[25] = 0.25; f32[26] = 0.45;
  f32[28] = 0.55; f32[29] = 0.75; f32[30] = 1.0;
  return buf;
}

function writeLayerColors(colors) {
  if (!device || !colorBuf) return;
  const data = new Float32Array(MAX_DENS_LAYERS * 4);
  for (let i = 0; i < MAX_DENS_LAYERS; i++) {
    const c = colors[i] || [0, 0, 0];
    data[i * 4] = c[0];
    data[i * 4 + 1] = c[1];
    data[i * 4 + 2] = c[2];
    data[i * 4 + 3] = 1;
  }
  device.queue.writeBuffer(colorBuf, 0, data);
}

export function isClipBakeGpuReady() {
  return Boolean(device && isoPipeline && beerPipeline && fxaaPipeline);
}
export function isClipMarchReady() {
  return Boolean(
    isClipBakeGpuReady() && ctx && sceneM > 1 &&
    (densLayerCount > 0 || sceneConstraints.length > 0),
  );
}

function noteGpuPresent(submitWallAt) {
  const now = performance.now();
  profilePresentWallMs = profilePresentWallMs * 0.85 + (now - submitWallAt) * 0.15;
  if (lastPresentAt > 0) profilePresentIntervalMs = profilePresentIntervalMs * 0.85 + (now - lastPresentAt) * 0.15;
  else profilePresentIntervalMs = now - submitWallAt;
  lastPresentAt = now;
}

export function getClipGpuProfile() {
  return {
    idctMs: profileBakeMs,
    marchMs: profileMarchMs,
    marchFbW: profileMarchFbW,
    marchFbH: profileMarchFbH,
    presentWallMs: profilePresentWallMs,
    presentIntervalMs: profilePresentIntervalMs,
    lastPresentAt,
    method: profileMethod,
    gridM: profileGridM,
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

function ensureOcclTex(w, h) {
  if (occlTex && occlW === w && occlH === h) return;
  if (occlTex) { try { occlTex.destroy(); } catch (_) {} }
  occlTex = device.createTexture({
    size: [w, h],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  occlW = w;
  occlH = h;
}

function ensureSceneColorTex(w, h) {
  if (sceneColorTex && sceneColorW === w && sceneColorH === h) return;
  if (sceneColorTex) { try { sceneColorTex.destroy(); } catch (_) {} }
  sceneColorTex = device.createTexture({
    size: [w, h],
    format: canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  sceneColorW = w;
  sceneColorH = h;
}

export function resizeClipGpuCanvas(pixelW, pixelH) {
  if (!canvas || !device) return { w: pixelW | 0, h: pixelH | 0 };
  if (!ctx) {
    ctx = canvas.getContext("webgpu");
    if (!ctx) return { w: pixelW | 0, h: pixelH | 0 };
  }
  const w = Math.max(1, pixelW | 0);
  const h = Math.max(1, pixelH | 0);
  const changed = canvas.width !== w || canvas.height !== h;
  if (changed) {
    canvas.width = w;
    canvas.height = h;
  }
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  if (changed || !canvas._clipConfigured) {
    canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({
      device,
      format: canvasFormat,
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    canvas._clipConfigured = true;
  }
  ensureOcclTex(w, h);
  ensureSceneColorTex(w, h);
  return { w: canvas.width, h: canvas.height };
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
  if (old) {
    void device.queue.onSubmittedWorkDone().then(() => {
      try { old.destroy(); } catch (_) {}
    });
  }
}

/** Update density colors without re-baking volumes. */
export function uploadSceneColors(colors) {
  densColors = (colors || []).slice(0, MAX_DENS_LAYERS).map((c) => c || [0.55, 0.75, 1]);
  writeLayerColors(densColors);
}

export function uploadSceneVolumes(scene) {
  if (!device || !scene) return null;
  const t0 = performance.now();
  const M = Math.max(2, scene.M | 0);
  const volN = M * M * M;
  sceneM = M;

  const cons = scene.constraints || [];
  const dens = (scene.densLayers || []).slice(0, MAX_DENS_LAYERS);
  densLayerCount = dens.length;
  densColors = dens.map((d) => {
    const c = d.color || [0.55, 0.75, 1];
    return [c[0], c[1], c[2]];
  });

  // Each constraint packs dens + gx + gy + gz (Chebyshev-diff IDCT slabs).
  const consStride = 4;
  const totalFloats = (cons.length * consStride + dens.length) * volN;
  scenePacked = totalFloats > 0 ? new Float32Array(Math.max(volN, totalFloats)) : null;
  let off = 0;
  const putVol = (src) => {
    if (!scenePacked) return;
    if (src && src.length) {
      scenePacked.set(src.length >= volN ? src.subarray(0, volN) : src, off);
    }
    off += volN;
  };
  sceneConstraints = cons.map((c) => {
    const base = off;
    putVol(c.dens);
    putVol(c.gx);
    putVol(c.gy);
    putVol(c.gz);
    return {
      color: c.color || [0.9, 0.45, 0.35],
      isoLevel: Number.isFinite(c.isoLevel) ? c.isoLevel : 0,
      base,
    };
  });
  densBase = off;
  densPacked = dens.length > 0;
  if (densPacked && scenePacked) {
    for (let i = 0; i < dens.length; i++) {
      putVol(dens[i].dens);
    }
  }

  sceneEpoch++;
  ensureVolumeBuf(Math.max(volN, scenePacked ? scenePacked.length : volN));
  if (scenePacked) device.queue.writeBuffer(volumeBuf, 0, scenePacked);
  writeLayerColors(densColors);

  profileBakeMs = profileBakeMs * 0.5 + (performance.now() - t0) * 0.5;
  profileGridM = M;
  profileMethod = "cpu-idct-scene";
  return { M, bakeMs: performance.now() - t0, epoch: sceneEpoch };
}

export function hasUploadedVolume() {
  return sceneM > 0 && (densLayerCount > 0 || sceneConstraints.length > 0);
}

export async function initClipBakeGpu(viewportEl) {
  if (isClipBakeGpuReady()) return true;
  if (initFailed) return false;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    try {
      if (!navigator.gpu) { initFailed = true; return false; }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) { initFailed = true; return false; }
      timestampsSupported = adapter.features.has("timestamp-query");
      const requiredFeatures = timestampsSupported ? ["timestamp-query"] : [];
      device = await adapter.requestDevice({ requiredFeatures });
      device.lost.then(() => {
        device = null;
        isoPipeline = beerPipeline = fxaaPipeline = null;
        initFailed = true;
      });
      if (timestampsSupported) {
        stampQuerySet = device.createQuerySet({ type: "timestamp", count: 2 });
        stampResolveBuf = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        stampReadBuf = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }
      drawParamBuf = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      drawParamBufBeer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      fxaaParamBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      fxaaSampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      });
      colorBuf = device.createBuffer({
        size: MAX_DENS_LAYERS * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      writeLayerColors([[0.55, 0.75, 1]]);
      ensureVolumeBuf(8 * 8 * 8);
      await ensurePipelinesForDegree(4);
      if (viewportEl) attachClipGpuCanvas(viewportEl);
      if (canvas) ctx = canvas.getContext("webgpu");
      return isClipBakeGpuReady();
    } catch (e) {
      console.warn("[clipBakeGpu] init failed", e);
      initFailed = true;
      device = null;
      isoPipeline = beerPipeline = fxaaPipeline = null;
      return false;
    }
  })();
  return initPromise;
}

async function compileChecked(label, code) {
  const mod = device.createShaderModule({ code });
  const info = await mod.getCompilationInfo();
  for (const m of info.messages) {
    if (m.type === "error") throw new Error(`${label}: ${m.message}`);
  }
  return mod;
}

export async function ensurePipelinesForDegree(_deg) {
  if (!device) return false;
  if (isoPipeline && beerPipeline && fxaaPipeline && builtEpoch === PIPELINE_EPOCH) {
    return true;
  }

  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const isoMod = await compileChecked("iso", makeIsoWgsl());
  const beerMod = await compileChecked("beer", makeBeerMultiWgsl());
  const fxaaMod = await compileChecked("fxaa", makeFxaaWgsl());

  const blendPremul = {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  };
  const blendMin = {
    color: { srcFactor: "one", dstFactor: "one", operation: "min" },
    alpha: { srcFactor: "one", dstFactor: "one", operation: "min" },
  };

  device.pushErrorScope("validation");
  const nextIso = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: isoMod, entryPoint: "vsMain" },
    fragment: {
      module: isoMod,
      entryPoint: "fsMain",
      targets: [
        { format: canvasFormat, blend: blendPremul },
        { format: "rgba16float", blend: blendMin },
      ],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`iso: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextBeer = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: beerMod, entryPoint: "vsMain" },
    fragment: {
      module: beerMod,
      entryPoint: "fsMain",
      targets: [{ format: canvasFormat, blend: blendPremul }],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`beer: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextFxaa = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: fxaaMod, entryPoint: "vsMain" },
    fragment: {
      module: fxaaMod,
      entryPoint: "fsMain",
      targets: [{ format: canvasFormat }],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`fxaa: ${err.message}`);
  }

  isoPipeline = nextIso;
  beerPipeline = nextBeer;
  fxaaPipeline = nextFxaa;
  builtEpoch = PIPELINE_EPOCH;
  return true;
}

function scheduleStampReadback() {
  if (!timestampsSupported || !stampReadBuf || stampReadPending) return;
  stampReadPending = true;
  stampReadBuf.mapAsync(GPUMapMode.READ).then(() => {
    const stamps = new BigInt64Array(stampReadBuf.getMappedRange().slice(0));
    stampReadBuf.unmap();
    stampReadPending = false;
    if (stamps[1] > stamps[0]) {
      profileMarchMs = profileMarchMs * 0.7 + Number(stamps[1] - stamps[0]) / 1e6 * 0.3;
    }
  }).catch(() => { stampReadPending = false; });
}

function darken(c, t) {
  return [c[0] * t, c[1] * t, c[2] * t];
}

export function renderClipFrameGpu({ camera, half, fbW, fbH, scale, steps }) {
  if (!isClipBakeGpuReady() || !ctx || !volumeBuf || !colorBuf || !fxaaParamBuf || !fxaaSampler) {
    return false;
  }
  if (densLayerCount < 1 && sceneConstraints.length < 1) return false;

  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const Mat = ndcToDirMatrix(camera, sx, sy);
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;
  const Mgrid = sceneM;

  const { w: marchW, h: marchH } = resizeClipGpuCanvas(fbW, fbH);
  if (!occlTex) ensureOcclTex(marchW, marchH);
  if (!sceneColorTex) ensureSceneColorTex(marchW, marchH);

  profileMarchFbW = marchW;
  profileMarchFbH = marchH;
  profileMethod = "gpu-iso+beer+fxaa";
  profileGridM = Mgrid;

  if (scenePacked) device.queue.writeBuffer(volumeBuf, 0, scenePacked);
  writeLayerColors(densColors);

  const sceneView = sceneColorTex.createView();
  const swapView = ctx.getCurrentTexture().createView();
  const occlView = occlTex.createView();

  // Clear scene color + occl (far = 1)
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
      ],
    });
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  for (const c of sceneConstraints) {
    device.queue.writeBuffer(
      drawParamBuf,
      0,
      packDrawParamsIso(
        marchW, marchH, Mgrid, steps, h, scale, c.isoLevel, c.base, ro, Mat,
        darken(c.color, 0.35), c.color,
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
      ],
    });
    pass.setPipeline(isoPipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  if (densLayerCount > 0 && densPacked) {
    device.queue.writeBuffer(
      drawParamBufBeer,
      0,
      packDrawParamsBeer(marchW, marchH, Mgrid, steps, h, scale, densBase, densLayerCount, ro, Mat),
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

  // FXAA → swapchain
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

  const submitWallAt = performance.now();
  void device.queue.onSubmittedWorkDone().then(() => {
    noteGpuPresent(submitWallAt);
    scheduleStampReadback();
  });
  return true;
}
