/**
 * WebGPU volume march: IDCT dens grids + iso manifolds + multi-layer Beer.
 */
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix } from "./clipGrid.js";

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
  volBaseB: f32,
  blendT: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;

struct VSOut { @builtin(position) pos: vec4f, }

struct FSOut {
  @location(0) color: vec4f,
  @location(1) occl: vec4f,
  @location(2) normal: vec4f,
  @builtin(frag_depth) depth: f32,
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
  let a = sampleFieldBase(u32(draw.volBase), p);
  let b = sampleFieldBase(u32(draw.volBaseB), p);
  return mix(a, b, draw.blendT);
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
  let b1 = u32(draw.volBaseB);
  let t = draw.blendT;
  let gxi = mix(sampleFieldBase(b0 + volN, p), sampleFieldBase(b1 + volN, p), t);
  let geta = mix(sampleFieldBase(b0 + 2u * volN, p), sampleFieldBase(b1 + 2u * volN, p), t);
  let gzeta = mix(sampleFieldBase(b0 + 3u * volN, p), sampleFieldBase(b1 + 3u * volN, p), t);
  let invH = select(0.0, 1.0 / half, abs(half) > 1e-12);
  return vec3f(gxi, geta, gzeta) * invH;
}

fn shadeIso(p: vec3f, rd: vec3f, n: vec3f) -> vec4f {
  let L = normalize(vec3f(0.35, 0.85, 0.45));
  let ndotl = max(dot(n, L), 0.0);
  let ambient = 0.42;
  let lambert = ambient + (1.0 - ambient) * ndotl;
  let base = mix(draw.absorb.xyz, draw.emit.xyz, 0.65);
  let rgb = base * lambert;
  return vec4f(rgb, 1.0);
}

fn isoNormal(p: vec3f, rd: vec3f) -> vec3f {
  var g = fieldGrad(p);
  let gl = length(g);
  var n = select(vec3f(0.0, 1.0, 0.0), g / gl, gl > 1e-8);
  if (dot(n, -rd) < 0.0) { n = -n; }
  return n;
}

fn marchIso(ro: vec3f, rd: vec3f, tEnter: f32, tExit: f32) -> FSOut {
  var out: FSOut;
  out.color = vec4f(0.0);
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);
  out.normal = vec4f(0.0);
  out.depth = 1.0;

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
      let d = clamp(hit / far, 0.0, 0.999);
      let p = ro + rd * hit;
      let n = isoNormal(p, rd);
      out.color = shadeIso(p, rd, n);
      out.occl = vec4f(d, 0.0, 0.0, 1.0);
      out.normal = vec4f(n * 0.5 + 0.5, 1.0);
      out.depth = d;
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
  out.normal = vec4f(0.0);
  out.depth = 1.0;

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
  if (!(tExit > tEnter + 1e-6)) {
    discard;
    return out;
  }
  let hit = marchIso(ro, rd, tEnter, tExit);
  // Miss: discard so we don't overwrite a closer prior iso's color/depth.
  if (hit.color.a < 0.5) {
    discard;
    return out;
  }
  return hit;
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
 * World reference grid / axes / fit-box edges.
 * Rasterized with a real viewProj, but frag_depth uses the same ray-t / far
 * encoding as the iso pass so we depth-test against isoDepth in-place (no copy).
 */
function makeGridWgsl() {
  return /* wgsl */ `
struct GridParams {
  viewProj: mat4x4f,
  ro: vec3f,
  half: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  fbW: f32,
  fbH: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: GridParams;

struct VSIn {
  @location(0) pos: vec3f,
  @location(1) color: vec4f,
}

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) color: vec4f,
}

struct FSOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@vertex
fn vsMain(v: VSIn) -> VSOut {
  var o: VSOut;
  o.clip = u.viewProj * vec4f(v.pos, 1.0);
  o.world = v.pos;
  o.color = v.color;
  return o;
}

@fragment
fn fsMain(in: VSOut) -> FSOut {
  var out: FSOut;
  let fbW = u.fbW; let fbH = u.fbH;
  let ndcX = -1.0 + 2.0 * in.clip.x / fbW;
  let ndcY = 1.0 - 2.0 * in.clip.y / fbH;
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(u.m0.xyz, xy1), dot(u.m1.xyz, xy1), dot(u.m2.xyz, xy1));
  let ro = u.ro; let half = u.half;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB); let tmax = max(tA, tB);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  let far = max(tExit, half * 4.0);
  let rd2 = max(dot(rd, rd), 1e-20);
  let t = dot(in.world - ro, rd) / rd2;
  if (!(t > 0.0)) { discard; }
  out.depth = clamp(t / far, 0.0, 0.999);
  // Premultiplied
  out.color = vec4f(in.color.rgb * in.color.a, in.color.a);
  return out;
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

/**
 * Screen-space AO for isosurfaces only.
 * Reads iso depth (occl.r = t/far) + world normal; darkens premul scene color.
 * Skips pixels with no iso (normal.a < 0.5 / occl >= 0.999).
 */
function makeSsaoWgsl() {
  return /* wgsl */ `
struct SsaoParams {
  fbW: u32,
  fbH: u32,
  _pad0: u32,
  _pad1: u32,
  half: f32,
  radius: f32,
  strength: f32,
  bias: f32,
  ro: vec3f,
  _p1: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
}

@group(0) @binding(0) var<uniform> p: SsaoParams;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var occlTex: texture_2d<f32>;
@group(0) @binding(3) var normalTex: texture_2d<f32>;

struct VSOut { @builtin(position) pos: vec4f, }

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var pts = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(pts[vi], 0.0, 1.0);
  return o;
}

fn rayDir(ndcX: f32, ndcY: f32) -> vec3f {
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  return vec3f(dot(p.m0.xyz, xy1), dot(p.m1.xyz, xy1), dot(p.m2.xyz, xy1));
}

fn boxFar(ro: vec3f, rd: vec3f) -> f32 {
  let half = p.half;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmax = max(tA, tB);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  return max(tExit, half * 4.0);
}

fn hash2(pxy: vec2f) -> f32 {
  return fract(sin(dot(pxy, vec2f(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let fbW = f32(p.fbW);
  let fbH = f32(p.fbH);
  let px = u32(clamp(floor(in.pos.x), 0.0, fbW - 1.0));
  let py = u32(clamp(floor(in.pos.y), 0.0, fbH - 1.0));
  let uv = vec2u(px, py);

  let color = textureLoad(sceneTex, uv, 0);
  let depthN = textureLoad(occlTex, uv, 0).r;
  let nEnc = textureLoad(normalTex, uv, 0);
  if (depthN >= 0.999 || nEnc.a < 0.5) {
    return color;
  }

  let ndcX = -1.0 + 2.0 * (f32(px) + 0.5) / fbW;
  let ndcY = 1.0 - 2.0 * (f32(py) + 0.5) / fbH;
  let rd = rayDir(ndcX, ndcY);
  let ro = p.ro;
  let far = boxFar(ro, rd);
  let t0 = depthN * far;
  let pos0 = ro + rd * t0;
  let n = normalize(nEnc.xyz * 2.0 - 1.0);

  // Screen-space radius shrinks with distance (approx).
  let radPx = clamp(p.radius * 0.5 * min(fbW, fbH) / max(t0, 0.2), 2.0, 28.0);
  let rot = hash2(vec2f(f32(px), f32(py))) * 6.2831853;
  let cR = cos(rot);
  let sR = sin(rot);

  var occ = 0.0;
  var wSum = 0.0;
  for (var i: u32 = 0u; i < 16u; i++) {
    let fi = f32(i);
    let ang = fi * 2.3999632 + rot; // golden angle
    let r = sqrt((fi + 0.5) / 16.0) * radPx;
    let off = vec2f(cos(ang) * r, sin(ang) * r);
    // rotate slightly with noise
    let ox = off.x * cR - off.y * sR;
    let oy = off.x * sR + off.y * cR;

    let sx = i32(px) + i32(round(ox));
    let sy = i32(py) + i32(round(oy));
    if (sx < 0 || sy < 0 || sx >= i32(p.fbW) || sy >= i32(p.fbH)) { continue; }
    let suv = vec2u(u32(sx), u32(sy));
    let dS = textureLoad(occlTex, suv, 0).r;
    if (dS >= 0.999) { continue; }

    let sndcX = -1.0 + 2.0 * (f32(sx) + 0.5) / fbW;
    let sndcY = 1.0 - 2.0 * (f32(sy) + 0.5) / fbH;
    let srd = rayDir(sndcX, sndcY);
    let sFar = boxFar(ro, srd);
    let tS = dS * sFar;
    let posS = ro + srd * tS;
    let v = posS - pos0;
    let dist = length(v);
    if (dist < 1e-5) { continue; }
    let vn = v / dist;
    // Hemisphere contribution + range falloff (Alchemy-ish).
    let nd = max(dot(n, vn) - p.bias, 0.0);
    let fall = 1.0 - smoothstep(0.0, p.radius, dist);
    let w = fall;
    occ += nd * fall;
    wSum += w;
  }

  var ao = 1.0;
  if (wSum > 1e-4) {
    ao = clamp(1.0 - (occ / wSum), 0.0, 1.0);
    // Mild contrast so creases read without crushing flats.
    ao = pow(ao, 1.25);
  }
  let factor = mix(1.0, ao, p.strength);
  return vec4f(color.rgb * factor, color.a);
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
let ssaoPipeline = null;
let gridPipeline = null;
let gridParamBuf = null;
let gridVertexBuf = null;
let gridVertexCapacity = 0;
let gridVertexCount = 0;
let gridHalf = NaN;

let drawParamBuf = null;
let drawParamBufBeer = null;
let fxaaParamBuf = null;
let ssaoParamBuf = null;
let volumeBuf = null;
let volumeCapacity = 0;
let colorBuf = null;

let occlTex = null;
let occlW = 0;
let occlH = 0;
/** Hardware depth for iso-vs-iso (frag_depth = hit/far, compare less). */
let depthTex = null;
let depthW = 0;
let depthH = 0;
/** World normals from iso hits (rgb = n*0.5+0.5, a = 1 if valid). */
let normalTex = null;
let normalW = 0;
let normalH = 0;
/** Intermediate color (iso+beer) before FXAA → swapchain. */
let sceneColorTex = null;
let sceneColorW = 0;
let sceneColorH = 0;
/** Ping-pong target for SSAO apply (iso darkening). */
let sceneColorAoTex = null;
let sceneColorAoW = 0;
let sceneColorAoH = 0;
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

const PIPELINE_EPOCH = 16;
let builtEpoch = -1;

/**
 * Classic 256-byte pack; volBase / volBaseB / blendT for GPU keyframe mix.
 * volBase at f32[11]; volBaseB + blendT after emit at f32[32], f32[33].
 */
function packDrawParamsIso(
  fbW, fbH, gridM, steps, half, scale, isoLevel, volBase, ro, M, absorb, emit,
  volBaseB = volBase, blendT = 0,
) {
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
  f32[32] = volBaseB;
  f32[33] = blendT;
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

/** SSAO uniforms — 128 bytes. */
function packSsaoParams(fbW, fbH, half, radius, strength, bias, ro, M) {
  const buf = new ArrayBuffer(128);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = fbW | 0; u32[1] = fbH | 0;
  f32[4] = half; f32[5] = radius; f32[6] = strength; f32[7] = bias;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
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

function pushGridVert(dst, i, x, y, z, r, g, b, a) {
  const o = i * 8;
  dst[o] = x; dst[o + 1] = y; dst[o + 2] = z; dst[o + 3] = 0;
  dst[o + 4] = r; dst[o + 5] = g; dst[o + 6] = b; dst[o + 7] = a;
}

function pushGridLine(dst, i, ax, ay, az, bx, by, bz, r, g, b, a) {
  pushGridVert(dst, i, ax, ay, az, r, g, b, a);
  pushGridVert(dst, i + 1, bx, by, bz, r, g, b, a);
  return i + 2;
}

function hexToRgb(hex) {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
}

/**
 * Build / upload world grid + axes + fit-box edges for the WebGPU path.
 * Matches main.js rebuildWorldGrid extents.
 * @param {number} half
 */
export function syncClipGpuWorldGrid(half) {
  const h = Math.max(0.5, half);
  if (!device) {
    gridHalf = h;
    return;
  }
  if (gridVertexBuf && Math.abs(gridHalf - h) < 1e-9) return;
  gridHalf = h;

  const extent = Math.ceil(h + 0.5);
  const size = extent * 2;
  const divisions = Math.max(2, size);
  const step = size / divisions;
  const lo = -size / 2;
  const hi = size / 2;
  const [majR, majG, majB] = hexToRgb(0x4a5568);
  const [minR, minG, minB] = hexToRgb(0x2a3140);
  const maxVerts = 3 * (divisions + 1) * 2 * 2 + 6 + 24 + 16;
  const data = new Float32Array(maxVerts * 8);
  let n = 0;

  const emitPlane = (axis, alpha) => {
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
  n = pushGridLine(data, n, 0, 0, 0, axisLen, 0, 0, 0.9, 0.35, 0.38, 0.95);
  n = pushGridLine(data, n, 0, 0, 0, 0, axisLen, 0, 0.35, 0.75, 0.48, 0.95);
  n = pushGridLine(data, n, 0, 0, 0, 0, 0, axisLen, 0.4, 0.65, 0.95, 0.95);

  // Fit-box wireframe
  const bh = h;
  const br = 0.23; const bg = 0.27; const bb = 0.35; const ba = 0.85;
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

  gridVertexCount = n;
  const bytes = n * 8 * 4;
  if (!gridVertexBuf || gridVertexCapacity < bytes) {
    if (gridVertexBuf) { try { gridVertexBuf.destroy(); } catch (_) {} }
    gridVertexCapacity = Math.max(bytes, 4096);
    gridVertexBuf = device.createBuffer({
      size: gridVertexCapacity,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }
  device.queue.writeBuffer(gridVertexBuf, 0, data.subarray(0, n * 8));
}

function packGridParams(viewProj, ro, half, M, fbW, fbH) {
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

export function isClipBakeGpuReady() {
  return Boolean(
    device && isoPipeline && beerPipeline && fxaaPipeline && ssaoPipeline && gridPipeline,
  );
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

function ensureDepthTex(w, h) {
  if (depthTex && depthW === w && depthH === h) return;
  if (depthTex) { try { depthTex.destroy(); } catch (_) {} }
  depthTex = device.createTexture({
    size: [w, h],
    format: "depth32float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  depthW = w;
  depthH = h;
}

function ensureNormalTex(w, h) {
  if (normalTex && normalW === w && normalH === h) return;
  if (normalTex) { try { normalTex.destroy(); } catch (_) {} }
  normalTex = device.createTexture({
    size: [w, h],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  normalW = w;
  normalH = h;
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

function ensureSceneColorAoTex(w, h) {
  if (sceneColorAoTex && sceneColorAoW === w && sceneColorAoH === h) return;
  if (sceneColorAoTex) { try { sceneColorAoTex.destroy(); } catch (_) {} }
  sceneColorAoTex = device.createTexture({
    size: [w, h],
    format: canvasFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  sceneColorAoW = w;
  sceneColorAoH = h;
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
  ensureDepthTex(w, h);
  ensureNormalTex(w, h);
  ensureSceneColorTex(w, h);
  ensureSceneColorAoTex(w, h);
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

  // Constraints: dens+gx+gy+gz per keyframe (K=1 if static). Dens: one slab each.
  const consStride = 4;
  let consFloats = 0;
  for (const c of cons) {
    const K = Array.isArray(c.keyframes) && c.keyframes.length > 0
      ? c.keyframes.length
      : 1;
    consFloats += K * consStride * volN;
  }
  const totalFloats = consFloats + dens.length * volN;
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
    const frames = Array.isArray(c.keyframes) && c.keyframes.length > 0
      ? c.keyframes
      : null;
    if (frames) {
      for (const fr of frames) {
        putVol(fr.dens);
        putVol(fr.gx);
        putVol(fr.gy);
        putVol(fr.gz);
      }
    } else {
      putVol(c.dens);
      putVol(c.gx);
      putVol(c.gy);
      putVol(c.gz);
    }
    const blend = c.blend || { i0: 0, i1: 0, t: 0 };
    const K = frames ? frames.length : 1;
    return {
      id: c.id || null,
      color: c.color || [0.9, 0.45, 0.35],
      isoLevel: Number.isFinite(c.isoLevel) ? c.isoLevel : 0,
      base,
      frameStride: consStride * volN,
      K,
      i0: blend.i0 | 0,
      i1: blend.i1 | 0,
      t: Number.isFinite(blend.t) ? blend.t : 0,
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
  const anyKf = sceneConstraints.some((c) => c.K > 1);
  profileMethod = anyKf ? "gpu-kf-scene" : "cpu-idct-scene";
  return { M, bakeMs: performance.now() - t0, epoch: sceneEpoch };
}

/**
 * Hot-path anim: update keyframe segment indices + blend t without rewriting volumes.
 * @param {{ id?: string, i0: number, i1: number, t: number }[]} blends
 */
export function setConstraintKeyframeBlends(blends) {
  if (!blends?.length || !sceneConstraints.length) return;
  const byId = new Map();
  for (const b of blends) {
    if (b?.id != null) byId.set(b.id, b);
  }
  for (let i = 0; i < sceneConstraints.length; i++) {
    const c = sceneConstraints[i];
    const b = (c.id != null && byId.get(c.id)) || blends[i];
    if (!b) continue;
    c.i0 = Math.max(0, Math.min((c.K || 1) - 1, b.i0 | 0));
    c.i1 = Math.max(0, Math.min((c.K || 1) - 1, b.i1 | 0));
    c.t = Number.isFinite(b.t) ? b.t : 0;
  }
}

export function hasUploadedVolume() {
  return sceneM > 0 && (densLayerCount > 0 || sceneConstraints.length > 0);
}

/** Clear the WebGPU overlay to transparent (no density / iso). */
export function clearClipGpuFrame(fbW, fbH) {
  if (!device || !ctx || !canvas) return false;
  const { w, h } = resizeClipGpuCanvas(fbW, fbH);
  const view = ctx.getCurrentTexture().createView();
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.end();
  device.queue.submit([enc.finish()]);
  profileMarchFbW = w;
  profileMarchFbH = h;
  profileMethod = "gpu-clear";
  return true;
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
        isoPipeline = beerPipeline = fxaaPipeline = ssaoPipeline = gridPipeline = null;
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
      ssaoParamBuf = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      gridParamBuf = device.createBuffer({
        size: 256,
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
      isoPipeline = beerPipeline = fxaaPipeline = ssaoPipeline = gridPipeline = null;
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
  if (
    isoPipeline && beerPipeline && fxaaPipeline && ssaoPipeline && gridPipeline &&
    builtEpoch === PIPELINE_EPOCH
  ) {
    return true;
  }

  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const isoMod = await compileChecked("iso", makeIsoWgsl());
  const beerMod = await compileChecked("beer", makeBeerMultiWgsl());
  const gridMod = await compileChecked("grid", makeGridWgsl());
  const fxaaMod = await compileChecked("fxaa", makeFxaaWgsl());
  const ssaoMod = await compileChecked("ssao", makeSsaoWgsl());

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
        { format: "rgba8unorm" },
      ],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
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
  const nextGrid = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: gridMod,
      entryPoint: "vsMain",
      buffers: [{
        arrayStride: 32,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 16, format: "float32x4" },
        ],
      }],
    },
    fragment: {
      module: gridMod,
      entryPoint: "fsMain",
      targets: [{ format: canvasFormat, blend: blendPremul }],
    },
    primitive: { topology: "line-list" },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled: false,
      depthCompare: "less",
    },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`grid: ${err.message}`);
  }

  device.pushErrorScope("validation");
  const nextSsao = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: ssaoMod, entryPoint: "vsMain" },
    fragment: {
      module: ssaoMod,
      entryPoint: "fsMain",
      targets: [{ format: canvasFormat }],
    },
    primitive: { topology: "triangle-list" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`ssao: ${err.message}`);
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
  gridPipeline = nextGrid;
  ssaoPipeline = nextSsao;
  fxaaPipeline = nextFxaa;
  builtEpoch = PIPELINE_EPOCH;
  if (Number.isFinite(gridHalf)) {
    const h = gridHalf;
    gridHalf = NaN; // force rebuild into new device buffers
    syncClipGpuWorldGrid(h);
  }
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

export function renderClipFrameGpu({ camera, half, fbW, fbH, scale, steps, ndcOffsetX = 0 }) {
  if (
    !isClipBakeGpuReady() || !ctx || !volumeBuf || !colorBuf ||
    !fxaaParamBuf || !ssaoParamBuf || !fxaaSampler || !gridParamBuf
  ) {
    return false;
  }
  if (densLayerCount < 1 && sceneConstraints.length < 1) return false;

  const o = camera.position;
  const { sx, sy } = perspectiveDirScale(camera);
  const Mat = offsetDirMatrix(ndcToDirMatrix(camera, sx, sy), ndcOffsetX);
  const ro = [o.x, o.y, o.z];
  const h = half ?? 2;
  const Mgrid = sceneM;

  const { w: marchW, h: marchH } = resizeClipGpuCanvas(fbW, fbH);
  if (!occlTex) ensureOcclTex(marchW, marchH);
  if (!depthTex) ensureDepthTex(marchW, marchH);
  if (!normalTex) ensureNormalTex(marchW, marchH);
  if (!sceneColorTex) ensureSceneColorTex(marchW, marchH);
  if (!sceneColorAoTex) ensureSceneColorAoTex(marchW, marchH);
  syncClipGpuWorldGrid(h);

  profileMarchFbW = marchW;
  profileMarchFbH = marchH;
  profileMethod = "gpu-iso+ssao+beer+grid+fxaa";
  profileGridM = Mgrid;

  if (scenePacked) device.queue.writeBuffer(volumeBuf, 0, scenePacked);
  writeLayerColors(densColors);

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

  for (const c of sceneConstraints) {
    const stride = c.frameStride || 0;
    const base0 = c.base + (c.i0 | 0) * stride;
    const base1 = c.base + (c.i1 | 0) * stride;
    const blendT = Number.isFinite(c.t) ? c.t : 0;
    device.queue.writeBuffer(
      drawParamBuf,
      0,
      packDrawParamsIso(
        marchW, marchH, Mgrid, steps, h, scale, c.isoLevel, base0, ro, Mat,
        darken(c.color, 0.35), c.color,
        base1, blendT,
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
  if (sceneConstraints.length > 0) {
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

  // World grid / axes / box — depth-test against iso depth in-place (no copy).
  if (gridVertexBuf && gridVertexCount > 0 && gridPipeline) {
    camera.updateMatrixWorld(true);
    const viewProj = new Float32Array(16);
    // Column-major: projection * view (Three stores column-major too).
    const e = camera.projectionMatrix.elements;
    const v = camera.matrixWorldInverse.elements;
    // viewProj = proj * view
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
      packGridParams(viewProj, ro, h, Mat, marchW, marchH),
    );
    const bg = device.createBindGroup({
      layout: gridPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gridParamBuf } }],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: sceneView, loadOp: "load", storeOp: "store" },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(gridPipeline);
    pass.setBindGroup(0, bg);
    pass.setVertexBuffer(0, gridVertexBuf);
    pass.draw(gridVertexCount);
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
