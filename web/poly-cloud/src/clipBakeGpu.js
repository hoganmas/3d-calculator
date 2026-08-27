/**
 * WebGPU volume march: IDCT dens grids + iso manifolds + multi-layer Beer.
 */
import { ndcToDirMatrix, perspectiveDirScale, offsetDirMatrix } from "./clipGrid.js";
import { hexToRgb01, EXPR_GRADIENTS, MAX_GRAD_STOPS } from "./expressions.js";

export const MAX_DENS_LAYERS = 8;

const DEFAULT_DENS_RGB = hexToRgb01(EXPR_GRADIENTS[0].color);
const DEFAULT_DENS_RGB2 = hexToRgb01(EXPR_GRADIENTS[0].color2);
const DEFAULT_ISO_RGB = hexToRgb01(EXPR_GRADIENTS[1].color);
const DEFAULT_ISO_RGB2 = hexToRgb01(EXPR_GRADIENTS[1].color2);

/** Spatial + normal gradient factor (position for volumes, normal for 2D isos). */
const GRADIENT_WGSL = `
const MAX_GRAD_STOPS: u32 = ${MAX_GRAD_STOPS}u;

fn gradientT(p: vec3f, half: f32) -> f32 {
  let u = p / max(half, 1e-6);
  let ty = clamp(u.y * 0.5 + 0.5, 0.0, 1.0);
  let tx = clamp(u.x * 0.5 + 0.5, 0.0, 1.0);
  let tz = clamp(u.z * 0.5 + 0.5, 0.0, 1.0);
  let tr = clamp(length(u) * 0.70710678, 0.0, 1.0);
  return clamp(0.36 * ty + 0.24 * tx + 0.24 * tz + 0.16 * tr, 0.0, 1.0);
}

fn isoGradientT(p: vec3f, n: vec3f, half: f32) -> f32 {
  let tp = gradientT(p, half);
  let gdir = normalize(vec3f(0.12, 0.94, 0.32));
  let tn = clamp(0.5 + 0.5 * dot(n, gdir), 0.0, 1.0);
  let tb = clamp(0.5 + 0.35 * n.y, 0.0, 1.0);
  return clamp(mix(tp, tn, 0.58) * 0.82 + tb * 0.18, 0.0, 1.0);
}

/** Piecewise-linear sample across up to MAX_GRAD_STOPS colors. Count in stops[0].w */
fn sampleGradStops(stops: ptr<function, array<vec4f, ${MAX_GRAD_STOPS}>>, t: f32) -> vec3f {
  let n = max(min(u32((*stops)[0].w), MAX_GRAD_STOPS), 1u);
  if (n <= 1u) { return (*stops)[0].xyz; }
  let x = clamp(t, 0.0, 1.0) * f32(n - 1u);
  let i = min(u32(floor(x)), n - 2u);
  let f = fract(x);
  return mix((*stops)[i].xyz, (*stops)[i + 1u].xyz, f);
}
`;

/** Iso field interpolant: tricubic Hermite (C¹) vs trilinear. Compile-time switch. */
let isoInterpHermite = true;

export function getIsoInterpHermite() {
  return isoInterpHermite;
}

/** @returns {boolean} true if the mode changed (iso pipeline must rebuild) */
export function setIsoInterpHermite(on) {
  const next = !!on;
  if (next === isoInterpHermite) return false;
  isoInterpHermite = next;
  isoPipeline = null;
  return true;
}

/** Classic DrawParams layout (matches 872b141). volBase in the _p1 slot. */
function makeIsoWgsl() {
  const volumeSample = isoInterpHermite
    ? /* wgsl */ `
  let a = sampleFieldHermite4(u32(draw.volBase), p);
  let b = sampleFieldHermite4(u32(draw.volBaseB), p);
  return mix(a.x, b.x, draw.blendT);`
    : /* wgsl */ `
  let a = sampleFieldBase(u32(draw.volBase), p);
  let b = sampleFieldBase(u32(draw.volBaseB), p);
  return mix(a, b, draw.blendT);`;

  const gradSample = isoInterpHermite
    ? /* wgsl */ `
  let a = sampleFieldHermite4(u32(draw.volBase), p);
  let b = sampleFieldHermite4(u32(draw.volBaseB), p);
  return mix(a.yzw, b.yzw, draw.blendT);`
    : /* wgsl */ `
  let half = draw.half;
  let volN = draw.gridM * draw.gridM * draw.gridM;
  let b0 = u32(draw.volBase);
  let b1 = u32(draw.volBaseB);
  let t = draw.blendT;
  let gxi = mix(sampleFieldBase(b0 + volN, p), sampleFieldBase(b1 + volN, p), t);
  let geta = mix(sampleFieldBase(b0 + 2u * volN, p), sampleFieldBase(b1 + 2u * volN, p), t);
  let gzeta = mix(sampleFieldBase(b0 + 3u * volN, p), sampleFieldBase(b1 + 3u * volN, p), t);
  let invH = select(0.0, 1.0 / half, abs(half) > 1e-12);
  return vec3f(gxi, geta, gzeta) * invH;`;

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
  volBaseB: f32,
  blendT: f32,
  gradCount: f32,
  _padG: f32,
  g0: vec4f,
  g1: vec4f,
  g2: vec4f,
  g3: vec4f,
  g4: vec4f,
  g5: vec4f,
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
  // Clamp to the fit domain instead of returning 0 outside — a hard exterior
  // zero creates discontinuous normals / grainy isos on the box faces.
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
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

/** Chebyshev-root ξ at dens-grid index i. */
fn nodeXi(i: i32) -> f32 {
  return cos((f32(i) + 0.5) * 3.141592653589793 / f32(draw.gridM));
}

fn h00(t: f32) -> f32 { let t2 = t * t; return (2.0 * t - 3.0) * t2 + 1.0; }
fn h10(t: f32) -> f32 { let t2 = t * t; return (t - 2.0) * t2 + t; }
fn h01(t: f32) -> f32 { let t2 = t * t; return (-2.0 * t + 3.0) * t2; }
fn h11(t: f32) -> f32 { let t2 = t * t; return (t - 1.0) * t2; }
fn h00d(t: f32) -> f32 { return 6.0 * t * (t - 1.0); }
fn h10d(t: f32) -> f32 { return 3.0 * t * t - 4.0 * t + 1.0; }
fn h01d(t: f32) -> f32 { return 6.0 * t * (1.0 - t); }
fn h11d(t: f32) -> f32 { return 3.0 * t * t - 2.0 * t; }

fn hermiteVal(t: f32, f0: f32, f1: f32, d0: f32, d1: f32) -> f32 {
  return h00(t) * f0 + h10(t) * d0 + h01(t) * f1 + h11(t) * d1;
}
fn hermiteDt(t: f32, f0: f32, f1: f32, d0: f32, d1: f32) -> f32 {
  return h00d(t) * f0 + h10d(t) * d0 + h01d(t) * f1 + h11d(t) * d1;
}

/**
 * Tricubic Hermite (tensor-product) using dens + ∂f/∂ξ slabs at the 8 cell corners.
 * Derivatives are scaled to the unit-cell parameter t∈[0,1]. Mixed partials are
 * estimated by finite differences of ∇f across the cell (C¹ Hermite).
 * Returns (f, ∂f/∂x, ∂f/∂y, ∂f/∂z) in world space.
 */
fn sampleFieldHermite4(base: u32, p: vec3f) -> vec4f {
  let half = draw.half;
  let M = i32(draw.gridM);
  let last = max(M - 2, 0);
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let fx = chebIndex(xi.x); let fy = chebIndex(xi.y); let fz = chebIndex(xi.z);
  let x0 = clamp(i32(floor(fx)), 0, last);
  let y0 = clamp(i32(floor(fy)), 0, last);
  let z0 = clamp(i32(floor(fz)), 0, last);
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let tz = clamp(fz - f32(z0), 0.0, 1.0);

  let volN = draw.gridM * draw.gridM * draw.gridM;
  let bGx = base + volN;
  let bGy = base + 2u * volN;
  let bGz = base + 3u * volN;
  let dltX = nodeXi(x0 + 1) - nodeXi(x0);
  let dltY = nodeXi(y0 + 1) - nodeXi(y0);
  let dltZ = nodeXi(z0 + 1) - nodeXi(z0);

  var F: array<f32, 8>;
  var DX: array<f32, 8>;
  var DY: array<f32, 8>;
  var DZ: array<f32, 8>;
  for (var k: i32 = 0; k < 2; k++) {
    for (var j: i32 = 0; j < 2; j++) {
      for (var i: i32 = 0; i < 2; i++) {
        let idx = i + j * 2 + k * 4;
        let ix = x0 + i; let iy = y0 + j; let iz = z0 + k;
        F[idx] = densAtBase(base, ix, iy, iz);
        DX[idx] = densAtBase(bGx, ix, iy, iz) * dltX;
        DY[idx] = densAtBase(bGy, ix, iy, iz) * dltY;
        DZ[idx] = densAtBase(bGz, ix, iy, iz) * dltZ;
      }
    }
  }

  // Cell-wide mixed partials from FD of first derivatives (same for all corners).
  var dxy = 0.0;
  var dxz = 0.0;
  var dyz = 0.0;
  for (var j: i32 = 0; j < 2; j++) {
    for (var k: i32 = 0; k < 2; k++) {
      let a = j * 2 + k * 4;
      dxy += 0.25 * (DY[1 + a] - DY[0 + a]);
      dxz += 0.25 * (DZ[1 + a] - DZ[0 + a]);
    }
  }
  for (var i: i32 = 0; i < 2; i++) {
    for (var k: i32 = 0; k < 2; k++) {
      dyz += 0.25 * (DZ[i + 2 + k * 4] - DZ[i + k * 4]);
    }
  }
  // Also average the dual FD for dxy/dxz for symmetry.
  var dxy2 = 0.0;
  var dxz2 = 0.0;
  var dyz2 = 0.0;
  for (var i: i32 = 0; i < 2; i++) {
    for (var k: i32 = 0; k < 2; k++) {
      dxy2 += 0.25 * (DX[i + 2 + k * 4] - DX[i + k * 4]);
    }
    for (var j: i32 = 0; j < 2; j++) {
      dxz2 += 0.25 * (DX[i + j * 2 + 4] - DX[i + j * 2]);
      dyz2 += 0.25 * (DY[i + j * 2 + 4] - DY[i + j * 2]);
    }
  }
  dxy = 0.5 * (dxy + dxy2);
  dxz = 0.5 * (dxz + dxz2);
  dyz = 0.5 * (dyz + dyz2);
  let dxyz = 0.5 * (
    (DZ[1 + 2 + 0] - DZ[0 + 2 + 0]) - (DZ[1 + 0 + 0] - DZ[0 + 0 + 0])
    + (DZ[1 + 2 + 4] - DZ[0 + 2 + 4]) - (DZ[1 + 0 + 4] - DZ[0 + 0 + 4])
  ) * 0.5;

  // X pass → 4 values on the yz face (pack j+2*k).
  var U: array<f32, 4>;
  var UY: array<f32, 4>;
  var UZ: array<f32, 4>;
  var UYZ: array<f32, 4>;
  var UX: array<f32, 4>;
  var UXY: array<f32, 4>;
  var UXZ: array<f32, 4>;
  var UXYZ: array<f32, 4>;
  for (var k: i32 = 0; k < 2; k++) {
    for (var j: i32 = 0; j < 2; j++) {
      let pjk = j + k * 2;
      let i0 = 0 + j * 2 + k * 4;
      let i1 = 1 + j * 2 + k * 4;
      U[pjk] = hermiteVal(tx, F[i0], F[i1], DX[i0], DX[i1]);
      UY[pjk] = hermiteVal(tx, DY[i0], DY[i1], dxy, dxy);
      UZ[pjk] = hermiteVal(tx, DZ[i0], DZ[i1], dxz, dxz);
      UYZ[pjk] = hermiteVal(tx, dyz, dyz, dxyz, dxyz);
      UX[pjk] = hermiteDt(tx, F[i0], F[i1], DX[i0], DX[i1]);
      UXY[pjk] = hermiteDt(tx, DY[i0], DY[i1], dxy, dxy);
      UXZ[pjk] = hermiteDt(tx, DZ[i0], DZ[i1], dxz, dxz);
      UXYZ[pjk] = hermiteDt(tx, dyz, dyz, dxyz, dxyz);
    }
  }

  // Y pass → 2 values on the z edge.
  let V0 = hermiteVal(ty, U[0], U[1], UY[0], UY[1]);
  let V1 = hermiteVal(ty, U[2], U[3], UY[2], UY[3]);
  let VZ0 = hermiteVal(ty, UZ[0], UZ[1], UYZ[0], UYZ[1]);
  let VZ1 = hermiteVal(ty, UZ[2], UZ[3], UYZ[2], UYZ[3]);
  let VX0 = hermiteVal(ty, UX[0], UX[1], UXY[0], UXY[1]);
  let VX1 = hermiteVal(ty, UX[2], UX[3], UXY[2], UXY[3]);
  let VY0 = hermiteDt(ty, U[0], U[1], UY[0], UY[1]);
  let VY1 = hermiteDt(ty, U[2], U[3], UY[2], UY[3]);
  let VYZ0 = hermiteDt(ty, UZ[0], UZ[1], UYZ[0], UYZ[1]);
  let VYZ1 = hermiteDt(ty, UZ[2], UZ[3], UYZ[2], UYZ[3]);
  let VXZ0 = hermiteVal(ty, UXZ[0], UXZ[1], UXYZ[0], UXYZ[1]);
  let VXZ1 = hermiteVal(ty, UXZ[2], UXZ[3], UXYZ[2], UXYZ[3]);

  // Z pass → value + parametric derivatives.
  let f = hermiteVal(tz, V0, V1, VZ0, VZ1);
  let dF_dtz = hermiteDt(tz, V0, V1, VZ0, VZ1);
  let dF_dty = hermiteVal(tz, VY0, VY1, VYZ0, VYZ1);
  let dF_dtx = hermiteVal(tz, VX0, VX1, VXZ0, VXZ1);

  let invH = select(0.0, 1.0 / half, abs(half) > 1e-12);
  let invDltX = select(0.0, 1.0 / dltX, abs(dltX) > 1e-15);
  let invDltY = select(0.0, 1.0 / dltY, abs(dltY) > 1e-15);
  let invDltZ = select(0.0, 1.0 / dltZ, abs(dltZ) > 1e-15);
  return vec4f(
    f,
    dF_dtx * invDltX * invH,
    dF_dty * invDltY * invH,
    dF_dtz * invDltZ * invH,
  );
}

fn sampleVolume(p: vec3f) -> f32 {
${volumeSample}
}

fn fieldAt(p: vec3f) -> f32 {
  var d = sampleVolume(p);
  if (d != d) { d = 0.0; }
  return d - draw.isoLevel;
}

/** Blended dens at an integer Chebyshev-grid vertex (pre-iso). */
fn blendedDens(ix: i32, iy: i32, iz: i32) -> f32 {
  let a = densAtBase(u32(draw.volBase), ix, iy, iz);
  let b = densAtBase(u32(draw.volBaseB), ix, iy, iz);
  return mix(a, b, draw.blendT);
}

/** Lower corner of the trilinear cell containing p (in Chebyshev index space). */
fn cellIndexAt(p: vec3f) -> vec3i {
  let half = draw.half;
  let M = i32(draw.gridM);
  let last = max(M - 2, 0);
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let x0 = clamp(i32(floor(chebIndex(xi.x))), 0, last);
  let y0 = clamp(i32(floor(chebIndex(xi.y))), 0, last);
  let z0 = clamp(i32(floor(chebIndex(xi.z))), 0, last);
  return vec3i(x0, y0, z0);
}

/**
 * Final-hit filter: reject hits in cells whose 8 corner dens values do not
 * bracket isoLevel. Hermite can overshoot, so this is conservative (may drop
 * some valid zeros) but still kills most ghost-halo near-misses.
 */
fn cellBracketsIso(p: vec3f) -> bool {
  let c = cellIndexAt(p);
  let iso = draw.isoLevel;
  var lo = 1e30;
  var hi = -1e30;
  for (var dz: i32 = 0; dz < 2; dz++) {
    for (var dy: i32 = 0; dy < 2; dy++) {
      for (var dx: i32 = 0; dx < 2; dx++) {
        let v = blendedDens(c.x + dx, c.y + dy, c.z + dz) - iso;
        lo = min(lo, v);
        hi = max(hi, v);
      }
    }
  }
  return lo <= 0.0 && hi >= 0.0;
}

/** ∇f of the active iso interpolant (world space). */
fn fieldGrad(p: vec3f) -> vec3f {
${gradSample}
}

${GRADIENT_WGSL}

fn shadeIso(p: vec3f, rd: vec3f, n: vec3f) -> vec4f {
  var stops: array<vec4f, ${MAX_GRAD_STOPS}>;
  stops[0] = vec4f(draw.g0.xyz, draw.gradCount);
  stops[1] = draw.g1;
  stops[2] = draw.g2;
  stops[3] = draw.g3;
  stops[4] = draw.g4;
  stops[5] = draw.g5;

  let L = normalize(vec3f(0.35, 0.85, 0.45));
  let V = normalize(-rd);
  let ndotl = max(dot(n, L), 0.0);
  let ndotv = max(dot(n, V), 0.0);

  let tBase = isoGradientT(p, n, draw.half);
  let tLight = mix(0.04, 0.96, pow(ndotl, 0.7));
  let t = clamp(mix(tBase, tLight, 0.7), 0.0, 1.0);
  var rgb = sampleGradStops(&stops, t);

  let nStops = max(min(u32(draw.gradCount), MAX_GRAD_STOPS), 1u);
  let c0 = stops[0].xyz;
  let c1 = stops[nStops - 1u].xyz;

  // Mild fresnel — keep some rim without making silhouettes scream with view angle.
  let fresnel = pow(1.0 - ndotv, 2.4);
  rgb += c1 * fresnel * 0.2;

  // Specular on blended albedo
  let H = normalize(L + V);
  let spec = pow(max(dot(n, H), 0.0), 40.0);
  rgb += mix(c0, c1, 0.7) * spec * 0.45;

  return vec4f(rgb, 1.0);
}

fn isoNormal(p: vec3f, rd: vec3f) -> vec3f {
  // Evaluate ∇f slightly inside the fit domain — endpoint clamp makes
  // boundary gradients noisy (grain where isos meet the box faces).
  let half = draw.half;
  let inset = max(1e-4, 2e-3 * half);
  let pIn = clamp(p, vec3f(-half + inset), vec3f(half - inset));
  var g = fieldGrad(pIn);
  let gl = length(g);
  var n = select(vec3f(0.0, 1.0, 0.0), g / gl, gl > 1e-8);
  if (dot(n, -rd) < 0.0) { n = -n; }
  return n;
}

/**
 * Walk [t0,t1] and return the FIRST sign-change sub-bracket (near zero along the ray).
 * Returns (lo, hi, flo, fAtT1). Empty when hi < lo; fAtT1 is always fieldAt(t1).
 */
fn firstZeroBracket(ro: vec3f, rd: vec3f, t0: f32, t1: f32, f0: f32, samples: u32) -> vec4f {
  var prevT = t0;
  var prevF = f0;
  let n = max(samples, 1u);
  var fEnd = f0;
  for (var s: u32 = 1u; s <= n; s++) {
    let t = mix(t0, t1, f32(s) / f32(n));
    let f = fieldAt(ro + rd * t);
    fEnd = f;
    if (prevF * f <= 0.0) {
      return vec4f(prevT, t, prevF, f); // w unused on hit path
    }
    prevT = t;
    prevF = f;
  }
  return vec4f(t1, t0, f0, fEnd); // empty: hi < lo
}

/** Bisect an already-isolated nearest-zero bracket (HEAD-cost refine). */
fn bisectIso(ro: vec3f, rd: vec3f, lo: f32, hi: f32, flo: f32) -> f32 {
  var a = lo;
  var b = hi;
  var fa = flo;
  for (var k: u32 = 0u; k < 12u; k++) {
    let mid = 0.5 * (a + b);
    let fm = fieldAt(ro + rd * mid);
    if (fa * fm <= 0.0) { b = mid; } else { a = mid; fa = fm; }
  }
  return 0.5 * (a + b);
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
  var dfdT = 0.0;
  var hasDfdT = false;

  for (var i: u32 = 0u; i < 192u; i++) {
    if (i >= steps) { break; }
    let s1 = min(s0 + dt, tExit);
    let ds = max(s1 - s0, 1e-8);

    // Far from iso along the ray → HEAD path: one endpoint sample only.
    let distT = select(0.0, abs(f0) / max(abs(dfdT), 1e-8), hasDfdT);
    let farFromIso = hasDfdT && (distT > 2.0 * dt);

    var f1: f32;
    var hit = -1.0;
    if (farFromIso) {
      f1 = fieldAt(ro + rd * s1);
      if (f0 * f1 < 0.0) {
        hit = bisectIso(ro, rd, s0, s1, f0);
      }
    } else {
      // Near band (or first step): densify; reuse endpoint from the scan.
      let nDense = select(
        4u,
        u32(clamp(8.0 * dt / max(distT, 1e-6), 1.0, 8.0)),
        hasDfdT,
      );
      let br = firstZeroBracket(ro, rd, s0, s1, f0, nDense);
      f1 = br.w;
      if (br.y > br.x) {
        hit = bisectIso(ro, rd, br.x, br.y, br.z);
      }
    }

    if (hit >= 0.0) {
      let p = ro + rd * hit;
      if (cellBracketsIso(p)) {
        let d = clamp(hit / far, 0.0, 0.999);
        let n = isoNormal(p, rd);
        out.color = shadeIso(p, rd, n);
        out.occl = vec4f(d, 0.0, 0.0, 1.0);
        out.normal = vec4f(n * 0.5 + 0.5, 1.0);
        out.depth = d;
        return out;
      }
    }

    dfdT = (f1 - f0) / ds;
    hasDfdT = true;
    s0 = s1;
    f0 = f1;
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
  // Stay clear of the discontinuous clamped boundary.
  let edgeEps = (tExit - tEnter) * 2e-4 + 5e-5 * half;
  tEnter = tEnter + edgeEps;
  let tExitIn = tExit - edgeEps;
  if (!(tExitIn > tEnter + 1e-6)) {
    discard;
    return out;
  }
  let hit = marchIso(ro, rd, tEnter, tExitIn);
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
@group(0) @binding(3) var<storage, read> layerGrads: array<vec4f>;

${GRADIENT_WGSL}

fn sampleLayerGrad(L: u32, t: f32) -> vec3f {
  let base = L * MAX_GRAD_STOPS;
  var stops: array<vec4f, ${MAX_GRAD_STOPS}>;
  for (var i: u32 = 0u; i < MAX_GRAD_STOPS; i++) {
    stops[i] = layerGrads[base + i];
  }
  return sampleGradStops(&stops, t);
}

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
  let half = draw.half;
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
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
        let gt = gradientT(p, half);
        let col = sampleLayerGrad(L, gt);
        sigma += sig;
        emitAcc += col * sig;
      }
    }
    if (sigma > 1e-8) {
      let absorb = exp(-sigma * ds);
      let opacity = 1.0 - absorb;
      let col = emitAcc / sigma;
      // Beer emission + soft ambient so wispy low-density regions stay luminous.
      rgb += T * opacity * col * (1.0 + 0.42);
      T *= absorb;
    }
    s += dt;
  }
  let a = 1.0 - T;
  if (a < 0.001) { return vec4f(0.0); }
  return vec4f(rgb, a);
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
 * Camera-facing axis letter billboards; same ray-t depth encoding as the grid
 * so isosurfaces can occlude them.
 */
function makeAxisLabelWgsl() {
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
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlasSamp: sampler;

struct VSIn {
  @location(0) pos: vec3f,
  @location(1) uv: vec2f,
}

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
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
  o.uv = v.uv;
  return o;
}

@fragment
fn fsMain(in: VSOut) -> FSOut {
  var out: FSOut;
  let tex = textureSample(atlas, atlasSamp, in.uv);
  if (tex.a < 0.02) { discard; }
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
  out.color = vec4f(tex.rgb * tex.a, tex.a);
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
    // Reject silhouette / box-edge samples (large ray-t discontinuity).
    if (abs(tS - t0) > p.radius * 2.5) { continue; }
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
  // Occlusion: slide toward shadow gradient stop (c0-like), not grey darkening
  let tOcc = (1.0 - factor) * 0.55;
  let peak = max(max(color.r, color.g), color.b);
  let hue = color.rgb / max(peak, 1e-4);
  let shadowStop = hue * peak * 0.62;
  let rgb = mix(color.rgb, mix(shadowStop, color.rgb, factor), tOcc);
  return vec4f(rgb, color.a);
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
let labelPipeline = null;
let labelVertexBuf = null;
/** @type {GPUTexture | null} */
let labelAtlasTex = null;
/** @type {GPUSampler | null} */
let labelAtlasSamp = null;
let labelAtlasDirty = true;
/** Scratch for 3 billboard quads (18 verts × 6 floats). */
const labelVertScratch = new Float32Array(18 * 6);

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
/** @type {number[][][]} rgb stops per dens layer */
let densGradStops = [];
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

const PIPELINE_EPOCH = 23;
let builtEpoch = -1;

/**
 * Classic pack; volBase / volBaseB / blendT for GPU keyframe mix.
 * Gradient stops g0..g5 after blend fields (count in gradCount).
 * @param {number[][]} [gradRgbs] list of [r,g,b] stops
 */
function packDrawParamsIso(
  fbW, fbH, gridM, steps, half, scale, isoLevel, volBase, ro, M, absorb, emit,
  volBaseB = volBase, blendT = 0, gradRgbs = null,
) {
  const buf = new ArrayBuffer(256);
  const u32 = new Uint32Array(buf);
  const f32 = new Float32Array(buf);
  u32[0] = fbW | 0; u32[1] = fbH | 0; u32[2] = gridM | 0; u32[3] = steps | 0;
  f32[4] = half; f32[5] = scale; f32[6] = isoLevel; u32[7] = 1;
  f32[8] = ro[0]; f32[9] = ro[1]; f32[10] = ro[2];
  f32[11] = volBase;
  f32[12] = M[0]; f32[13] = M[1]; f32[14] = M[2];
  f32[16] = M[3]; f32[17] = M[4]; f32[18] = M[5];
  f32[20] = M[6]; f32[21] = M[7]; f32[22] = M[8];
  // volBaseB / blendT / gradCount at 24..26 (was absorb/emit slot)
  f32[24] = volBaseB;
  f32[25] = blendT;
  const stops = normalizeRgbStops(gradRgbs, absorb, emit);
  f32[26] = stops.length;
  f32[27] = 0;
  for (let i = 0; i < MAX_GRAD_STOPS; i++) {
    const c = stops[Math.min(i, stops.length - 1)];
    const o = 28 + i * 4;
    f32[o] = c[0]; f32[o + 1] = c[1]; f32[o + 2] = c[2]; f32[o + 3] = 1;
  }
  return buf;
}

/** @param {number[][] | null} gradRgbs @param {number[]} absorb @param {number[]} emit */
function normalizeRgbStops(gradRgbs, absorb, emit) {
  let stops = Array.isArray(gradRgbs) && gradRgbs.length
    ? gradRgbs.map((c) => [c[0], c[1], c[2]])
    : [absorb, emit];
  if (stops.length < 1) stops = [DEFAULT_ISO_RGB, DEFAULT_ISO_RGB2];
  if (stops.length === 1) stops = [stops[0], stops[0]];
  if (stops.length > MAX_GRAD_STOPS) stops = stops.slice(0, MAX_GRAD_STOPS);
  return stops;
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

function writeLayerColors(layerStopsList) {
  if (!device || !colorBuf) return;
  const data = new Float32Array(MAX_DENS_LAYERS * MAX_GRAD_STOPS * 4);
  for (let L = 0; L < MAX_DENS_LAYERS; L++) {
    const raw = layerStopsList?.[L];
    const stops = normalizeRgbStops(
      Array.isArray(raw) && raw.length && Array.isArray(raw[0]) ? raw : null,
      DEFAULT_DENS_RGB,
      DEFAULT_DENS_RGB2,
    );
    for (let i = 0; i < MAX_GRAD_STOPS; i++) {
      const c = stops[Math.min(i, stops.length - 1)];
      const o = (L * MAX_GRAD_STOPS + i) * 4;
      data[o] = c[0];
      data[o + 1] = c[1];
      data[o + 2] = c[2];
      data[o + 3] = i === 0 ? stops.length : 0;
    }
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

let themeGridMajor = 0x6b5a82;
let themeGridMinor = 0x3d2f55;
let themeBoxRgb = [0.35, 0.29, 0.45];
let themeAxisXRgb = [0.9, 0.35, 0.38];
let themeAxisYRgb = [0.35, 0.75, 0.48];
let themeAxisZRgb = [0.79, 0.66, 0.91];
let themeLabelStroke = "rgba(26, 18, 40, 0.58)";

/**
 * @param {{ gridMajor?: number, gridMinor?: number, boxEdgeRgb?: number[], axisXRgb?: number[], axisYRgb?: number[], axisZRgb?: number[], labelStroke?: string }} colors
 */
export function applyClipGpuTheme(colors) {
  if (colors.gridMajor) themeGridMajor = colors.gridMajor;
  if (colors.gridMinor) themeGridMinor = colors.gridMinor;
  if (colors.boxEdgeRgb) themeBoxRgb = colors.boxEdgeRgb;
  if (colors.axisXRgb) themeAxisXRgb = colors.axisXRgb;
  if (colors.axisYRgb) themeAxisYRgb = colors.axisYRgb;
  if (colors.axisZRgb) themeAxisZRgb = colors.axisZRgb;
  if (colors.labelStroke) themeLabelStroke = colors.labelStroke;
  gridHalf = -1;
  labelAtlasDirty = true;
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
  const [majR, majG, majB] = hexToRgb(themeGridMajor);
  const [minR, minG, minB] = hexToRgb(themeGridMinor);
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

function rgb01Css(rgb) {
  const r = Math.round(Math.min(1, Math.max(0, rgb[0])) * 255);
  const g = Math.round(Math.min(1, Math.max(0, rgb[1])) * 255);
  const b = Math.round(Math.min(1, Math.max(0, rgb[2])) * 255);
  return `rgb(${r},${g},${b})`;
}

/** Bake x/y/z glyphs into a 3-cell atlas (theme-colored). */
function ensureAxisLabelAtlas() {
  if (!device) return;
  if (!labelAtlasDirty && labelAtlasTex && labelAtlasSamp) return;

  const cell = 128;
  const canvasEl = document.createElement("canvas");
  canvasEl.width = cell * 3;
  canvasEl.height = cell;
  const c2d = canvasEl.getContext("2d");
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

  if (!labelAtlasTex) {
    labelAtlasTex = device.createTexture({
      size: [canvasEl.width, canvasEl.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  device.queue.copyExternalImageToTexture(
    { source: canvasEl },
    { texture: labelAtlasTex },
    [canvasEl.width, canvasEl.height],
  );
  if (!labelAtlasSamp) {
    labelAtlasSamp = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
  }
  if (!labelVertexBuf) {
    labelVertexBuf = device.createBuffer({
      size: labelVertScratch.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
  }
  labelAtlasDirty = false;
}

/**
 * Camera-facing x/y/z billboards at axis tips (world size ~0.42).
 * @param {{ matrixWorld: { elements: ArrayLike<number> } }} camera
 * @param {number} half
 */
function uploadAxisLabelBillboards(camera, half) {
  ensureAxisLabelAtlas();
  if (!device || !labelVertexBuf) return 0;

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
  device.queue.writeBuffer(labelVertexBuf, 0, labelVertScratch);
  return vi;
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
    device && isoPipeline && beerPipeline && fxaaPipeline && ssaoPipeline && gridPipeline && labelPipeline,
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
    isoInterp: isoInterpHermite ? "hermite" : "trilinear",
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

/** @param {{ color?: number[], color2?: number[], colors?: number[][] }} d */
function stopsFromLayer(d) {
  if (Array.isArray(d?.colors) && d.colors.length) {
    return d.colors.map((c) => [c[0], c[1], c[2]]);
  }
  return normalizeRgbStops(
    null,
    d?.color || DEFAULT_DENS_RGB,
    d?.color2 || DEFAULT_DENS_RGB2,
  );
}

/** Update density layer gradient colors without re-baking volumes.
 * @param {number[][][]} layerStopsList per-layer list of [r,g,b] stops
 */
export function uploadSceneColors(layerStopsList) {
  densGradStops = (layerStopsList || []).slice(0, MAX_DENS_LAYERS).map((stops) =>
    normalizeRgbStops(
      Array.isArray(stops) && stops.length && Array.isArray(stops[0]) ? stops : null,
      DEFAULT_DENS_RGB,
      DEFAULT_DENS_RGB2,
    ),
  );
  writeLayerColors(densGradStops);
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
  densGradStops = dens.map((d) => stopsFromLayer(d));

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
    const stops = stopsFromLayer(c);
    return {
      id: c.id || null,
      color: stops[0],
      color2: stops[stops.length - 1],
      colors: stops,
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
  writeLayerColors(densGradStops);

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
        isoPipeline = beerPipeline = fxaaPipeline = ssaoPipeline = gridPipeline = labelPipeline = null;
        labelAtlasTex = null;
        labelAtlasSamp = null;
        labelVertexBuf = null;
        labelAtlasDirty = true;
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
        size: MAX_DENS_LAYERS * MAX_GRAD_STOPS * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      writeLayerColors([[DEFAULT_DENS_RGB, DEFAULT_DENS_RGB2]]);
      ensureVolumeBuf(8 * 8 * 8);
      await ensurePipelinesForDegree(4);
      if (viewportEl) attachClipGpuCanvas(viewportEl);
      if (canvas) ctx = canvas.getContext("webgpu");
      return isClipBakeGpuReady();
    } catch (e) {
      console.warn("[clipBakeGpu] init failed", e);
      initFailed = true;
      device = null;
      isoPipeline = beerPipeline = fxaaPipeline = ssaoPipeline = gridPipeline = labelPipeline = null;
      labelAtlasTex = null;
      labelAtlasSamp = null;
      labelVertexBuf = null;
      labelAtlasDirty = true;
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
    isoPipeline && beerPipeline && fxaaPipeline && ssaoPipeline && gridPipeline && labelPipeline &&
    builtEpoch === PIPELINE_EPOCH
  ) {
    return true;
  }

  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const isoMod = await compileChecked("iso", makeIsoWgsl());
  const beerMod = await compileChecked("beer", makeBeerMultiWgsl());
  const gridMod = await compileChecked("grid", makeGridWgsl());
  const labelMod = await compileChecked("axisLabel", makeAxisLabelWgsl());
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
  const nextLabel = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: labelMod,
      entryPoint: "vsMain",
      buffers: [{
        arrayStride: 24,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 16, format: "float32x2" },
        ],
      }],
    },
    fragment: {
      module: labelMod,
      entryPoint: "fsMain",
      targets: [{ format: canvasFormat, blend: blendPremul }],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: {
      format: "depth32float",
      depthWriteEnabled: false,
      depthCompare: "less",
    },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`axisLabel: ${err.message}`);
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
  labelPipeline = nextLabel;
  ssaoPipeline = nextSsao;
  fxaaPipeline = nextFxaa;
  builtEpoch = PIPELINE_EPOCH;
  labelAtlasDirty = true;
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
  writeLayerColors(densGradStops);

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

  // World grid / axes / box / axis labels — depth-test against iso depth in-place.
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
    const labelVertCount = labelPipeline
      ? uploadAxisLabelBillboards(camera, h)
      : 0;
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

    if (labelPipeline && labelVertexBuf && labelAtlasTex && labelAtlasSamp && labelVertCount > 0) {
      const labelBg = device.createBindGroup({
        layout: labelPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gridParamBuf } },
          { binding: 1, resource: labelAtlasTex.createView() },
          { binding: 2, resource: labelAtlasSamp },
        ],
      });
      pass.setPipeline(labelPipeline);
      pass.setBindGroup(0, labelBg);
      pass.setVertexBuffer(0, labelVertexBuf);
      pass.draw(labelVertCount);
    }

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
