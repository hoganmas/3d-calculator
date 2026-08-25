/**
 * WebGPU clip-grid path (golden): tile-parallel dens bake (Chebyshev nodes +
 * Clenshaw fill; exact dens for narrow tiles) + Beer–Lambert fullscreen march.
 * See research/poly/notes/clip-space-babbage.md.
 *
 * CPU/WebGL fallback: main.js via bakeClipGridFibers + DataTexture.
 */

import {
  ndcToDirMatrix,
  perspectiveDirScale,
  viewFiberWindow,
  MAX_DEG,
} from "./clipGrid.js";

/** Absolute caps (pipeline arrays specialize to active fit deg ≤ this). */
const MAX_N = MAX_DEG + 1;
const MAX_1D_N = 3 * MAX_DEG + 1;
export const MAX_COEFFS = MAX_N * MAX_N * MAX_N;
/** Default max tile width for amortized dens fill (auto shrinks when far). */
const GPU_BABBAGE_TILE = 1024;
const WG_SIZE = 64;

/**
 * Tile width for GPU dens bake.
 * @param {number} deg
 * @param {number} max1d  3N
 * @param {number | null | undefined | "exact"} tileOverride
 *   null/`auto`: distance-adaptive (shrink when the fit box is a small screen footprint).
 *   `exact`: tile=D → per-pixel exact dens.
 *   positive: force that tile width (≥ D+1 for Clenshaw path).
 * @param {{ half?: number, tMid?: number, sx?: number, sy?: number, width?: number, height?: number }} [view]
 */
function gpuBabbageTile(deg, max1d, tileOverride = null, view = null) {
  const D = Math.max(1, max1d | 0);

  if (tileOverride === "exact") {
    return D; // span < D → exact dens path
  }
  if (typeof tileOverride === "number" && Number.isFinite(tileOverride) && tileOverride > 0) {
    return Math.max(D + 1, tileOverride | 0);
  }

  // Auto: match tile to projected fit-box size. Far away the box is a few pixels
  // of signal in a field of exterior zeros — large tiles waste Clenshaw on empty
  // space and soften the bump. Near: keep large tiles (amortize seeds).
  const half = view?.half ?? 2;
  const tMid = Math.max(Math.abs(view?.tMid ?? half), 1e-6);
  const sx = Math.max(view?.sx ?? 1, 1e-6);
  const sy = Math.max(view?.sy ?? 1, 1e-6);
  const w = Math.max(1, view?.width | 0);
  const h = Math.max(1, view?.height | 0);
  // NDC half-extent of a world offset `half` at depth tMid (perspective chart).
  const ndcRx = half / (tMid * sx);
  const ndcRy = half / (tMid * sy);
  const pixSpan = Math.min(ndcRx * w, ndcRy * h); // full box width in atlas px
  // Cover the footprint with some slack; never below D+1, never above default.
  const target = Math.ceil(pixSpan * 25.0);
  return Math.max(D + 1, Math.min(GPU_BABBAGE_TILE, target));
}

/** Per-degree sizes — WGSL locals/workgroup memory match fit deg (no max-N tax). */
function degSizes(deg) {
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
  fillMode: u32, // 0 = Chebyshev+Clenshaw, 1 = equispaced Newton (Babbage)
  half: f32,
  tMid: f32,
  tHw: f32,
  _pad1: f32,
  // Fiber mid-point on the *center* ray, computed CPU-side in f64:
  //   anchor = ro + tMid·rdCenter,  rdCenter = M·(0,0,1).
  // Bake seeds use p = anchor + tMid·(rd−rdC) + (t−tMid)·rd so far-camera
  // cancellation is not done in f32.
  anchor: vec3f,
  _pad2: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coeffs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outAtlas: array<f32>;
@group(0) @binding(3) var<storage, read_write> diffStore: array<f32>;

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

/**
 * Stable world point on a pixel ray when the camera is far from the fit box.
 * Algebraically equal to ro + t*rd, but the large cancel happens on the CPU
 * when forming anchor; GPU only adds O(box)-sized terms.
 */
fn rayPoint(rd: vec3f, t: f32) -> vec3f {
  let tMid = params.tMid;
  let rdC = vec3f(params.m0.z, params.m1.z, params.m2.z);
  return params.anchor + (rd - rdC) * tMid + rd * (t - tMid);
}

fn rayDir(px: f32, py: f32) -> vec3f {
  let width = params.width;
  let height = params.height;
  let ndcX = -1.0 + (2.0 / f32(width)) * (px + 0.5);
  let ndcY = -1.0 + (2.0 / f32(height)) * (py + 0.5);
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

/** T_k(ξ_j) at Chebyshev roots ξ_j = cos(π(j+½)/M). */
fn chebTAtRoot(k: u32, j: u32, M: u32) -> f32 {
  return cos(3.141592653589793 * f32(k) * (f32(j) + 0.5) / f32(M));
}

fn writePixel(px: u32, py: u32, dens: ptr<function, array<f32, MAX_1D_N>>, nAlpha: u32) {
  let width = params.width;
  let height = params.height;
  let half = params.half;
  let rd = rayDir(f32(px), f32(py));
  for (var j: u32 = 0u; j < MAX_1D_N; j++) {
    if (j >= nAlpha) { break; }
    let t = params.tMid + params.tHw * chebRoot(j, nAlpha);
    let p = rayPoint(rd, t);
    let inside = abs(p.x) <= half && abs(p.y) <= half && abs(p.z) <= half;
    var v = select(0.0, (*dens)[j], inside);
    if (v != v) { v = 0.0; }
    v = clamp(v, -CAP, CAP);
    outAtlas[(j * height + py) * width + px] = v;
  }
}

/**
 * Dens seed at pixel (px,py): evaluate in box-normalized coords ξ = p/half ∈ [-1,1]^3.
 * Coeffs are pre-scaled ĉ_ijk = c_ijk half^{i+j+k} so eval(ĉ, ξ) = f(p).
 * Outside the fit box → 0 before Chebyshev/Newton (exterior world powers poison Δ/DCT).
 */
fn exactDensAt(px: f32, py: f32, dens: ptr<function, array<f32, MAX_1D_N>>, deg: u32, nAlpha: u32) {
  let rd = rayDir(px, py);
  let half = params.half;
  let invH = 1.0 / half;
  for (var j: u32 = 0u; j < MAX_1D_N; j++) {
    if (j >= nAlpha) {
      (*dens)[j] = 0.0;
      continue;
    }
    let t = params.tMid + params.tHw * chebRoot(j, nAlpha);
    let p = rayPoint(rd, t);
    if (abs(p.x) > half || abs(p.y) > half || abs(p.z) > half) {
      (*dens)[j] = 0.0;
      continue;
    }
    (*dens)[j] = evalMonomial3D(deg, p * invH);
  }
}

fn tileSpan(tx: u32, py: u32) -> vec4i {
  let tile = params.tile;
  let xBase = max(0i, params.xGrid0 + i32(tx * tile));
  let xEnd = min(i32(params.width), params.xGrid0 + i32((tx + 1u) * tile));
  return vec4i(xBase, xEnd, 0, 0);
}

/** Map pixel index to Chebyshev parameter u∈[-1,1] on [xBase, xEnd). */
fn pixelToChebU(px: f32, xFirst: f32, xLast: f32) -> f32 {
  let hw = 0.5 * (xLast - xFirst);
  if (abs(hw) < 1e-8) { return 0.0; }
  return clamp((px - 0.5 * (xFirst + xLast)) / hw, -1.0, 1.0);
}

/** Clenshaw: Σ_{k=0}^D c_k T_k(x), coeffs in diffStore[base..]. */
fn clenshawStore(base: u32, D: u32, x: f32) -> f32 {
  var b1 = 0.0;
  var b2 = 0.0;
  for (var k: i32 = i32(D); k >= 1; k--) {
    let b0 = diffStore[base + u32(k)] + 2.0 * x * b1 - b2;
    b2 = b1;
    b1 = b0;
  }
  return diffStore[base] + x * b1 - b2;
}

fn coarseStep(span: u32, D: u32) -> u32 {
  if (span <= D) { return 1u; }
  let hNeed = max(1u, (span + D - 1u) / D);
  var h = 1u;
  while (h < hNeed) { h = h * 2u; }
  return h;
}

fn computeOrigin(xBase: i32, xEnd: i32, D: u32) -> vec2u {
  let span = u32(xEnd - xBase - 1);
  let h = coarseStep(span, D);
  let cover = D * h;
  var xOrigin = xBase + ((i32(span) - i32(cover)) >> 1);
  if (xOrigin < 0) { xOrigin = 0; }
  if (xOrigin + i32(cover) >= i32(params.width)) {
    xOrigin = max(0i, i32(params.width) - i32(cover));
  }
  return vec2u(u32(xOrigin), h);
}

fn evalNewtonStore(tileRow: u32, j: u32, D: u32, t: f32) -> f32 {
  let base = tileRow * SEED_DIFF_N + j * MAX_1D_N;
  var acc = diffStore[base];
  var binom = 1.0;
  for (var k: u32 = 1u; k <= D; k++) {
    binom = binom * (t - f32(k - 1u)) / f32(k);
    acc += binom * diffStore[base + k];
  }
  return acc;
}

/** Pass A: seeds + Chebyshev coeffs or Newton Δ (exact tiles write atlas). */
@compute @workgroup_size(${WG_SIZE}, 1, 1)
fn bakeSeedMain(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid3: vec3u,
) {
  let lid = lid3.x;
  let tx = wid.x;
  let py = wid.y;
  if (py >= params.height || tx >= params.nTilesX) { return; }

  let deg = params.deg;
  let D = params.max1d;
  let nAlpha = D + 1u;
  let M = D + 1u;
  let useNewton = params.fillMode == 1u;
  let bounds = tileSpan(tx, py);
  let xBase = bounds.x;
  let xEnd = bounds.y;
  if (xBase >= xEnd) { return; }

  let span = u32(xEnd - xBase - 1);
  let tileRow = py * params.nTilesX + tx;

  if (span < D) {
    var densExact: array<f32, MAX_1D_N>;
    for (var px: u32 = u32(xBase) + lid; px < u32(xEnd); px += WG_SIZE) {
      exactDensAt(f32(px), f32(py), &densExact, deg, nAlpha);
      writePixel(px, py, &densExact, nAlpha);
    }
    return;
  }

  if (useNewton) {
    let oh = computeOrigin(xBase, xEnd, D);
    let xOrigin = i32(oh.x);
    let h = oh.y;
    if (lid <= D) {
      var densSeed: array<f32, MAX_1D_N>;
      exactDensAt(f32(xOrigin + i32(lid * h)), f32(py), &densSeed, deg, nAlpha);
      for (var j: u32 = 0u; j < MAX_1D_N; j++) {
        if (j >= nAlpha) { break; }
        seedSamples[j * MAX_1D_N + lid] = densSeed[j];
      }
    }
    workgroupBarrier();
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
      let outBase = tileRow * SEED_DIFF_N + base;
      for (var i2: u32 = 0u; i2 <= D; i2++) {
        diffStore[outBase + i2] = diffTable[base + i2];
      }
    }
    return;
  }

  // Chebyshev + Clenshaw
  let xFirst = f32(xBase);
  let xLast = f32(xEnd - 1);
  if (lid <= D) {
    let xi = chebRoot(lid, M);
    let px = 0.5 * (xFirst + xLast) + 0.5 * (xLast - xFirst) * xi;
    var densSeed: array<f32, MAX_1D_N>;
    exactDensAt(px, f32(py), &densSeed, deg, nAlpha);
    for (var j: u32 = 0u; j < MAX_1D_N; j++) {
      if (j >= nAlpha) { break; }
      seedSamples[j * MAX_1D_N + lid] = densSeed[j];
    }
  }
  workgroupBarrier();
  if (lid < nAlpha) {
    let base = lid * MAX_1D_N;
    let invM = 1.0 / f32(M);
    for (var k: u32 = 0u; k <= D; k++) {
      var s = 0.0;
      for (var i: u32 = 0u; i <= D; i++) {
        s += seedSamples[base + i] * chebTAtRoot(k, i, M);
      }
      let scale = select(invM, 2.0 * invM, k > 0u);
      diffTable[base + k] = scale * s;
    }
    let outBase = tileRow * SEED_DIFF_N + base;
    for (var i2: u32 = 0u; i2 <= D; i2++) {
      diffStore[outBase + i2] = diffTable[base + i2];
    }
  }
}

/** Pass B: Clenshaw or Newton fill (no-op for exact tiles). */
@compute @workgroup_size(${WG_SIZE}, 1, 1)
fn bakeFillMain(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid3: vec3u,
) {
  let lid = lid3.x;
  let tx = wid.x;
  let py = wid.y;
  if (py >= params.height || tx >= params.nTilesX) { return; }

  let D = params.max1d;
  let nAlpha = D + 1u;
  let useNewton = params.fillMode == 1u;
  let bounds = tileSpan(tx, py);
  let xBase = bounds.x;
  let xEnd = bounds.y;
  if (xBase >= xEnd) { return; }

  let span = u32(xEnd - xBase - 1);
  if (span < D) { return; }

  let tileRow = py * params.nTilesX + tx;
  var dens: array<f32, MAX_1D_N>;

  if (useNewton) {
    let oh = computeOrigin(xBase, xEnd, D);
    let xOrigin = i32(oh.x);
    let h = oh.y;
    for (var px: u32 = u32(xBase) + lid; px < u32(xEnd); px += WG_SIZE) {
      let t = (f32(i32(px) - xOrigin)) / f32(h);
      for (var j: u32 = 0u; j < MAX_1D_N; j++) {
        if (j >= nAlpha) {
          dens[j] = 0.0;
          continue;
        }
        dens[j] = evalNewtonStore(tileRow, j, D, t);
      }
      writePixel(px, py, &dens, nAlpha);
    }
    return;
  }

  let xFirst = f32(xBase);
  let xLast = f32(xEnd - 1);
  for (var px: u32 = u32(xBase) + lid; px < u32(xEnd); px += WG_SIZE) {
    let u = pixelToChebU(f32(px), xFirst, xLast);
    for (var j: u32 = 0u; j < MAX_1D_N; j++) {
      if (j >= nAlpha) {
        dens[j] = 0.0;
        continue;
      }
      dens[j] = clenshawStore(tileRow * SEED_DIFF_N + j * MAX_1D_N, D, u);
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
  anchor: vec3f,
  _p4: f32,
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

/** T_k at Chebyshev root ξ_j = cos(π(j+½)/M). Same as bake pass. */
fn chebTAtRoot(k: u32, j: u32, M: u32) -> f32 {
  return cos(3.141592653589793 * f32(k) * (f32(j) + 0.5) / f32(M));
}

/** Nodal samples at Cheb u-roots → Cheb coeffs (discrete transform). */
fn uChebDCT(
  samples: ptr<function, array<f32, MAX_1D_N>>,
  coeff: ptr<function, array<f32, MAX_1D_N>>,
  nA: u32,
) {
  let D = nA - 1u;
  let M = nA;
  let invM = 1.0 / f32(M);
  for (var k: u32 = 0u; k < MAX_1D_N; k++) {
    if (k > D) {
      (*coeff)[k] = 0.0;
      continue;
    }
    var s = 0.0;
    for (var j: u32 = 0u; j < MAX_1D_N; j++) {
      if (j >= M) { break; }
      s += (*samples)[j] * chebTAtRoot(k, j, M);
    }
    let scale = select(invM, 2.0 * invM, k > 0u);
    (*coeff)[k] = scale * s;
  }
}

/** Σ_{k=0..D} c_k T_k(u) via Clenshaw (matches bake x-fill). */
fn clenshaw1D(coeff: ptr<function, array<f32, MAX_1D_N>>, D: u32, u: f32) -> f32 {
  var b1 = 0.0;
  var b2 = 0.0;
  for (var k: i32 = i32(D); k >= 1; k--) {
    let b0 = (*coeff)[u32(k)] + 2.0 * u * b1 - b2;
    b2 = b1;
    b1 = b0;
  }
  return (*coeff)[0] + u * b1 - b2;
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

  var uCoeff: array<f32, MAX_1D_N>;
  let D = draw.max1d;
  var dMin = densSamp[0];
  var dMax = densSamp[0];
  if (nA > 1u) {
    uChebDCT(&densSamp, &uCoeff, nA);
    dMin = densSamp[0];
    dMax = densSamp[0];
    for (var k: u32 = 1u; k < nA; k++) {
      dMin = min(dMin, densSamp[k]);
      dMax = max(dMax, densSamp[k]);
    }
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
    let p = draw.anchor + (rd - vec3f(draw.m0.z, draw.m1.z, draw.m2.z)) * tMid + rd * (s - tMid);

    var dval = 0.0;
    let inBox = abs(p.x) <= half && abs(p.y) <= half && abs(p.z) <= half;
    let inU = abs(u) <= 1.0;
    if (inBox && inU) {
      if (nA > 1u) {
        dval = clenshaw1D(&uCoeff, D, u);
        dval = clamp(dval, dMin, dMax);
      } else {
        dval = densSamp[0];
      }
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
let bakeSeedPipeline = null;
/** @type {GPUComputePipeline | null} */
let bakeFillPipeline = null;
/** Shared explicit layout so seed+fill share one bind group. */
/** @type {GPUBindGroupLayout | null} */
let bakeBindGroupLayout = null;
/** @type {GPUPipelineLayout | null} */
let bakePipelineLayout = null;
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
let atlasBack = null; // unused (kept for retire compatibility)
let atlasCapacity = 0;
/** @type {GPUBuffer | null} */
let diffBuf = null;
let diffCapacity = 0;
/** @type {GPUBindGroup | null} */
let marchBindGroup = null;

/** Timestamp profiling (seed / fill / march). */
let timestampsSupported = false;
/** @type {GPUQuerySet | null} */
let stampQuerySet = null;
/** @type {GPUBuffer | null} */
let stampResolveBuf = null;
/** @type {GPUBuffer | null} */
let stampReadBuf = null;
let stampReadPending = false;
let profileSeedMs = 0;
let profileFillMs = 0;
let profileMarchMs = 0;
let profileMarchFbW = 0;
let profileMarchFbH = 0;
let profilePresentWallMs = 0;
let profilePresentIntervalMs = 0;
let lastPresentAt = 0;
let profileMethod = "";
let profileTile = 0;
let profileNTilesX = 0;

/** @type {HTMLCanvasElement | null} */
let canvas = null;
/** @type {GPUCanvasContext | null} */
let ctx = null;
let canvasFormat = "bgra8unorm";
let initPromise = null;
let initFailed = false;

/** Pack Bake Params (must match BAKE_WGSL Params; size 112, buffer 128). */
function packBakeParams(
  width,
  height,
  deg,
  max1d,
  tile,
  xGrid0,
  nTilesX,
  fillMode,
  half,
  tMid,
  tHw,
  anchor,
  M,
) {
  const buf = new ArrayBuffer(128);
  const u32 = new Uint32Array(buf, 0, 8);
  u32[0] = width;
  u32[1] = height;
  u32[2] = deg;
  u32[3] = max1d;
  u32[4] = tile;
  u32[6] = nTilesX;
  u32[7] = fillMode | 0;
  new Int32Array(buf, 0, 8)[5] = xGrid0 | 0;
  const f32 = new Float32Array(buf, 32);
  f32[0] = half;
  f32[1] = tMid;
  f32[2] = tHw;
  f32[3] = 0;
  f32[4] = anchor[0];
  f32[5] = anchor[1];
  f32[6] = anchor[2];
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
  const anchor = state.anchor || state.ro;
  f32[28] = anchor[0];
  f32[29] = anchor[1];
  f32[30] = anchor[2];
  f32[31] = 0;
  return buf;
}

export function isClipBakeGpuReady() {
  return Boolean(device && bakeSeedPipeline && bakeFillPipeline && marchPipeline);
}

/** True when WebGPU clip pipelines + canvas are ready for a per-frame bake+march. */
export function isClipMarchReady() {
  return Boolean(device && bakeSeedPipeline && bakeFillPipeline && marchPipeline && ctx);
}

/** Wall-clock from queue.submit until onSubmittedWorkDone (includes GPU + driver). */
function noteGpuPresent(submitWallAt) {
  const now = performance.now();
  profilePresentWallMs =
    profilePresentWallMs * 0.85 + (now - submitWallAt) * 0.15;
  if (lastPresentAt > 0) {
    profilePresentIntervalMs =
      profilePresentIntervalMs * 0.85 + (now - lastPresentAt) * 0.15;
  } else {
    profilePresentIntervalMs = now - submitWallAt;
  }
  lastPresentAt = now;
}

/** Latest GPU timestamp profile (ms). Zeros if timestamps unavailable / not yet resolved. */
export function getClipGpuProfile() {
  return {
    seedMs: profileSeedMs,
    fillMs: profileFillMs,
    marchMs: profileMarchMs,
    marchFbW: profileMarchFbW,
    marchFbH: profileMarchFbH,
    presentWallMs: profilePresentWallMs,
    presentIntervalMs: profilePresentIntervalMs,
    lastPresentAt,
    method: profileMethod,
    tile: profileTile,
    nTilesX: profileNTilesX,
    timestamps: timestampsSupported,
  };
}

/** Clear smoothed GPU timestamps (e.g. when march scale changes). */
export function resetClipGpuProfile() {
  profileSeedMs = 0;
  profileFillMs = 0;
  profileMarchMs = 0;
  profileMarchFbW = 0;
  profileMarchFbH = 0;
}

/**
 * Create/attach the WebGPU canvas inside the viewport (under HUD, over Three).
 */
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
      timestampsSupported = adapter.features.has("timestamp-query");
      const requiredFeatures = [];
      if (timestampsSupported) requiredFeatures.push("timestamp-query");
      device = await adapter.requestDevice({ requiredFeatures });
      device.lost.then(() => {
        device = null;
        bakeSeedPipeline = null;
        bakeFillPipeline = null;
        marchPipeline = null;
        pipelineDeg = null;
        initFailed = true;
      });

      if (timestampsSupported) {
        stampQuerySet = device.createQuerySet({ type: "timestamp", count: 6 });
        stampResolveBuf = device.createBuffer({
          size: 6 * 8,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        stampReadBuf = device.createBuffer({
          size: 6 * 8,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
      }

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
      bakeSeedPipeline = null;
      bakeFillPipeline = null;
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
  if (pipelineDeg === sz.deg && bakeSeedPipeline && bakeFillPipeline && marchPipeline) {
    return true;
  }

  const bakeMod = device.createShaderModule({ code: makeBakeWgsl(sz) });

  bakeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  bakePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bakeBindGroupLayout],
  });

  device.pushErrorScope("validation");
  const nextSeed = device.createComputePipeline({
    layout: bakePipelineLayout,
    compute: { module: bakeMod, entryPoint: "bakeSeedMain" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`bake seed pipeline deg=${sz.deg}: ${err.message}`);
  }
  device.pushErrorScope("validation");
  const nextFill = device.createComputePipeline({
    layout: bakePipelineLayout,
    compute: { module: bakeMod, entryPoint: "bakeFillMain" },
  });
  {
    const err = await device.popErrorScope();
    if (err) throw new Error(`bake fill pipeline deg=${sz.deg}: ${err.message}`);
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

  bakeSeedPipeline = nextSeed;
  bakeFillPipeline = nextFill;
  marchPipeline = nextMarch;
  pipelineDeg = sz.deg;
  marchBindGroup = null;
  bakeBindGroup = null;
  uploadedCoeffDeg = -1;
  uploadedCoeffHalf = NaN;
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
let uploadedCoeffHalf = NaN;
let uploadedCoeffRef = null;

function ensureDiffBuf(floatCount) {
  const aligned = Math.max(256, Math.ceil((floatCount * 4) / 256) * 256);
  if (diffBuf && diffCapacity >= aligned) return;
  if (aligned > storageBufferBudget()) {
    throw new Error(`diff ${aligned}B exceeds GPU storage budget`);
  }
  const old = diffBuf;
  diffBuf = device.createBuffer({
    size: aligned,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  diffCapacity = aligned;
  bakeBindGroup = null;
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
  if (!bakeBindGroupLayout || !atlasFront || !diffBuf) return null;
  if (bakeBindGroup) return bakeBindGroup;
  bakeBindGroup = device.createBindGroup({
    layout: bakeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: bakeParamBuf } },
      { binding: 1, resource: { buffer: coeffBuf } },
      { binding: 2, resource: { buffer: atlasFront } },
      { binding: 3, resource: { buffer: diffBuf } },
    ],
  });
  return bakeBindGroup;
}

function invalidateAtlasBindGroups() {
  marchBindGroup = null;
  bakeBindGroup = null;
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
      if (stamps[1] > stamps[0]) profileSeedMs = profileSeedMs * 0.7 + ns(0, 1) * 0.3;
      if (stamps[3] > stamps[2]) profileFillMs = profileFillMs * 0.7 + ns(2, 3) * 0.3;
      if (stamps[5] > stamps[4]) profileMarchMs = profileMarchMs * 0.7 + ns(4, 5) * 0.3;
    })
    .catch(() => {
      stampReadPending = false;
    });
}

/**
 * Per-frame GPU path: seed pass → fill pass → march, with optional timestamps.
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
  tileOverride = null,
  /** @type {"chebyshev" | "newton"} */
  fillMode = "chebyshev",
}) {
  if (!device || !bakeSeedPipeline || !bakeFillPipeline || !marchPipeline || !ctx || !worldMono) {
    return false;
  }
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
  // Center-ray world point at tMid — f64 cancel, then upload as f32 O(box) anchor.
  const rdCenter = [M[2], M[5], M[8]];
  const anchor = [
    ro[0] + tMid * rdCenter[0],
    ro[1] + tMid * rdCenter[1],
    ro[2] + tMid * rdCenter[2],
  ];

  const tile = gpuBabbageTile(d, max1d, tileOverride, {
    half: h,
    tMid,
    sx,
    sy,
    width,
    height,
  });
  const nTilesX = Math.max(1, Math.ceil(width / tile));
  const xGrid0 = -Math.floor((nTilesX * tile - width) / 2);
  const exactPath = tile <= max1d;
  const fillModeU32 = fillMode === "newton" ? 1 : 0;

  const outBytes = width * height * nAlpha * 4;
  const prevAtlas = atlasFront;
  ensureAtlasPair(outBytes);
  ensureDiffBuf(nTilesX * height * sz.seedDiffN);
  if (!atlasFront || !diffBuf) return false;
  if (atlasFront !== prevAtlas) invalidateAtlasBindGroups();

  // Upload ĉ_ijk = c_ijk · half^{i+j+k} so GPU seeds eval at ξ = p/half ∈ [-1,1]^3.
  if (
    uploadedCoeffRef !== worldMono ||
    uploadedCoeffDeg !== d ||
    uploadedCoeffHalf !== h
  ) {
    const coeffs = new Float32Array(sz.maxCoeffs);
    const n = d + 1;
    for (let i = 0; i <= d; i++) {
      const hi = h ** i;
      for (let j = 0; j <= d; j++) {
        const hij = hi * h ** j;
        for (let k = 0; k <= d; k++) {
          const idx = i + j * n + k * n * n;
          coeffs[idx] = (worldMono[idx] || 0) * hij * h ** k;
        }
      }
    }
    device.queue.writeBuffer(coeffBuf, 0, coeffs);
    uploadedCoeffRef = worldMono;
    uploadedCoeffDeg = d;
    uploadedCoeffHalf = h;
  }

  const method = exactPath
    ? "gpu-exact-dens"
    : fillModeU32 === 1
      ? "gpu-babbage-newton"
      : "gpu-cheb-clenshaw";

  const state = {
    width,
    height,
    nAlpha,
    max1d,
    deg: d,
    M,
    ro,
    anchor,
    half: h,
    tMid,
    tHw,
    sx,
    sy,
    tile,
    fillMode: fillModeU32 === 1 ? "newton" : "chebyshev",
    method,
    gpuResident: true,
  };
  profileMethod = state.method;
  profileTile = tile;
  profileNTilesX = nTilesX;

  device.queue.writeBuffer(
    bakeParamBuf,
    0,
    packBakeParams(
      width,
      height,
      d,
      max1d,
      tile,
      xGrid0,
      nTilesX,
      fillModeU32,
      h,
      tMid,
      tHw,
      anchor,
      M,
    ),
  );
  const bakeBind = bindBakeToFront();
  if (!marchBindGroup) bindMarchToFront();
  if (!bakeBind || !marchBindGroup) return false;

  // Resize WebGPU canvas before packing march uniforms (must match fragment count).
  resizeClipGpuCanvas(fbW, fbH);
  const marchW = canvas?.width ?? fbW;
  const marchH = canvas?.height ?? fbH;
  profileMarchFbW = marchW;
  profileMarchFbH = marchH;
  device.queue.writeBuffer(
    drawParamBuf,
    0,
    packDrawParams(state, marchW, marchH, scale, steps, absorb, emit),
  );

  const enc = device.createCommandEncoder();
  const useStamps = timestampsSupported && stampQuerySet;

  {
    const pass = enc.beginComputePass(
      useStamps
        ? {
            timestampWrites: {
              querySet: stampQuerySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          }
        : undefined,
    );
    pass.setPipeline(bakeSeedPipeline);
    pass.setBindGroup(0, bakeBind);
    pass.dispatchWorkgroups(nTilesX, height, 1);
    pass.end();
  }

  {
    const pass = enc.beginComputePass(
      useStamps
        ? {
            timestampWrites: {
              querySet: stampQuerySet,
              beginningOfPassWriteIndex: 2,
              endOfPassWriteIndex: 3,
            },
          }
        : undefined,
    );
    pass.setPipeline(bakeFillPipeline);
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
      ...(useStamps
        ? {
            timestampWrites: {
              querySet: stampQuerySet,
              beginningOfPassWriteIndex: 4,
              endOfPassWriteIndex: 5,
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
    enc.resolveQuerySet(stampQuerySet, 0, 6, stampResolveBuf, 0);
    enc.copyBufferToBuffer(stampResolveBuf, 0, stampReadBuf, 0, 48);
  }

  const submitWallAt = performance.now();
  device.queue.submit([enc.finish()]);
  void device.queue.onSubmittedWorkDone().then(() => {
    noteGpuPresent(submitWallAt);
    if (useStamps) scheduleStampReadback();
  });
  return true;
}
