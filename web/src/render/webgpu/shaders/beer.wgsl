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
  flowLayerStart: f32,
  debugTier: f32,
  nearEdgeActive: f32,
  _p4: f32,
  densBlend: array<vec4f, 8>,
  densT: array<vec4f, 2>,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var occlIsoTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> layerGrads: array<vec4f>;

{{GRADIENT_WGSL}}

fn sampleLayerGradAt(L: u32, t: f32) -> vec3f {
  let base = L * MAX_GRAD_STOPS;
  var stops: array<vec4f, {{MAX_GRAD_STOPS}}>;
  for (var i: u32 = 0u; i < MAX_GRAD_STOPS; i++) {
    stops[i] = layerGrads[base + i];
  }
  return sampleGradStops(&stops, t);
}

fn sampleLayerGrad(L: u32, t: f32) -> vec3f {
  let base = L * MAX_GRAD_STOPS;
  var stops: array<vec4f, {{MAX_GRAD_STOPS}}>;
  for (var i: u32 = 0u; i < MAX_GRAD_STOPS; i++) {
    stops[i] = layerGrads[base + i];
  }
  return sampleGradStops(&stops, t);
}

struct VSOut { @builtin(position) pos: vec4f, }

struct FSOut {
  @location(0) color: vec4f,
  @location(1) occl: vec4f,
}

const OCCL_ALPHA: f32 = 0.15;

/** isoRefineDebug tint: keep the sample's luminance, swap in a tier color. */
fn beerDebugTint(c: vec4f, rgb: vec3f) -> vec4f {
  if (c.a < 0.001) { return c; }
  let lin = c.rgb / max(c.a, 1e-4);
  let lum = clamp(dot(lin, vec3f(0.299, 0.587, 0.114)), 0.08, 1.0);
  return vec4f(rgb * lum * c.a, c.a);
}

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

fn densBlendT(L: u32) -> f32 {
  let v = draw.densT[select(0u, 1u, L >= 4u)];
  let i = L % 4u;
  if (i == 0u) { return v.x; }
  if (i == 1u) { return v.y; }
  if (i == 2u) { return v.z; }
  return v.w;
}

fn isoOcclusionForVolumePixel(pos: vec2f, fbW: f32, fbH: f32) -> f32 {
  let isoDims = textureDimensions(occlIsoTex);
  let isoW = f32(isoDims.x);
  let isoH = f32(isoDims.y);
  let ux = clamp(pos.x / fbW, 0.0, 1.0);
  let uy = clamp(pos.y / fbH, 0.0, 1.0);
  if (isoW <= fbW + 0.5 && isoH <= fbH + 0.5) {
    if (isoW > fbW - 0.5 && isoH > fbH - 0.5) {
      // Same-res compose: exact 1:1 texel match, nearest is correct as-is.
      let ix = u32(min(floor(ux * isoW), isoW - 1.0));
      let iy = u32(min(floor(uy * isoH), isoH - 1.0));
      return textureLoad(occlIsoTex, vec2u(ix, iy), 0).r;
    }
    // Coarser occupancy stretched over many beer pixels (16× interiors / 4×
    // mid beer): a nearest lookup here held one clip depth flat across an
    // entire coarse texel, so smoothly-varying interior depth stepped in
    // hard per-tile blocks that only became visible once bilinear-upscaled
    // to compose res (dense/thin banding "repeating" with the tile grid).
    // Bilinear across the 2×2 neighborhood fixes that — except right at a
    // silhouette, where blending a hit depth with a miss (1.0) would forge a
    // bogus mid-value clip; bail to unclipped there since mixed footprints
    // remarch at a finer tier anyway.
    let sx = clamp(ux * isoW - 0.5, 0.0, isoW - 1.0);
    let sy = clamp(uy * isoH - 0.5, 0.0, isoH - 1.0);
    let x0 = u32(floor(sx));
    let y0 = u32(floor(sy));
    let x1 = min(x0 + 1u, u32(isoW) - 1u);
    let y1 = min(y0 + 1u, u32(isoH) - 1u);
    let tx = sx - f32(x0);
    let ty = sy - f32(y0);
    let d00 = textureLoad(occlIsoTex, vec2u(x0, y0), 0).r;
    let d10 = textureLoad(occlIsoTex, vec2u(x1, y0), 0).r;
    let d01 = textureLoad(occlIsoTex, vec2u(x0, y1), 0).r;
    let d11 = textureLoad(occlIsoTex, vec2u(x1, y1), 0).r;
    if (d00 >= 0.999 || d10 >= 0.999 || d01 >= 0.999 || d11 >= 0.999) {
      return 1.0;
    }
    // Unswapped: standard bilinear order (an extra U-axis inversion on top
    // of the prior U-swap cancels back to this).
    return mix(mix(d00, d10, tx), mix(d01, d11, tx), ty);
  }
  // Clip tex finer than beer: clip only fully-covered interior tiles (min depth).
  // Mixed footprints stay unclipped and remarch at compose res.
  let x0 = i32(clamp(floor(pos.x / fbW * isoW), 0.0, isoW - 1.0));
  let y0 = i32(clamp(floor(pos.y / fbH * isoH), 0.0, isoH - 1.0));
  var x1 = i32(clamp(ceil((pos.x + 1.0) / fbW * isoW) - 1.0, 0.0, isoW - 1.0));
  var y1 = i32(clamp(ceil((pos.y + 1.0) / fbH * isoH) - 1.0, 0.0, isoH - 1.0));
  x1 = max(x0, x1);
  y1 = max(y0, y1);
  var dMin = 1.0;
  var dMax = 0.0;
  let spanX = min(x1 - x0, 7);
  let spanY = min(y1 - y0, 7);
  for (var iy = 0; iy < 8; iy++) {
    if (iy > spanY) { break; }
    for (var ix = 0; ix < 8; ix++) {
      if (ix > spanX) { break; }
      let d = textureLoad(occlIsoTex, vec2u(u32(x0 + ix), u32(y0 + iy)), 0).r;
      dMin = min(dMin, d);
      dMax = max(dMax, d);
    }
  }
  if (dMin >= 0.999 || dMax >= 0.999) { return 1.0; }
  return dMin;
}

// Box-path fraction (of one edge length) below which a ray counts as
// "grazing" the box — near the silhouette, regardless of which face/edge.
const BEER_NEAR_EDGE_PATH_FRAC: f32 = 0.12;
// Box scaled up by this fraction when checking for near-misses just outside
// the true bounds — a ray that only clips the expanded box is treated the
// same as a grazing hit (path length 0 < the threshold above).
const BEER_NEAR_EDGE_MISS_MARGIN: f32 = 0.08;

/**
 * True near the box's screen-space silhouette. A grazing ray — one whose
 * path through the box is short relative to the box's own edge length —
 * naturally covers the whole silhouette outline (any face, any angle), not
 * just literal 3D edges: cross a single face at a shallow angle and the
 * path is short too. A ray that misses the true box but clips a slightly
 * expanded one counts as well (path length 0), covering pixels just outside
 * the bounds. Cheap/downres beer marches alias most in this region (a thin,
 * grazing cross-section of the box, or the abrupt cutoff at its edge), so
 * treat it like a coarse-mixed tile and force the finer beer tiers.
 */
fn beerNearBoxEdgeAtNdc(ndcX: f32, ndcY: f32) -> bool {
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(draw.m0.xyz, xy1), dot(draw.m1.xyz, xy1), dot(draw.m2.xyz, xy1));
  let ro = draw.ro; let half = max(draw.half, 1e-6);
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let halfExp = half * (1.0 + BEER_NEAR_EDGE_MISS_MARGIN);
  let tAExp = (-vec3f(halfExp) - ro) * invRd;
  let tBExp = (vec3f(halfExp) - ro) * invRd;
  let tminExp = min(tAExp, tBExp); let tmaxExp = max(tAExp, tBExp);
  let tEnterExp = max(max(max(tminExp.x, tminExp.y), tminExp.z), 0.0);
  let tExitExp = min(min(tmaxExp.x, tmaxExp.y), tmaxExp.z);
  // Doesn't even clip the expanded box — nowhere near the bounds.
  if (!(tExitExp > tEnterExp + 1e-6)) { return false; }
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB); let tmax = max(tA, tB);
  let tEnter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  let pathLen = max(0.0, tExit - tEnter); // 0 when the true box is missed
  return pathLen < BEER_NEAR_EDGE_PATH_FRAC * (2.0 * half);
}

/**
 * Single sample at this pixel's own center — matches blit.wgsl's
 * blitNearBoxEdge exactly, both evaluated at compose resolution. Only the
 * final beer tier calls this (draw.nearEdgeActive is false for the mid
 * pass): the mid composite always defers exact near-edge pixels to final
 * (see fsMainSwap in blit.wgsl), so mid never has a reason to claim them
 * itself, undilated or otherwise — an earlier dilated version of this,
 * meant to give the mid tier its own safety cushion, ended up double-
 * compositing with the cheap layer instead once mid stopped keeping
 * near-edge pixels at all.
 */
fn beerNearBoxEdge(pos: vec2f, fbW: f32, fbH: f32) -> bool {
  let ndcX = -1.0 + 2.0 * pos.x / fbW;
  let ndcY = 1.0 - 2.0 * pos.y / fbH;
  return beerNearBoxEdgeAtNdc(ndcX, ndcY);
}

fn marchBeer(pos: vec2f) -> FSOut {
  var out: FSOut;
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);

  let fbW = f32(draw.fbW); let fbH = f32(draw.fbH);
  let ndcX = -1.0 + 2.0 * pos.x / fbW;
  let ndcY = 1.0 - 2.0 * pos.y / fbH;
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
  if (!(tExit > tEnter + 1e-6)) {
    out.color = vec4f(0.0);
    return out;
  }

  // Must match isoHermite.wgsl's normalization exactly — isoD below is read
  // straight out of that pass's occl.r.
  let far = max(tExit - tEnter, half * 4.0);
  // Same-res: clip beer to iso. Finer iso: leave tExit alone (isoD == 1).
  let isoD = isoOcclusionForVolumePixel(pos, fbW, fbH);
  if (isoD < 0.999) { tExit = min(tExit, tEnter + isoD * far); }
  if (!(tExit > tEnter + 1e-6)) {
    out.color = vec4f(0.0);
    out.occl = vec4f(isoD, 0.0, 0.0, 1.0);
    return out;
  }

  var steps = draw.steps;
  if (steps < 8u) { steps = 8u; }
  if (steps > 96u) { steps = 96u; }
  let dt = (tExit - tEnter) / f32(steps);
  let ds = length(rd) * dt;
  let volN = draw.gridM * draw.gridM * draw.gridM;
  let densBase = u32(draw.densBase);
  let nLay = min(draw.layerCount, {{MAX_DENS_LAYERS}}u);

  var rgb = vec3f(0.0); var T = 1.0; var s = tEnter + 0.5 * dt;
  var densD = 1.0;
  for (var i: u32 = 0u; i < 96u; i++) {
    if (i >= steps) { break; }
    if (T < 0.002) { break; }
    let p = ro + rd * s;
    var sigma = 0.0; var emitAcc = vec3f(0.0);
    for (var L: u32 = 0u; L < {{MAX_DENS_LAYERS}}u; L++) {
      if (L >= nLay) { break; }
      var dval = 0.0;
      let packed = draw.densBlend[L];
      let stride = u32(packed.y);
      if (stride > 0u) {
        let base0 = u32(packed.x) + u32(packed.z) * stride;
        let bt = densBlendT(L);
        if (bt <= 1e-5 || packed.z == packed.w) {
          dval = sampleLayer(base0, p);
        } else if (bt >= 0.999) {
          dval = sampleLayer(u32(packed.x) + u32(packed.w) * stride, p);
        } else {
          let base1 = u32(packed.x) + u32(packed.w) * stride;
          dval = mix(sampleLayer(base0, p), sampleLayer(base1, p), bt);
        }
      } else {
        dval = sampleLayer(densBase + L * volN, p);
      }
      if (dval != dval) { dval = 0.0; }
      var col: vec3f;
      let isFlow = f32(L) >= draw.flowLayerStart && draw.flowLayerStart >= 0.0;
      if (isFlow) { continue; }
      let gt = gradientT(p, half);
      col = sampleLayerGrad(L, gt);
      dval = clamp(dval, -4.0, 8.0);
      let sig = min(max(0.0, draw.scale * dval), 40.0);
      if (sig > 1e-8) {
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
      let alpha = 1.0 - T;
      if (densD >= 0.999 && alpha >= OCCL_ALPHA) {
        densD = clamp((s - tEnter) / far, 0.0, 0.999);
      }
    }
    s += dt;
  }
  let a = 1.0 - T;
  // Deferred iso clip stores volume depth only; same-res still mins with isoD.
  out.occl = vec4f(min(isoD, densD), 0.0, 0.0, 1.0);
  if (a < 0.001) {
    out.color = vec4f(0.0);
    return out;
  }
  out.color = vec4f(rgb, a);
  if (draw.debugTier > 0.5) {
    var tint = vec3f(0.15, 0.82, 1.0); // cheap — cyan, matches iso's lo/mid tint
    if (draw.debugTier > 2.5) { tint = vec3f(1.0, 0.25, 0.85); } // final — magenta
    else if (draw.debugTier > 1.5) { tint = vec3f(1.0, 0.6, 0.1); } // mid — orange
    out.color = beerDebugTint(out.color, tint);
  }
  return out;
}

@fragment
fn fsMain(in: VSOut) -> FSOut {
  return marchBeer(in.pos.xy);
}
