{{ISO_EDGE_WGSL}}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var occlIsoTex: texture_2d<f32>;
@group(0) @binding(3) var occTex: texture_2d<f32>;

// Prefix of beer.wgsl's DrawParams (same buffer, same field offsets) — just
// enough to recompute the ray/box test exactly. fsMain-only; fsMainSwap
// doesn't reference it, so its auto layout omits binding 4.
struct BeerBoxParams {
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
}
@group(0) @binding(4) var<uniform> beerBox: BeerBoxParams;

const BLIT_NEAR_EDGE_PATH_FRAC: f32 = 0.12;
const BLIT_NEAR_EDGE_MISS_MARGIN: f32 = 0.08;

/** Same test as beer.wgsl's beerNearBoxEdgeAtNdc, duplicated since blit.wgsl
 *  isn't concatenated with beer.wgsl — evaluated here at compose
 *  (destW/destH) resolution so it's phase-exact with the mid/final tiers'
 *  own recompute of it (a smooth analytic ray/box test, not a quantized
 *  texel lookup, so it classifies the same physical location identically at
 *  any resolution). A grazing ray (short path through the box, any
 *  face/edge) or a near-miss against a slightly expanded box both count —
 *  see beer.wgsl's doc comment for why. Keep the two constants above in
 *  sync with BEER_NEAR_EDGE_PATH_FRAC / BEER_NEAR_EDGE_MISS_MARGIN. */
fn blitNearBoxEdge(pos: vec2f, fbW: f32, fbH: f32) -> bool {
  let ndcX = -1.0 + 2.0 * pos.x / fbW;
  let ndcY = 1.0 - 2.0 * pos.y / fbH;
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(beerBox.m0.xyz, xy1), dot(beerBox.m1.xyz, xy1), dot(beerBox.m2.xyz, xy1));
  let ro = beerBox.ro;
  let half = max(beerBox.half, 1e-6);
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let halfExp = half * (1.0 + BLIT_NEAR_EDGE_MISS_MARGIN);
  let tAExp = (-vec3f(halfExp) - ro) * invRd;
  let tBExp = (vec3f(halfExp) - ro) * invRd;
  let tminExp = min(tAExp, tBExp); let tmaxExp = max(tAExp, tBExp);
  let tEnterExp = max(max(max(tminExp.x, tminExp.y), tminExp.z), 0.0);
  let tExitExp = min(min(tmaxExp.x, tmaxExp.y), tmaxExp.z);
  if (!(tExitExp > tEnterExp + 1e-6)) { return false; }
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB); let tmax = max(tA, tB);
  let tEnter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  let pathLen = max(0.0, tExit - tEnter);
  return pathLen < BLIT_NEAR_EDGE_PATH_FRAC * (2.0 * half);
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.uv = vec2f(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
  return o;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let beer = textureSample(srcTex, srcSamp, in.uv);
  let isoDims = textureDimensions(occlIsoTex);
  let destW = f32(isoDims.x);
  let destH = f32(isoDims.y);
  // Leave occupancy-refine tiles empty: the matching beer cascade remarchs
  // them (16× mixed → 4× beer, 4× mixed → 1× beer). Also defer box-silhouette
  // pixels to the finer beer tiers — the cheap layer's downres march aliases
  // most right at the box edge.
  let nearEdge = blitNearBoxEdge(in.pos.xy, destW, destH);
  if (nearEdge || isoNeedRefine(occTex, destW, destH, in.pos.xy)) {
    return vec4f(0.0);
  }
  return beer;
}

// Manual bilinear over srcTex for the mid (second-lowest / 4×) beer cascade's
// color upscale. midBeerTex is cleared to (0,0,0,0) outside the coarse-mixed
// tiles it actually shaded, so a plain bilinear fetch straddling that
// boundary would blend real premultiplied color with hard zero — thinning
// alpha into a ring. Guard against that by falling back to the true nearest
// corner (by tx/ty, not always (x0,y0)) whenever the 2×2 neighborhood mixes
// a shaded and a cleared texel — picking the wrong corner here previously
// forged transparency for every fragment whose nearest sample was actually
// shaded but (x0,y0) wasn't.
@fragment
fn fsMainSwap(in: VSOut) -> @location(0) vec4f {
  let isoDims = textureDimensions(occlIsoTex);
  let destW = f32(isoDims.x);
  let destH = f32(isoDims.y);
  // Defer box-silhouette pixels to the final (compose-res, highest-quality)
  // tier instead of settling for the mid tier's 4× — same hand-off pattern
  // as isoNeedRefine below: this uses the exact same undilated test the
  // final pass's own gate uses, so the two are perfectly complementary (no
  // gap, no double-composite) — mid still shaded a dilated cushion around
  // this via beerNearBoxEdge, so pixels just outside this exact boundary
  // still get composited from mid below.
  if (blitNearBoxEdge(in.pos.xy, destW, destH) || isoNeedRefine(occTex, destW, destH, in.pos.xy)) {
    return vec4f(0.0);
  }
  let srcDims = vec2f(textureDimensions(srcTex));
  let sx = clamp(in.uv.x * srcDims.x - 0.5, 0.0, srcDims.x - 1.0);
  let sy = clamp(in.uv.y * srcDims.y - 0.5, 0.0, srcDims.y - 1.0);
  let x0 = u32(floor(sx));
  let y0 = u32(floor(sy));
  let x1 = min(x0 + 1u, u32(srcDims.x) - 1u);
  let y1 = min(y0 + 1u, u32(srcDims.y) - 1u);
  let tx = sx - f32(x0);
  let ty = sy - f32(y0);
  let n00 = textureLoad(srcTex, vec2u(x0, y0), 0);
  let n10 = textureLoad(srcTex, vec2u(x1, y0), 0);
  let n01 = textureLoad(srcTex, vec2u(x0, y1), 0);
  let n11 = textureLoad(srcTex, vec2u(x1, y1), 0);
  let minA = min(min(n00.a, n10.a), min(n01.a, n11.a));
  let maxA = max(max(n00.a, n10.a), max(n01.a, n11.a));
  if (maxA > 0.001 && minA < 0.001) {
    let nx = select(x0, x1, tx >= 0.5);
    let ny = select(y0, y1, ty >= 0.5);
    return textureLoad(srcTex, vec2u(nx, ny), 0);
  }
  return mix(mix(n00, n10, tx), mix(n01, n11, tx), ty);
}
