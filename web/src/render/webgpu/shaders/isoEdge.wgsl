const ISO_OCC_HIT: f32 = 0.999;
const ISO_DEPTH_CREASE: f32 = 0.02;
const ISO_REFINE_NONE: u32 = 0u;
const ISO_REFINE_EDGE: u32 = 1u;
const ISO_REFINE_INTERSECT: u32 = 2u;

/** Fine pixel → coarse texel-center coords (same mapping as upsample bilinear). */
fn isoCoarseTexel(occlTex: texture_2d<f32>, fineW: f32, fineH: f32, fragPos: vec2f) -> vec2f {
  let dims = vec2f(textureDimensions(occlTex));
  let fw = max(fineW, 1.0);
  let fh = max(fineH, 1.0);
  return vec2f(fragPos.x * dims.x / fw, fragPos.y * dims.y / fh) - 0.5;
}

/** 0 = keep coarse, 1 = occupancy/depth edge, 2 = two isos meet in the 2×2. */
fn isoRefineKind(occlTex: texture_2d<f32>, fineW: f32, fineH: f32, fragPos: vec2f) -> u32 {
  let dims = textureDimensions(occlTex);
  let src = isoCoarseTexel(occlTex, fineW, fineH, fragPos);
  let x0 = i32(floor(src.x));
  let y0 = i32(floor(src.y));
  var nHit2 = 0u;
  var nHit4 = 0u;
  var dMin = 1.0;
  var dMax = 0.0;
  var id0 = 0.0;
  var mixedIso = false;
  var nearBound = false;
  // 4×4 = bilinear 2×2 plus a 1-texel ring. On mobile the coarse texel is
  // several screen pixels, so the ring is the crawling cyan band inside the
  // box silhouette.
  for (var iy = -1; iy <= 2; iy++) {
    for (var ix = -1; ix <= 2; ix++) {
      let x = u32(clamp(x0 + ix, 0, i32(dims.x) - 1));
      let y = u32(clamp(y0 + iy, 0, i32(dims.y) - 1));
      let inFoot = (ix >= 0 && ix <= 1 && iy >= 0 && iy <= 1);
      let o = textureLoad(occlTex, vec2u(x, y), 0);
      let d = o.r;
      if (d < ISO_OCC_HIT) {
        nHit4 += 1u;
        if (inFoot) {
          nHit2 += 1u;
          dMin = min(dMin, d);
          dMax = max(dMax, d);
          let id = round(o.g);
          if (id >= 0.5) {
            if (id0 < 0.5) {
              id0 = id;
            } else if (abs(id - id0) > 0.5) {
              mixedIso = true;
            }
          }
          if (o.b > 0.5) { nearBound = true; }
        }
      } else if (inFoot) {
        dMax = max(dMax, 1.0);
      }
    }
  }
  if (mixedIso) { return ISO_REFINE_INTERSECT; }
  if (nHit4 > 0u && nHit4 < 16u) { return ISO_REFINE_EDGE; }
  if (nHit2 == 4u && (dMax - dMin) > ISO_DEPTH_CREASE) { return ISO_REFINE_EDGE; }
  if (nearBound) { return ISO_REFINE_EDGE; }
  return ISO_REFINE_NONE;
}

fn isoNeedRefine(occlTex: texture_2d<f32>, fineW: f32, fineH: f32, fragPos: vec2f) -> bool {
  return isoRefineKind(occlTex, fineW, fineH, fragPos) != ISO_REFINE_NONE;
}

/** Keep lighting, swap albedo — cyan coarse / orange edge / magenta iso-x in debug mode. */
fn isoDebugTint(c: vec4f, rgb: vec3f) -> vec4f {
  if (c.a < 0.001) { return c; }
  let lin = c.rgb / max(c.a, 1e-4);
  let lum = clamp(dot(lin, vec3f(0.299, 0.587, 0.114)), 0.08, 1.0);
  return vec4f(rgb * lum * c.a, c.a);
}
