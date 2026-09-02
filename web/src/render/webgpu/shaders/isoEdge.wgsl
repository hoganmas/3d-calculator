const ISO_OCC_HIT: f32 = 0.999;
const ISO_DEPTH_CREASE: f32 = 0.02;

/** Fine pixel → coarse texel-center coords (same mapping as upsample bilinear). */
fn isoCoarseTexel(occlTex: texture_2d<f32>, fineW: f32, fineH: f32, fragPos: vec2f) -> vec2f {
  let dims = vec2f(textureDimensions(occlTex));
  let fw = max(fineW, 1.0);
  let fh = max(fineH, 1.0);
  return vec2f(fragPos.x * dims.x / fw, fragPos.y * dims.y / fh) - 0.5;
}

/** Mixed occupancy or depth crease in the 2×2 coarse occl taps covering this fine pixel. */
fn isoNeedRefine(occlTex: texture_2d<f32>, fineW: f32, fineH: f32, fragPos: vec2f) -> bool {
  let dims = textureDimensions(occlTex);
  let src = isoCoarseTexel(occlTex, fineW, fineH, fragPos);
  let x0 = i32(floor(src.x));
  let y0 = i32(floor(src.y));
  var nHit = 0u;
  var dMin = 1.0;
  var dMax = 0.0;
  for (var iy = 0; iy < 2; iy++) {
    for (var ix = 0; ix < 2; ix++) {
      let x = u32(clamp(x0 + ix, 0, i32(dims.x) - 1));
      let y = u32(clamp(y0 + iy, 0, i32(dims.y) - 1));
      let d = textureLoad(occlTex, vec2u(x, y), 0).r;
      if (d < ISO_OCC_HIT) {
        nHit += 1u;
        dMin = min(dMin, d);
        dMax = max(dMax, d);
      } else {
        dMax = max(dMax, 1.0);
      }
    }
  }
  if (nHit > 0u && nHit < 4u) { return true; }
  if (nHit == 4u && (dMax - dMin) > ISO_DEPTH_CREASE) { return true; }
  return false;
}

/** Keep lighting, swap albedo — cyan coarse / orange refine in debug mode. */
fn isoDebugTint(c: vec4f, rgb: vec3f) -> vec4f {
  if (c.a < 0.001) { return c; }
  let lin = c.rgb / max(c.a, 1e-4);
  let lum = clamp(dot(lin, vec3f(0.299, 0.587, 0.114)), 0.08, 1.0);
  return vec4f(rgb * lum * c.a, c.a);
}

