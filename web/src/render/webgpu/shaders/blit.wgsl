{{ISO_EDGE_WGSL}}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var occlIsoTex: texture_2d<f32>;
@group(0) @binding(3) var occTex: texture_2d<f32>;

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
  // them (16× mixed → 4× beer, 4× mixed → 1× beer).
  if (isoNeedRefine(occTex, destW, destH, in.pos.xy)) {
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
  if (isoNeedRefine(occTex, destW, destH, in.pos.xy)) {
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
