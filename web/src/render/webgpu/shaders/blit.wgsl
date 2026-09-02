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
  // Occupancy-refine tiles remarch beer at compose res. Interior iso keeps
  // the volume-res beer (already tExit-clipped to iso in beer.wgsl).
  if (isoNeedRefine(occTex, destW, destH, in.pos.xy)) {
    return vec4f(0.0);
  }
  return beer;
}
