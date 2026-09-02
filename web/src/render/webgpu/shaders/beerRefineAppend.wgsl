{{ISO_EDGE_WGSL}}

@group(0) @binding(4) var occTex: texture_2d<f32>;

@fragment
fn fsRefine(in: VSOut) -> @location(0) vec4f {
  // Occupancy tiles (16× / 4×) that mix iso and empty remarch beer at compose
  // res with iso tExit clip. Other tiles keep the cheap volume-res beer.
  if (!isoNeedRefine(occTex, f32(draw.fbW), f32(draw.fbH), in.pos.xy)) {
    discard;
    return vec4f(0.0);
  }
  return marchBeer(in.pos.xy).color;
}
