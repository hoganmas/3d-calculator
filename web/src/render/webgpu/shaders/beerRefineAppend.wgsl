{{ISO_EDGE_WGSL}}

@group(0) @binding(4) var occTex: texture_2d<f32>;

@fragment
fn fsRefine(in: VSOut) -> @location(0) vec4f {
  // Occupancy-mixed tiles remarch beer at this pass's resolution with iso
  // tExit clip (4× mid or 1× compose). Other tiles keep a coarser beer pass.
  if (!isoNeedRefine(occTex, f32(draw.fbW), f32(draw.fbH), in.pos.xy)) {
    discard;
    return vec4f(0.0);
  }
  return marchBeer(in.pos.xy).color;
}
