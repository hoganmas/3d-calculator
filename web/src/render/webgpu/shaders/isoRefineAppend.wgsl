{{ISO_EDGE_WGSL}}

@group(0) @binding(2) var coarseOccl: texture_2d<f32>;

@fragment
fn fsRefine(in: VSOut) -> FSOut {
  var out: FSOut;
  out.color = vec4f(0.0);
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);
  out.normal = vec4f(0.0);
  out.depth = 1.0;
  if (!isoNeedRefine(coarseOccl, f32(draw.fbW), f32(draw.fbH), in.pos.xy)) {
    discard;
    return out;
  }
  var hit = marchPixel(in.pos.xy);
  if (draw.debugTint > 0.5 && hit.color.a > 0.001) {
    hit.color = isoDebugTint(hit.color, vec3f(1.0, 0.36, 0.08));
  }
  return hit;
}
