{{ISO_EDGE_WGSL}}

@group(0) @binding(4) var occTex: texture_2d<f32>;

@fragment
fn fsRefine(in: VSOut) -> @location(0) vec4f {
  // Occupancy-mixed tiles remarch beer at this pass's resolution with iso
  // tExit clip (4× mid or 1× compose). Other tiles keep a coarser beer pass.
  // Box-silhouette tiles remarch too, but only for the final pass
  // (draw.nearEdgeActive is false for the mid call) — the cheap layer
  // already deferred them (blit.wgsl), and the mid layer's own composite
  // mask (fsMainSwap) always defers them again to final rather than ever
  // keeping mid's answer, so mid claiming them here would just be wasted
  // work with no pixel that ever gets composited from it.
  let fbW = f32(draw.fbW);
  let fbH = f32(draw.fbH);
  let nearEdge = draw.nearEdgeActive > 0.5 && beerNearBoxEdge(in.pos.xy, fbW, fbH);
  if (!isoNeedRefine(occTex, fbW, fbH, in.pos.xy) && !nearEdge) {
    discard;
    return vec4f(0.0);
  }
  return marchBeer(in.pos.xy).color;
}
