{{ISO_EDGE_WGSL}}

@group(0) @binding(4) var occTex: texture_2d<f32>;

@fragment
fn fsRefine(in: VSOut) -> @location(0) vec4f {
  // Occupancy-mixed tiles remarch beer at this pass's resolution with iso
  // tExit clip (4× mid or 1× compose). Other tiles keep a coarser beer pass.
  // Box-silhouette tiles remarch too, regardless of iso occupancy — the
  // cheap layer already deferred them (blit.wgsl). Both mid and final claim
  // near-edge here (draw.nearEdgeActive is unconditionally true for both
  // callers) the same way both can claim an iso-mixed tile: the mid layer's
  // own composite mask (fsMainSwap in blit.wgsl) decides whether mid's
  // answer is kept or handed off to final, using the same undilated test
  // final's gate uses below — so the two stay complementary, letting
  // silhouette pixels actually reach full compose resolution instead of
  // being stuck at mid's 4×.
  let fbW = f32(draw.fbW);
  let fbH = f32(draw.fbH);
  // draw.dilateNdc: the mid pass evaluates this test on its own (coarser)
  // pixel grid, which can disagree near the boundary with the compose-res
  // grid blit.wgsl/the final pass use for the same physical location — not
  // from texel quantization (this is a continuous test) but from the two
  // grids' pixel centers landing on opposite sides of the boundary. The
  // caller passes half this pass's own NDC pixel width for the mid call
  // (dilating across its own footprint, see beerNearBoxEdge) and 0 for the
  // final call (exact match with blit.wgsl, no coarser grid to disagree with).
  let nearEdge = draw.nearEdgeActive > 0.5
    && beerNearBoxEdge(in.pos.xy, fbW, fbH, draw.dilateNdc);
  if (!isoNeedRefine(occTex, fbW, fbH, in.pos.xy) && !nearEdge) {
    discard;
    return vec4f(0.0);
  }
  return marchBeer(in.pos.xy).color;
}
