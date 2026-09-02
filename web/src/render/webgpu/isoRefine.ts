/** Occupancy-guided iso refine: coarse march, display-sized compose, edge-only hi-res rays. */

/** Coarse occl.r below this is a surface hit (misses clear to 1). */
export const ISO_OCC_HIT = 0.999;
/** Normalized-depth jump across a 2×2 of hits that still counts as an edge. */
export const ISO_DEPTH_CREASE = 0.02;

export const ISO_REFINE_NONE = 0;
export const ISO_REFINE_EDGE = 1;
export const ISO_REFINE_INTERSECT = 2;

/**
 * Fine iso compose size: the display framebuffer.
 * Sharing pixels with the box overlay avoids an FXAA stretch that shifts NDC.
 */
export function isoFineFramebufferSize(
  coarseW: number,
  coarseH: number,
  outW: number,
  outH: number,
): { fw: number; fh: number } {
  const cw = Math.max(1, coarseW | 0);
  const ch = Math.max(1, coarseH | 0);
  const ow = Math.max(1, outW | 0);
  const oh = Math.max(1, outH | 0);
  return {
    fw: Math.max(cw, ow),
    fh: Math.max(ch, oh),
  };
}

export function isoRefineEnabled(
  coarseW: number,
  coarseH: number,
  outW: number,
  outH: number,
): boolean {
  const { fw, fh } = isoFineFramebufferSize(coarseW, coarseH, outW, outH);
  return fw > (coarseW | 0) || fh > (coarseH | 0);
}

/**
 * 0 = keep coarse, 1 = occupancy/depth edge, 2 = two isos meet in the 2×2.
 * `id**` are 1-based layer ids packed in occl.g (0 = miss / unknown).
 */
export function isoRefineKindFromCoarse2x2(
  d00: number,
  d10: number,
  d01: number,
  d11: number,
  id00 = 0,
  id10 = 0,
  id01 = 0,
  id11 = 0,
): number {
  const ds = [d00, d10, d01, d11];
  const ids = [id00, id10, id01, id11];
  let nHit = 0;
  let dMin = 1;
  let dMax = 0;
  let id0 = 0;
  let mixedIso = false;
  for (let i = 0; i < 4; i++) {
    const d = ds[i];
    if (d < ISO_OCC_HIT) {
      nHit++;
      dMin = Math.min(dMin, d);
      dMax = Math.max(dMax, d);
      const id = Math.round(ids[i]);
      if (id >= 1) {
        if (id0 === 0) id0 = id;
        else if (id !== id0) mixedIso = true;
      }
    } else {
      dMax = Math.max(dMax, 1);
    }
  }
  if (mixedIso) return ISO_REFINE_INTERSECT;
  if (nHit > 0 && nHit < 4) return ISO_REFINE_EDGE;
  if (nHit === 4 && dMax - dMin > ISO_DEPTH_CREASE) return ISO_REFINE_EDGE;
  return ISO_REFINE_NONE;
}

/** Mixed occupancy, a depth crease, or two isos in the four covering coarse samples. */
export function isoNeedRefineFromCoarse2x2(
  d00: number,
  d10: number,
  d01: number,
  d11: number,
  id00 = 0,
  id10 = 0,
  id01 = 0,
  id11 = 0,
): boolean {
  return isoRefineKindFromCoarse2x2(d00, d10, d01, d11, id00, id10, id01, id11) !== ISO_REFINE_NONE;
}
