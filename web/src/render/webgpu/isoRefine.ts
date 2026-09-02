/** Occupancy-guided iso refine: coarse march, display-sized compose, edge-only hi-res rays. */

/** Coarse occl.r below this is a surface hit (misses clear to 1). */
export const ISO_OCC_HIT = 0.999;
/** Normalized-depth jump across a 2×2 of hits that still counts as an edge. */
export const ISO_DEPTH_CREASE = 0.02;

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

/** Mixed occupancy or a depth crease in the four covering coarse occl.r samples. */
export function isoNeedRefineFromCoarse2x2(
  d00: number,
  d10: number,
  d01: number,
  d11: number,
): boolean {
  const ds = [d00, d10, d01, d11];
  let nHit = 0;
  let dMin = 1;
  let dMax = 0;
  for (const d of ds) {
    if (d < ISO_OCC_HIT) {
      nHit++;
      dMin = Math.min(dMin, d);
      dMax = Math.max(dMax, d);
    } else {
      dMax = Math.max(dMax, 1);
    }
  }
  if (nHit > 0 && nHit < 4) return true;
  if (nHit === 4 && dMax - dMin > ISO_DEPTH_CREASE) return true;
  return false;
}
