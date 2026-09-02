/** Occupancy-guided iso refine: 16× coarse occupancy, optional 4× mid, slider-sized compose. */

import { ISO_COARSE_DOWNSCALE } from "../../app/qualityMapping.js";

/** Coarse occl.r below this is a surface hit (misses clear to 1). */
export const ISO_OCC_HIT = 0.999;
/** Normalized-depth jump across a 2×2 of hits that still counts as an edge. */
export const ISO_DEPTH_CREASE = 0.02;

export const ISO_REFINE_NONE = 0;
export const ISO_REFINE_EDGE = 1;
export const ISO_REFINE_INTERSECT = 2;

/**
 * Mid occupancy between coarse and compose. Only used when compose is finer than 4×
 * (surface quality 1–2×), so mixed 16× tiles remarch at 4× instead of at full res.
 */
export const ISO_MID_DOWNSCALE = 4;

function isoFramebufferSizeAtDownscale(
  coarseW: number,
  coarseH: number,
  outW: number,
  outH: number,
  downscale: number,
): { fw: number; fh: number } {
  const cw = Math.max(1, coarseW | 0);
  const ch = Math.max(1, coarseH | 0);
  const ow = Math.max(1, outW | 0);
  const oh = Math.max(1, outH | 0);
  const d = Math.min(ISO_COARSE_DOWNSCALE, Math.max(1, downscale | 0));
  return {
    fw: Math.max(cw, Math.round(ow / d)),
    fh: Math.max(ch, Math.round(oh / d)),
  };
}

/**
 * Fine iso compose size: display / fineDownscale, never coarser than the occupancy pass.
 * Coarse occupancy is 16×; `fineDownscale` is the slider (lowest downsampling).
 */
export function isoFineFramebufferSize(
  coarseW: number,
  coarseH: number,
  outW: number,
  outH: number,
  fineDownscale = 1,
): { fw: number; fh: number } {
  return isoFramebufferSizeAtDownscale(coarseW, coarseH, outW, outH, fineDownscale);
}

export function isoRefineEnabled(
  coarseW: number,
  coarseH: number,
  outW: number,
  outH: number,
  fineDownscale = 1,
): boolean {
  const { fw, fh } = isoFineFramebufferSize(coarseW, coarseH, outW, outH, fineDownscale);
  return fw > (coarseW | 0) || fh > (coarseH | 0);
}

/**
 * 4× mid G-buffer when it sits strictly between coarse occupancy and fine compose.
 * `null` when the slider is already 4× or coarser (two-tier is enough).
 */
export function isoMidFramebufferSize(
  coarseW: number,
  coarseH: number,
  outW: number,
  outH: number,
  fineDownscale = 1,
): { mw: number; mh: number } | null {
  const fine = isoFineFramebufferSize(coarseW, coarseH, outW, outH, fineDownscale);
  const mid = isoFramebufferSizeAtDownscale(coarseW, coarseH, outW, outH, ISO_MID_DOWNSCALE);
  const finerThanCoarse = mid.fw > (coarseW | 0) || mid.fh > (coarseH | 0);
  const coarserThanFine = mid.fw < fine.fw || mid.fh < fine.fh;
  if (!finerThanCoarse || !coarserThanFine) return null;
  return { mw: mid.fw, mh: mid.fh };
}

export function isoMidRefineEnabled(
  coarseW: number,
  coarseH: number,
  outW: number,
  outH: number,
  fineDownscale = 1,
): boolean {
  return isoMidFramebufferSize(coarseW, coarseH, outW, outH, fineDownscale) !== null;
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
  b00 = 0,
  b10 = 0,
  b01 = 0,
  b11 = 0,
): number {
  const ds = [d00, d10, d01, d11];
  const ids = [id00, id10, id01, id11];
  const bounds = [b00, b10, b01, b11];
  let nHit = 0;
  let dMin = 1;
  let dMax = 0;
  let id0 = 0;
  let mixedIso = false;
  let nearBound = false;
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
      if (bounds[i] > 0.5) nearBound = true;
    } else {
      dMax = Math.max(dMax, 1);
    }
  }
  if (mixedIso) return ISO_REFINE_INTERSECT;
  if (nHit > 0 && nHit < 4) return ISO_REFINE_EDGE;
  if (nHit === 4 && dMax - dMin > ISO_DEPTH_CREASE) return ISO_REFINE_EDGE;
  if (nearBound) return ISO_REFINE_EDGE;
  return ISO_REFINE_NONE;
}

/** Mixed occupancy, depth crease, iso intersection, or a hit on a box face. */
export function isoNeedRefineFromCoarse2x2(
  d00: number,
  d10: number,
  d01: number,
  d11: number,
  id00 = 0,
  id10 = 0,
  id01 = 0,
  id11 = 0,
  b00 = 0,
  b10 = 0,
  b01 = 0,
  b11 = 0,
): boolean {
  return isoRefineKindFromCoarse2x2(
    d00, d10, d01, d11, id00, id10, id01, id11, b00, b10, b01, b11,
  ) !== ISO_REFINE_NONE;
}

/** True when a 4×4 coarse neighborhood (2×2 footprint + 1-texel ring) is mixed. */
export function isoDilatedOccupancyNeedsRefine(nHit: number, nTexels = 16): boolean {
  return nHit > 0 && nHit < nTexels;
}
