/**
 * Iso texels covered by one volume (beer) pixel. Inclusive [x0, x1] × [y0, y1].
 * Uses ceil on the far edge so a ~2× iso (393 vs 197) still spans two texels;
 * `floor(end) - 1` collapsed that to a single nearest sample.
 */
export function volumePixelIsoFootprint(
  vx: number,
  vy: number,
  volW: number,
  volH: number,
  isoW: number,
  isoH: number,
): { x0: number; y0: number; x1: number; y1: number } {
  const vw = Math.max(1, volW);
  const vh = Math.max(1, volH);
  const iw = Math.max(1, isoW);
  const ih = Math.max(1, isoH);
  const x0 = Math.min(iw - 1, Math.max(0, Math.floor((vx / vw) * iw)));
  const y0 = Math.min(ih - 1, Math.max(0, Math.floor((vy / vh) * ih)));
  const x1 = Math.min(iw - 1, Math.max(x0, Math.ceil(((vx + 1) / vw) * iw) - 1));
  const y1 = Math.min(ih - 1, Math.max(y0, Math.ceil(((vy + 1) / vh) * ih) - 1));
  return { x0, y0, x1, y1 };
}

export const ISO_OCC_HIT = 0.999;

/** True when each volume pixel maps to more than one iso texel on an axis. */
export function isoFinerThanVolume(volW: number, volH: number, isoW: number, isoH: number): boolean {
  return isoW > volW + 0.5 || isoH > volH + 0.5;
}

/**
 * Cheap volume beer clips 16× interiors against coarse iso. Coarse-mixed
 * tiles remarch beer at 4× (same pixels as mid iso) so tExit is 1:1 with
 * that cascade instead of nearest-sampling 4× depth into 2× beer.
 */
export function beerClipCascade(midRefine: boolean): "coarse" | "compose" {
  return midRefine ? "coarse" : "compose";
}

/** True when coarse-mixed tiles get a 4× beer pass before the 1× edge remarch. */
export function beerHasMidPass(midRefine: boolean): boolean {
  return midRefine;
}

/**
 * Iso depth used to shorten beer tExit for a volume pixel's iso footprint.
 * All-miss or mixed → 1 (no clip; mixed tiles remarch at compose).
 * All hits → nearest iso (interior cloud stays in front, behind is cut).
 */
export function beerIsoClipDepth(dMin: number, dMax: number): number {
  if (dMin >= ISO_OCC_HIT || dMax >= ISO_OCC_HIT) return 1;
  return dMin;
}

/** @deprecated Mixed tiles still defer; interior all-hit tiles now clip in beer. */
export function beerDeferredIsoDepth(volW: number, volH: number, isoW: number, isoH: number): boolean {
  return isoFinerThanVolume(volW, volH, isoW, isoH);
}
