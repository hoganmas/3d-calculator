/** Whether a layer must refit this uploadFit pass. */
export function layerNeedsRefit(
  _fromAnim: boolean,
  contentDirty: boolean,
  paramDepends: boolean,
): boolean {
  return contentDirty || paramDepends;
}

export interface AnimFitLayer {
  id?: string;
  latex?: string;
  freeParams?: string[];
  keyframes?: { length: number } | null;
}

/**
 * True when an anim tick still needs Chebyshev compile / CPU keyframe ensure.
 * 1-D GPU blend layers are handled every frame by `tickGpuKeyframeBlends`.
 */
export function animTickNeedsCpuFit(
  layers: readonly AnimFitLayer[],
  dirty: ReadonlySet<string>,
  gpuBlendReady: (id: string) => boolean,
): boolean {
  for (const layer of layers) {
    const free = layer.freeParams;
    const kf = (layer.keyframes?.length ?? 0) > 1 && !!layer.id && gpuBlendReady(layer.id);
    if (!free) {
      if (kf) continue;
      if (dirty.size > 0) return true;
      continue;
    }
    const hit = free.filter((p) => dirty.has(p));
    if (hit.length === 0) continue;
    if (hit.length === 1 && kf) continue;
    return true;
  }
  return false;
}

/**
 * True when a baked field layer is gone or its latex no longer matches.
 * Parameter/alias rows are not bake layers, so extra live expressions are ignored.
 */
export function bakeLatexDrift(
  layers: readonly AnimFitLayer[],
  latexById: ReadonlyMap<string, string>,
): boolean {
  for (const layer of layers) {
    if (!layer.id) continue;
    const latex = latexById.get(layer.id);
    if (latex == null) return true;
    if (layer.latex != null && layer.latex !== latex) return true;
  }
  return false;
}
