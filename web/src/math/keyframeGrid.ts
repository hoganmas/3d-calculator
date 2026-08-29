/** Default samples per param axis; K^N grid (512 cells when N=3 and K=8). */
export const KEYFRAME_GRID_K = 8;

/** Largest standard grid: KEYFRAME_GRID_K^3. */
export const MAX_KEYFRAME_GRID_CELLS = KEYFRAME_GRID_K ** 3;

export interface HypercellCorner {
  index: number;
  weight: number;
}

export interface HypercellBlend {
  coords0: number[];
  coords1: number[];
  ts: number[];
  corners: HypercellCorner[];
}

export function totalFrameCount(K: number, nDims: number): number {
  if (nDims <= 0 || K <= 0) return 0;
  return K ** nDims;
}

/** Row-major: dim 0 slowest, last dim fastest. */
export function coordsFromIndex(idx: number, K: number, nDims: number): number[] {
  const coords = new Array<number>(nDims);
  let x = idx | 0;
  for (let d = nDims - 1; d >= 0; d--) {
    coords[d] = x % K;
    x = Math.floor(x / K);
  }
  return coords;
}

export function linearIndex(coords: readonly number[], K: number): number {
  let idx = 0;
  for (let d = 0; d < coords.length; d++) {
    idx = idx * K + (coords[d] ?? 0);
  }
  return idx;
}

export function paramValuesAtIndex(
  paramNames: readonly string[],
  mins: readonly number[],
  maxs: readonly number[],
  K: number,
  idx: number,
): Record<string, number> {
  const coords = coordsFromIndex(idx, K, paramNames.length);
  const out: Record<string, number> = {};
  for (let d = 0; d < paramNames.length; d++) {
    const min = mins[d] ?? 0;
    const max = maxs[d] ?? 1;
    out[paramNames[d]!] = min + ((max - min) * coords[d]!) / Math.max(1, K - 1);
  }
  return out;
}

export function hypercellBlend(
  mins: readonly number[],
  maxs: readonly number[],
  K: number,
  values: readonly number[],
): HypercellBlend {
  const nDims = mins.length;
  const coords0: number[] = [];
  const coords1: number[] = [];
  const ts: number[] = [];
  for (let d = 0; d < nDims; d++) {
    const span = Math.max(1e-12, (maxs[d] ?? 1) - (mins[d] ?? 0));
    const u = Math.min(1, Math.max(0, ((values[d] ?? 0) - (mins[d] ?? 0)) / span));
    if (K <= 1) {
      coords0.push(0);
      coords1.push(0);
      ts.push(0);
    } else {
      const x = u * (K - 1);
      const i0 = Math.min(K - 2, Math.max(0, Math.floor(x)));
      coords0.push(i0);
      coords1.push(i0 + 1);
      ts.push(x - i0);
    }
  }
  const corners: HypercellCorner[] = [];
  const nCorners = 1 << nDims;
  for (let mask = 0; mask < nCorners; mask++) {
    const coords: number[] = [];
    let weight = 1;
    for (let d = 0; d < nDims; d++) {
      const use1 = (mask >> d) & 1;
      coords.push(use1 ? coords1[d]! : coords0[d]!);
      weight *= use1 ? ts[d]! : 1 - ts[d]!;
    }
    corners.push({ index: linearIndex(coords, K), weight });
  }
  return { coords0, coords1, ts, corners };
}

function manhattanIndexDist(a: number, b: number, K: number, nDims: number): number {
  const ca = coordsFromIndex(a, K, nDims);
  const cb = coordsFromIndex(b, K, nDims);
  let d = 0;
  for (let i = 0; i < nDims; i++) d += Math.abs((ca[i] ?? 0) - (cb[i] ?? 0));
  return d;
}

/** Priority fill outward from hypercell corners through the full K^N grid. */
export function bakeOrderND(K: number, nDims: number, priorityIndices: readonly number[]): number[] {
  const total = totalFrameCount(K, nDims);
  const seen = new Set<number>();
  const order: number[] = [];
  const push = (i: number) => {
    const k = i | 0;
    if (k < 0 || k >= total || seen.has(k)) return;
    seen.add(k);
    order.push(k);
  };
  for (const p of priorityIndices) push(p);
  let dist = 0;
  while (order.length < total && dist <= nDims * K) {
    for (let idx = 0; idx < total; idx++) {
      if (seen.has(idx)) continue;
      for (const p of priorityIndices) {
        if (manhattanIndexDist(idx, p, K, nDims) <= dist) {
          push(idx);
          break;
        }
      }
    }
    dist++;
  }
  for (let idx = 0; idx < total; idx++) push(idx);
  return order;
}

/** 1D segment encoded as hypercell (two corners). */
export function segmentToHypercell(i0: number, i1: number, t: number): HypercellCorner[] {
  return [
    { index: i0, weight: 1 - t },
    { index: i1, weight: t },
  ];
}
