/** Chebyshev-root grid index → world coordinate on [-half, half]. */
export function chebWorld(i: number, M: number, half: number): number {
  const u = Math.cos((Math.PI * (2 * i + 1)) / (2 * M));
  return u * half;
}

export function densIndex(ix: number, iy: number, iz: number, M: number): number {
  return ix + iy * M + iz * M * M;
}
