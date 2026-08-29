/**
 * Trilinear resample a cubic volume to target resolution.
 * Concurrent keyframe ladders leave clouds at different native Ms; packing a short
 * dens into a larger scene grid (zero-pad / prefix copy) reads as tearing in beer-lambert.
 */
export function resampleVolumeGrid(src: Float32Array, dstM: number): Float32Array {
  const srcM = Math.round(Math.cbrt(src.length));
  const dstN = dstM * dstM * dstM;
  if (srcM <= 0 || dstM <= 0) return new Float32Array(dstN);
  if (srcM === dstM && src.length >= dstN) {
    return src.length === dstN ? src : src.subarray(0, dstN);
  }
  const dst = new Float32Array(dstN);
  const scale = srcM === 1 || dstM === 1 ? 0 : (srcM - 1) / (dstM - 1);
  const at = (x: number, y: number, z: number) => src[x + srcM * (y + srcM * z)]!;
  for (let z = 0; z < dstM; z++) {
    const zf = z * scale;
    const z0 = Math.min(srcM - 1, Math.floor(zf));
    const z1 = Math.min(srcM - 1, z0 + 1);
    const tz = zf - z0;
    for (let y = 0; y < dstM; y++) {
      const yf = y * scale;
      const y0 = Math.min(srcM - 1, Math.floor(yf));
      const y1 = Math.min(srcM - 1, y0 + 1);
      const ty = yf - y0;
      for (let x = 0; x < dstM; x++) {
        const xf = x * scale;
        const x0 = Math.min(srcM - 1, Math.floor(xf));
        const x1 = Math.min(srcM - 1, x0 + 1);
        const tx = xf - x0;
        const c00 = at(x0, y0, z0) * (1 - tx) + at(x1, y0, z0) * tx;
        const c01 = at(x0, y0, z1) * (1 - tx) + at(x1, y0, z1) * tx;
        const c10 = at(x0, y1, z0) * (1 - tx) + at(x1, y1, z0) * tx;
        const c11 = at(x0, y1, z1) * (1 - tx) + at(x1, y1, z1) * tx;
        const c0 = c00 * (1 - ty) + c10 * ty;
        const c1 = c01 * (1 - ty) + c11 * ty;
        dst[x + dstM * (y + dstM * z)] = c0 * (1 - tz) + c1 * tz;
      }
    }
  }
  return dst;
}
