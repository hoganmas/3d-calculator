/** CPU mirror of `sampleGradStops` in `shaders/common/gradient.wgsl`. */

export type Rgb = [number, number, number];

/** Piecewise-linear gradient sample (correct `f = x - i` at t=1). */
export function sampleGradStops(stops: Rgb[], t: number): Rgb {
  const n = Math.max(Math.min(stops.length, 6), 1);
  if (n <= 1) return stops[0] ?? [0, 0, 0];
  const x = Math.min(Math.max(t, 0), 1) * (n - 1);
  const i = Math.min(Math.floor(x), n - 2);
  const f = x - i;
  const a = stops[i]!;
  const b = stops[i + 1]!;
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

/** Pre-fix WGSL used `fract(x)` — at t=1 with two stops both endpoints returned stop[0]. */
export function sampleGradStopsFractBug(stops: Rgb[], t: number): Rgb {
  const n = Math.max(Math.min(stops.length, 6), 1);
  if (n <= 1) return stops[0] ?? [0, 0, 0];
  const x = Math.min(Math.max(t, 0), 1) * (n - 1);
  const i = Math.min(Math.floor(x), n - 2);
  const f = x - Math.floor(x);
  const a = stops[i]!;
  const b = stops[i + 1]!;
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

/** Flow shader samples endpoints at t=0 and t=1 for speed coloring. */
export function flowSpeedColorEndpoints(
  stops: Rgb[],
  speedNorm: number,
  sample: (stops: Rgb[], t: number) => Rgb = sampleGradStops,
): Rgb {
  const col1 = sample(stops, 0);
  const col2 = sample(stops, 1);
  const u = Math.min(Math.max(speedNorm, 0), 1);
  return [
    col1[0] + (col2[0] - col1[0]) * u,
    col1[1] + (col2[1] - col1[1]) * u,
    col1[2] + (col2[2] - col1[2]) * u,
  ];
}
