/** Pure quality → hyperparameter mappings (no DOM / GPU deps). */

function clampQ(q: number): number {
  return Math.min(100, Math.max(0, Math.round(q)));
}

function roundSteps(steps: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.round(steps / 4) * 4));
}

/** Volume (Beer) march resolution divisor from scalar-field quality (1 = full res). */
export function qualityToMarchDownscale(q: number): number {
  const t = clampQ(q) / 100;
  const raw = 8 * Math.pow(1 / 8, t);
  return Math.min(16, Math.max(1, Math.round(raw)));
}

/** Iso-surface march resolution divisor from surface quality (same curve as volume). */
export function qualityToIsoMarchDownscale(q: number): number {
  return qualityToMarchDownscale(q);
}

/** Beer / scalar-volume ray-march step count. */
export function qualityToVolumeSteps(q: number): number {
  const raw = 12 + (clampQ(q) / 100) * 56;
  return roundSteps(raw, 8, 96);
}

/** @deprecated Prefer qualityToVolumeSteps — kept for existing imports/tests. */
export function qualityToSteps(q: number): number {
  return qualityToVolumeSteps(q);
}

/** Iso-surface ray-march step count (Hermite pass allows higher budgets). */
export function qualityToIsoSteps(q: number): number {
  const raw = 16 + (clampQ(q) / 100) * 112;
  return roundSteps(raw, 16, 192);
}

/** Chebyshev degree for field fit (shared fit grid). */
export function qualityToDeg(q: number): number {
  return Math.min(128, Math.max(8, Math.round(12 + (clampQ(q) / 100) * 52)));
}

/** Particles seeded per flow layer. */
export function qualityToParticleCount(q: number): number {
  const raw = 300 + (clampQ(q) / 100) * 7700;
  return Math.min(32000, Math.max(100, Math.round(raw / 100) * 100));
}

/** Trail history length for flow ribbons. */
export function qualityToTrailSteps(q: number): number {
  return Math.min(32, Math.max(2, Math.round(8 + (clampQ(q) / 100) * 24)));
}

function inferFromMap(
  value: number,
  map: (q: number) => number,
): number {
  let bestQ = 50;
  let bestErr = Infinity;
  for (let q = 0; q <= 100; q++) {
    const err = Math.abs(map(q) - value);
    if (err < bestErr) {
      bestErr = err;
      bestQ = q;
    }
  }
  return bestQ;
}

function inferScalarQuality(downscale: number, volumeSteps: number): number {
  const downQ = inferFromMap(downscale, qualityToMarchDownscale);
  const stepsQ = inferFromMap(volumeSteps, qualityToVolumeSteps);
  return Math.round((downQ + stepsQ) / 2);
}

function inferSurfaceQuality(isoDownscale: number, isoSteps: number): number {
  const downQ = inferFromMap(isoDownscale, qualityToIsoMarchDownscale);
  const stepsQ = inferFromMap(isoSteps, qualityToIsoSteps);
  return Math.round((downQ + stepsQ) / 2);
}

function inferPrecisionQuality(deg: number): number {
  return clampQ(((deg - 12) / 52) * 100);
}

function inferVectorQuality(particles: number, trailSteps: number): number {
  const particleQ = clampQ(((particles - 300) / 7700) * 100);
  const trailQ = clampQ(((trailSteps - 8) / 24) * 100);
  return Math.round((particleQ + trailQ) / 2);
}

export function inferQualityFromSettings(input: {
  marchDownscale: number;
  isoMarchDownscale: number;
  deg: number;
  steps: number;
  isoSteps: number;
  flowParticleCount: number;
  flowTrailSteps: number;
}): {
  scalarQuality: number;
  surfaceQuality: number;
  vectorQuality: number;
  precisionQuality: number;
} {
  return {
    scalarQuality: inferScalarQuality(input.marchDownscale, input.steps),
    surfaceQuality: inferSurfaceQuality(input.isoMarchDownscale, input.isoSteps),
    vectorQuality: inferVectorQuality(input.flowParticleCount, input.flowTrailSteps),
    precisionQuality: inferPrecisionQuality(input.deg),
  };
}
