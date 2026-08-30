/** Pure quality → hyperparameter mappings (no DOM / GPU deps). */

function clampQ(q: number): number {
  return Math.min(100, Math.max(0, Math.round(q)));
}

function roundSteps(steps: number): number {
  return Math.min(96, Math.max(8, Math.round(steps / 4) * 4));
}

/** Raymarch resolution divisor from scalar-field quality (1 = full res). */
export function qualityToMarchDownscale(q: number): number {
  const t = clampQ(q) / 100;
  const raw = 8 * Math.pow(1 / 8, t);
  return Math.min(16, Math.max(1, Math.round(raw)));
}

/** Ray-march step count (shared by scalar volumes and iso surfaces). */
export function qualityToSteps(q: number): number {
  const raw = 12 + (clampQ(q) / 100) * 56;
  return roundSteps(raw);
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

function inferScalarQuality(downscale: number): number {
  let bestQ = 50;
  let bestErr = Infinity;
  for (let q = 0; q <= 100; q++) {
    const err = Math.abs(qualityToMarchDownscale(q) - downscale);
    if (err < bestErr) {
      bestErr = err;
      bestQ = q;
    }
  }
  return bestQ;
}

function inferSurfaceQuality(steps: number): number {
  let bestQ = 50;
  let bestErr = Infinity;
  for (let q = 0; q <= 100; q++) {
    const err = Math.abs(qualityToSteps(q) - steps);
    if (err < bestErr) {
      bestErr = err;
      bestQ = q;
    }
  }
  return bestQ;
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
  deg: number;
  steps: number;
  flowParticleCount: number;
  flowTrailSteps: number;
}): {
  scalarQuality: number;
  surfaceQuality: number;
  vectorQuality: number;
  precisionQuality: number;
} {
  return {
    scalarQuality: inferScalarQuality(input.marchDownscale),
    surfaceQuality: inferSurfaceQuality(input.steps),
    vectorQuality: inferVectorQuality(input.flowParticleCount, input.flowTrailSteps),
    precisionQuality: inferPrecisionQuality(input.deg),
  };
}
