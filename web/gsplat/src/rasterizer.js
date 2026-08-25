/**
 * Kerbl-style 3DGS software rasterizer (tile + sorted front-to-back)
 * with tunable transmittance early-out and optional Appendix-A T blend.
 */

const TILE = 16;
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const INV_SQRT2 = Math.SQRT1_2;

/** Side-effect sink so probeLog’s −Math.log cannot be DCE’d. */
let probeLogSink = 0;

/** Cap matching α≤0.999 → w=−log(1−0.999). */
const W_MAX = -Math.log(1 - 0.999);

/** LUT for f(x)=−log(1−e^{−x}), x∈[LUT_X0, LUT_X1]. */
const LUT_N = 1024;
const LUT_X0 = 1e-4;
const LUT_X1 = 12;
const LUT_SCALE = (LUT_N - 1) / (LUT_X1 - LUT_X0);
const lutNegLog1mExp = new Float32Array(LUT_N);
for (let i = 0; i < LUT_N; i++) {
  const x = LUT_X0 + (i / (LUT_N - 1)) * (LUT_X1 - LUT_X0);
  lutNegLog1mExp[i] = -Math.log(1 - Math.exp(-x));
}

/** Linear interpolate f(x)=−log(1−e^{−x}). */
function lutNegLog1mExpNeg(x) {
  if (!(x > 0)) return W_MAX;
  if (x <= LUT_X0) {
    const w = -Math.log(x); // f(x)∼−log(x) as x→0
    return w > W_MAX ? W_MAX : w;
  }
  if (x >= LUT_X1) return Math.exp(-x);
  const t = (x - LUT_X0) * LUT_SCALE;
  const i = t | 0;
  const f = t - i;
  return lutNegLog1mExp[i] * (1 - f) + lutNegLog1mExp[i + 1] * f;
}

function opticalDepthExactFromAlpha(alpha) {
  return -Math.log(1 - alpha);
}

/**
 * Exact fused w(r²,o) = −log(1 − min(0.999, o·e^{−½r²})).
 * Still one gauss exp (baseline for “exact” mode).
 */
function fusedOpticalDepthExact(r2, opacity, _negLogO) {
  const alpha = Math.min(0.999, opacity * Math.exp(-0.5 * r2));
  if (alpha < 1e-4) return 0;
  return opticalDepthExactFromAlpha(alpha);
}

/**
 * LUT fused: o·e^{−½r²} = e^{−(½r² − ln o)}, so
 *   w = −log(1 − e^{−x}) with x = ½r² − ln(o) = ½r² + negLogO
 * negLogO = −ln(o) is precomputed per Gaussian at project time.
 */
function fusedOpticalDepthLut(r2, _opacity, negLogO) {
  const x = 0.5 * r2 + negLogO;
  const w = lutNegLog1mExpNeg(x);
  if (w < 1e-4) return 0;
  return w > W_MAX ? W_MAX : w;
}

/**
 * Fast piecewise fused w(r², o) (analytic regimes; see prior notes).
 */
function fusedOpticalDepthFast(r2, opacity, _negLogO) {
  const o = opacity;
  const x2 = 0.5 * r2;
  const x = Math.sqrt(x2);

  if (o >= 0.99) {
    if (x <= 1e-3) return W_MAX;
    if (x <= 0.8) {
      const x4 = x2 * x2;
      const x6 = x4 * x2;
      const w =
        -2 * Math.log(x) + 0.5 * x2 + x4 * (1 / 24) + x6 * (1 / 180);
      return w > W_MAX ? W_MAX : w;
    }
    if (x >= 1.5) {
      const e1 = Math.exp(-x2);
      return e1 + 0.5 * e1 * e1;
    }
    const alpha = Math.min(0.999, o * Math.exp(-x2));
    if (alpha < 1e-4) return 0;
    return opticalDepthExactFromAlpha(alpha);
  }

  if (x >= 1.5) {
    const u = o * Math.exp(-x2);
    if (u < 1e-4) return u;
    return u + u * u * (0.5 + u * (1 / 3));
  }
  if (x <= 0.8) {
    const eApprox = 1 - x2 + x2 * x2 * (0.5 - x2 * (1 / 6));
    const u = Math.min(0.999, o * Math.max(0, eApprox));
    if (u < 1e-4) return u;
    if (u < 0.55) {
      return u + u * u * (0.5 + u * (1 / 3 + u * (0.25 + u * 0.2)));
    }
    return opticalDepthExactFromAlpha(u);
  }
  const alpha = Math.min(0.999, o * Math.exp(-x2));
  if (alpha < 1e-4) return 0;
  if (alpha < 0.55) {
    const u = alpha;
    return u + u * u * (0.5 + u * (1 / 3 + u * (0.25 + u * 0.2)));
  }
  return opticalDepthExactFromAlpha(alpha);
}

function selectFusedOpticalDepth(mode) {
  if (mode === "lut") return fusedOpticalDepthLut;
  if (mode === "fast") return fusedOpticalDepthFast;
  return fusedOpticalDepthExact;
}

function mat4LookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ];
}

function mul4(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

/** Affine / homogeneous transform (no divide). */
function transformPoint4(m, p) {
  const x = p[0], y = p[1], z = p[2];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ];
}

function transformPoint(m, p) {
  const c = transformPoint4(m, p);
  const nw = c[3] || 1e-12;
  return [c[0] / nw, c[1] / nw, c[2] / nw];
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

/** Abramowitz & Stegun erf → standard-normal CDF (~1.5e-7). */
function normCdfAS(x) {
  const z = x * INV_SQRT2;
  const sign = z < 0 ? -1 : 1;
  const ax = Math.abs(z);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-ax * ax));
  return 0.5 * (1 + sign * y);
}

/**
 * Logistic approx to Φ: 1/(1+exp(-λx)), λ≈1.702 (Bowling et al. / Tocher).
 * ~1e-2 max abs error; one exp, no poly — faster hot path.
 */
const LOGISTIC_LAMBDA = 1.702;
function normCdfLogistic(x) {
  if (x >= 8) return 1;
  if (x <= -8) return 0;
  return 1 / (1 + Math.exp(-LOGISTIC_LAMBDA * x));
}

/** LUT for Φ(x) on [−8, 8], linear interp (built from A&S). */
const CDF_LUT_N = 1024;
const CDF_LUT_X0 = -8;
const CDF_LUT_X1 = 8;
const CDF_LUT_SCALE = (CDF_LUT_N - 1) / (CDF_LUT_X1 - CDF_LUT_X0);
const lutNormCdf = new Float32Array(CDF_LUT_N);
for (let i = 0; i < CDF_LUT_N; i++) {
  const x = CDF_LUT_X0 + (i / (CDF_LUT_N - 1)) * (CDF_LUT_X1 - CDF_LUT_X0);
  lutNormCdf[i] = normCdfAS(x);
}
function normCdfLut(x) {
  if (x <= CDF_LUT_X0) return 0;
  if (x >= CDF_LUT_X1) return 1;
  const t = (x - CDF_LUT_X0) * CDF_LUT_SCALE;
  const i = t | 0;
  const f = t - i;
  return lutNormCdf[i] * (1 - f) + lutNormCdf[i + 1] * f;
}

function selectNormCdf(mode) {
  if (mode === "lut") return normCdfLut;
  if (mode === "logistic") return normCdfLogistic;
  return normCdfAS;
}

/** α = 1−e^{−w}; cheap for small w (skips expm1). */
function alphaFromW(w) {
  if (w < 1e-4) return w;
  if (w < 0.25) {
    // 1−e^{−w} ≈ w − w²/2 + w³/6
    return w * (1 - w * (0.5 - w * (1 / 6)));
  }
  const a = -Math.expm1(-w);
  return a > 0.999 ? 0.999 : a;
}

/** LUT for e^{−x}, x∈[0, EXP_X1] — Beer / Φ transmittance. */
const EXP_LUT_N = 1024;
const EXP_X1 = 16; // e^{−16} ≪ ε
const EXP_LUT_SCALE = (EXP_LUT_N - 1) / EXP_X1;
const lutExpNeg = new Float32Array(EXP_LUT_N);
for (let i = 0; i < EXP_LUT_N; i++) {
  lutExpNeg[i] = Math.exp(-(i / (EXP_LUT_N - 1)) * EXP_X1);
}
function expNegLut(x) {
  if (!(x > 0)) return 1;
  if (x >= EXP_X1) return 0;
  const t = x * EXP_LUT_SCALE;
  const i = t | 0;
  const f = t - i;
  return lutExpNeg[i] * (1 - f) + lutExpNeg[i + 1] * f;
}

function normPdf(x) {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}

/**
 * Appendix A scales: δ/σ and ν/σ are functions of w only.
 * LUT over w∈[0,W_MAX] so appendix setup is two loads (no pdf/cdf/sqrt).
 */
const APP_LUT_N = 512;
const APP_LUT_SCALE = (APP_LUT_N - 1) / W_MAX;
const lutAppD = new Float32Array(APP_LUT_N); // δ/σ
const lutAppN = new Float32Array(APP_LUT_N); // ν/σ
(function buildAppendixScaleLut() {
  for (let i = 0; i < APP_LUT_N; i++) {
    const w = (i / (APP_LUT_N - 1)) * W_MAX;
    if (w < 1e-8) {
      lutAppD[i] = 0;
      lutAppN[i] = 1;
      continue;
    }
    const d =
      (1 / w) * (-SQRT_2PI + Math.sqrt(2 * Math.PI + 2 * w * w));
    const phi = Math.max(normPdf(d), 1e-12);
    const n =
      (Math.abs(Math.expm1(w)) * Math.exp(-w * normCdfAS(d))) /
      (w * phi * SQRT_2PI);
    lutAppD[i] = d;
    lutAppN[i] = Math.max(n, 1e-8);
  }
})();

/** Appendix A: exp(w Φ) ≈ 1+(e^w−1) Φ((z−μ−δ)/ν). Writes into hit. */
function prepareAppendixHit(h, cdf, zf, prof) {
  const w = h.w;
  let d;
  let nScale;
  if (!(w > 0)) {
    d = 0;
    nScale = 1;
  } else if (w >= W_MAX) {
    d = lutAppD[APP_LUT_N - 1];
    nScale = lutAppN[APP_LUT_N - 1];
  } else {
    const t = w * APP_LUT_SCALE;
    const i = t | 0;
    const f = t - i;
    d = lutAppD[i] * (1 - f) + lutAppD[i + 1] * f;
    nScale = lutAppN[i] * (1 - f) + lutAppN[i + 1] * f;
  }
  const sig = h.sigma;
  h.invNu = 1 / Math.max(sig * nScale, 1e-8);
  h.expm1w = Math.expm1(w);
  h.muDelta = h.mu + sig * d;
  const cdfZf = cdf((zf - h.mu) * h.invSigma);
  h.cdfZf = cdfZf;
  h.survFar = expNegLut(w * cdfZf); // exp(−w Φ(zf)), reused every T eval
  h.appendixReady = true;
  if (prof) {
    prof.appendixCalls++;
    prof.cdfCalls++;
  }
}

function clampAppendixT(t) {
  return t < 0 ? 0 : t > 1.5 ? 1.5 : t;
}

function noteAppendixSlice(prof, n) {
  if (!prof) return;
  prof.tSliceCalls++;
  prof.tSliceIters += n;
  prof.cdfCalls += n;
}

/** Single-front Appendix-A (maxΦ=1 / winLen=1). */
function T_appendixOne(h, z, zf, cdf, prof) {
  if (!h.appendixReady) prepareAppendixHit(h, cdf, zf, prof);
  noteAppendixSlice(prof, 1);
  return h.survFar * (1 + h.expm1w * cdf((z - h.muDelta) * h.invNu));
}

/** Two-front unrolled Appendix-A. hits[hi-1] nearest-to-z, hits[hi-2] older. */
function T_appendixTwo(hits, hi, z, zf, cdf, prof) {
  const h0 = hits[hi - 1];
  const h1 = hits[hi - 2];
  if (!h0.appendixReady) prepareAppendixHit(h0, cdf, zf, prof);
  if (!h1.appendixReady) prepareAppendixHit(h1, cdf, zf, prof);
  const c0 = cdf((z - h0.muDelta) * h0.invNu);
  const c1 = cdf((z - h1.muDelta) * h1.invNu);
  const sum = 1 + h0.expm1w * c0 + h1.expm1w * h0.eW * c1;
  noteAppendixSlice(prof, 2);
  return h0.survFar * h1.survFar * sum;
}

function T_appendixThree(hits, hi, z, zf, cdf, prof) {
  const h0 = hits[hi - 1];
  const h1 = hits[hi - 2];
  const h2 = hits[hi - 3];
  if (!h0.appendixReady) prepareAppendixHit(h0, cdf, zf, prof);
  if (!h1.appendixReady) prepareAppendixHit(h1, cdf, zf, prof);
  if (!h2.appendixReady) prepareAppendixHit(h2, cdf, zf, prof);
  const c0 = cdf((z - h0.muDelta) * h0.invNu);
  const c1 = cdf((z - h1.muDelta) * h1.invNu);
  const c2 = cdf((z - h2.muDelta) * h2.invNu);
  const e0 = h0.eW;
  const e01 = e0 * h1.eW;
  const sum =
    1 + h0.expm1w * c0 + h1.expm1w * e0 * c1 + h2.expm1w * e01 * c2;
  noteAppendixSlice(prof, 3);
  return h0.survFar * h1.survFar * h2.survFar * sum;
}

function T_appendixFour(hits, hi, z, zf, cdf, prof) {
  const h0 = hits[hi - 1];
  const h1 = hits[hi - 2];
  const h2 = hits[hi - 3];
  const h3 = hits[hi - 4];
  if (!h0.appendixReady) prepareAppendixHit(h0, cdf, zf, prof);
  if (!h1.appendixReady) prepareAppendixHit(h1, cdf, zf, prof);
  if (!h2.appendixReady) prepareAppendixHit(h2, cdf, zf, prof);
  if (!h3.appendixReady) prepareAppendixHit(h3, cdf, zf, prof);
  const c0 = cdf((z - h0.muDelta) * h0.invNu);
  const c1 = cdf((z - h1.muDelta) * h1.invNu);
  const c2 = cdf((z - h2.muDelta) * h2.invNu);
  const c3 = cdf((z - h3.muDelta) * h3.invNu);
  const e0 = h0.eW;
  const e01 = e0 * h1.eW;
  const e012 = e01 * h2.eW;
  const sum =
    1 +
    h0.expm1w * c0 +
    h1.expm1w * e0 * c1 +
    h2.expm1w * e01 * c2 +
    h3.expm1w * e012 * c3;
  noteAppendixSlice(prof, 4);
  return h0.survFar * h1.survFar * h2.survFar * h3.survFar * sum;
}

/**
 * Appendix-A T(z) for hits[lo..hi) — generic path (winLen > 4 / unlimited).
 * Product order: μ-ascending (nearest-to-z first).
 */
function T_appendixRange(hits, lo, hi, z, zf, cdf, prof) {
  if (hi <= lo) return 1;
  let farT = 1;
  let ePrefix = 1;
  let sum = 1;
  let n = 0;
  for (let t = hi - 1; t >= lo; t--) {
    const h = hits[t];
    if (!h.appendixReady) prepareAppendixHit(h, cdf, zf, prof);
    sum += h.expm1w * ePrefix * cdf((z - h.muDelta) * h.invNu);
    ePrefix *= h.eW;
    farT *= h.survFar;
    n++;
  }
  noteAppendixSlice(prof, n);
  return farT * sum;
}

/**
 * Perspective-correct projection: screen radius ∝ focal * worldScale / distance.
 * Uses Euclidean distance to the camera eye (unambiguous) and projects a
 * camera-right offset so size tracks perspective zoom.
 */
function projectGaussian(mean, scaleVec, eye, view, viewProj, width, height, focal, near) {
  const dx = mean[0] - eye[0];
  const dy = mean[1] - eye[1];
  const dz = mean[2] - eye[2];
  const dist = Math.hypot(dx, dy, dz);
  if (dist < near) return null;

  const viewP = transformPoint4(view, mean);
  const viewDepth = -viewP[2];
  if (viewDepth < near) return null;

  const clip = transformPoint(viewProj, mean);
  if (clip[2] < -1.2 || clip[2] > 1.2) return null;

  const sx = (clip[0] * 0.5 + 0.5) * width;
  const sy = (1 - (clip[1] * 0.5 + 0.5)) * height;
  const s = (scaleVec[0] + scaleVec[1] + scaleVec[2]) / 3;

  // Camera +X axis in world (row 0 of rotation / col basis of view)
  const camRight = [view[0], view[4], view[8]];
  const tip = [
    mean[0] + camRight[0] * s,
    mean[1] + camRight[1] * s,
    mean[2] + camRight[2] * s,
  ];
  const tipClip = transformPoint(viewProj, tip);
  const tipSx = (tipClip[0] * 0.5 + 0.5) * width;
  const tipSy = (1 - (tipClip[1] * 0.5 + 0.5)) * height;
  // Fallback to focal/dist if tip projection is degenerate
  let radius = Math.hypot(tipSx - sx, tipSy - sy);
  if (!(radius > 1e-3)) {
    radius = (s * focal) / dist;
  }
  radius = Math.max(0.5, radius);

  return { sx, sy, radius, depth: viewDepth, dist, sigma: Math.max(s, 1e-4) };
}

/**
 * @param {object} scene
 * @param {object} camera { eye, target, up, fovy, width, height }
 * @param {object} opts   { eps, background, maxPerPixel, blendMode, profile, maxInteract, kSigma, cdfMode, logMode }
 *   blendMode: "alpha" | "approx" | "probe" | "probeLog" | "probeWin" | "probeBeer"
 *   maxInteract: Φ-front cap per emitter (0 = unlimited)
 *   cdfMode: "lut" | "logistic" | "as"
 *   logMode: "exact" | "fast" | "lut"
 */
export function renderFrame(scene, camera, opts = {}) {
  const width = camera.width | 0;
  const height = camera.height | 0;
  const eps = opts.eps ?? 1e-4;
  const bg = opts.background ?? [0.02, 0.03, 0.05];
  const blendMode =
    opts.blendMode === "approx"
      ? "approx"
      : opts.blendMode === "probe" ||
          opts.blendMode === "probeLog" ||
          opts.blendMode === "probeWin" ||
          opts.blendMode === "probeBeer"
        ? opts.blendMode
        : "alpha";
  const maxPerPixel = opts.maxPerPixel ?? 64;
  const kSigma = opts.kSigma ?? 2.5;
  const maxInteract = opts.maxInteract ?? 0;
  const cdfMode =
    opts.cdfMode === "as"
      ? "as"
      : opts.cdfMode === "logistic"
        ? "logistic"
        : "lut";
  const cdf = selectNormCdf(cdfMode);
  const logMode =
    opts.logMode === "exact"
      ? "exact"
      : opts.logMode === "fast"
        ? "fast"
        : "lut";
  const fusedOpticalDepth = selectFusedOpticalDepth(logMode);
  const detailProf = opts.profile === true; // hot-loop counters (Profile button)
  const probeBeer = blendMode === "probeBeer";
  const probeAlphaT =
    blendMode === "probe" || blendMode === "probeWin";
  const probeLogOnly = blendMode === "probeLog";
  const useWindowLoop =
    blendMode === "approx" ||
    blendMode === "probe" ||
    blendMode === "probeWin" ||
    probeBeer;
  const doOpticalLog =
    blendMode === "approx" || blendMode === "probe" || probeBeer;
  const hardCapWindow = blendMode === "approx" || probeBeer;
  const near = 0.05;
  const far = 200;
  const fovy = camera.fovy;

  const prof = {
    projectMs: 0,
    sortMs: 0,
    tileMs: 0,
    clearMs: 0,
    gatherMs: 0,
    compositeMs: 0,
    rasterMs: 0,
    appendixCalls: 0,
    tSliceCalls: 0,
    tSliceIters: 0,
    cdfCalls: 0,
    pixelsComposited: 0,
    hitSum: 0,
    hitMax: 0,
    listTests: 0,
    interactPairs: 0,
    beerPairs: 0,
    cappedPairs: 0,
    frontScans: 0,
    interactMax: 0,
    maxInteract,
    kSigma,
    cdfMode,
    logMode,
  };

  let tMark = performance.now();
  const view = mat4LookAt(camera.eye, camera.target, camera.up);
  const proj = mat4Perspective(fovy, width / height, near, far);
  const viewProj = mul4(proj, view);
  const focal = height / (2 * Math.tan(fovy * 0.5));

  const n = scene.count;
  const projected = [];
  let radiusMin = Infinity;
  let radiusMax = 0;
  let radiusSum = 0;
  for (let i = 0; i < n; i++) {
    const p = projectGaussian(
      scene.means[i],
      scene.scales[i],
      camera.eye,
      view,
      viewProj,
      width,
      height,
      focal,
      near,
    );
    if (!p) continue;
    radiusMin = Math.min(radiusMin, p.radius);
    radiusMax = Math.max(radiusMax, p.radius);
    radiusSum += p.radius;
    projected.push({
      i,
      sx: p.sx,
      sy: p.sy,
      radius: p.radius,
      depth: p.depth,
      sigma: p.sigma,
      color: scene.colors[i],
      opacity: scene.opacities[i],
      // −ln(o) once per Gaussian for LUT fused w (x = ½r² + negLogO)
      negLogO: -Math.log(Math.max(scene.opacities[i], 1e-8)),
    });
  }
  if (true) {
    prof.projectMs = performance.now() - tMark;
    tMark = performance.now();
  }

  projected.sort((a, b) => a.depth - b.depth);
  if (true) {
    prof.sortMs = performance.now() - tMark;
    tMark = performance.now();
  }

  const tilesX = Math.ceil(width / TILE);
  const tilesY = Math.ceil(height / TILE);
  const tileLists = Array.from({ length: tilesX * tilesY }, () => []);

  for (const g of projected) {
    const r = g.radius * 3.0;
    const x0 = Math.max(0, Math.floor((g.sx - r) / TILE));
    const x1 = Math.min(tilesX - 1, Math.floor((g.sx + r) / TILE));
    const y0 = Math.max(0, Math.floor((g.sy - r) / TILE));
    const y1 = Math.min(tilesY - 1, Math.floor((g.sy + r) / TILE));
    for (let ty = y0; ty <= y1; ty++) {
      for (let tx = x0; tx <= x1; tx++) {
        tileLists[ty * tilesX + tx].push(g);
      }
    }
  }
  if (true) {
    prof.tileMs = performance.now() - tMark;
    tMark = performance.now();
  }

  const rgba = new Uint8ClampedArray(width * height * 4);
  const bgR = Math.round(bg[0] * 255);
  const bgG = Math.round(bg[1] * 255);
  const bgB = Math.round(bg[2] * 255);
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    rgba[idx] = bgR;
    rgba[idx + 1] = bgG;
    rgba[idx + 2] = bgB;
    rgba[idx + 3] = 255;
  }
  if (true) {
    prof.clearMs = performance.now() - tMark;
    tMark = performance.now();
  }

  let splatEvals = 0;
  let earlyOutPixels = 0;
  let pixelsFilled = 0;
  // Sliding-window hit pool (near→far). winLo..nHits = active Φ candidates.
  const hitPool = Array.from({ length: maxPerPixel }, () => ({
    mu: 0,
    sigma: 0,
    w: 0,
    surv: 1,
    eW: 1,
    expm1w: 0,
    survFar: 1,
    color: null,
    appendixReady: false,
    muDelta: 0,
    invNu: 0,
    invSigma: 0,
    cdfZf: -1,
  }));
  const approxProf =
    detailProf && blendMode === "approx" ? prof : null;
  const probeProf =
    detailProf && (probeAlphaT || probeLogOnly || probeBeer) ? prof : null;
  const zf = 0;
  const phiCap = maxInteract > 0 ? maxInteract | 0 : 0; // 0 = unlimited
  if (probeAlphaT || probeLogOnly || probeBeer) {
    prof.probeAlphaT = probeAlphaT;
    prof.probeKind = blendMode;
    if (probeBeer) prof.probeBeer = true;
  }

  const t0 = performance.now();

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const list = tileLists[ty * tilesX + tx];
      if (!list.length) continue;

      const xStart = tx * TILE;
      const yStart = ty * TILE;
      const xEnd = Math.min(width, xStart + TILE);
      const yEnd = Math.min(height, yStart + TILE);

      for (let y = yStart; y < yEnd; y++) {
        for (let x = xStart; x < xEnd; x++) {
          let r = 0, gch = 0, b = 0;
          let exited = false;
          let hits = 0;

          if (useWindowLoop) {
            // Fused near→far sliding window (approx + probe / probeWin).
            // beerT = ∏ surv over Beer'd fronts; winT = ∏ surv over live window.
            // Both stay in α-space (multiplies only) — no expNeg(Σw) per hit.
            let nHits = 0;
            let winLo = 0;
            let beerT = 1;
            let winT = 1;
            let lastT = 1;
            let Talpha = 1;

            for (let k = 0; k < list.length; k++) {
              if (detailProf) prof.listTests++;
              const s = list[k];
              const dx = x + 0.5 - s.sx;
              const dy = y + 0.5 - s.sy;
              const invR = 1 / s.radius;
              const u = dx * invR;
              const v = dy * invR;
              const r2 = u * u + v * v;
              if (r2 > 9.0) continue;

              const curMu = -s.depth;
              const curSigma = s.sigma;
              let alpha;
              let w;

              if (doOpticalLog) {
                // Fused w(r²,o) — no gauss exp→α→−log middle (approx + probe).
                w = fusedOpticalDepth(r2, s.opacity, s.negLogO);
                if (w < 1e-4) continue;
                alpha = alphaFromW(w);
              } else {
                // probeWin: classic α; dummy w for window bookkeeping
                const gauss = Math.exp(-0.5 * r2);
                alpha = Math.min(0.999, s.opacity * gauss);
                if (alpha < 1e-4) continue;
                w = alpha;
              }

              // Evict non-overlapping fronts → Beer.
              while (winLo < nHits) {
                const old = hitPool[winLo];
                if (approxProf || probeProf) prof.frontScans++;
                if (old.mu - curMu <= kSigma * (old.sigma + curSigma)) break;
                beerT *= old.surv;
                winT *= old.eW; // drop from live ∏surv
                winLo++;
                if (approxProf || probeProf) prof.beerPairs++;
              }

              // Hard-cap window to maxΦ (oldest → Beer). Approx + probeBeer.
              if (hardCapWindow && phiCap > 0) {
                while (nHits - winLo > phiCap) {
                  const old = hitPool[winLo++];
                  beerT *= old.surv;
                  winT *= old.eW;
                  if (approxProf || probeProf) prof.cappedPairs++;
                }
              }
              if (winLo === nHits) winT = 1; // kill float drift when empty

              const winLen = nHits - winLo;
              let T;

              if (probeAlphaT) {
                T = Talpha;
                if (probeProf) {
                  probeProf.interactPairs += winLen > 0 ? 1 : 0;
                  probeProf.cappedPairs += Math.max(0, winLen - 1);
                  if (winLen > 0 && probeProf.interactMax < 1)
                    probeProf.interactMax = 1;
                }
              } else if (probeBeer) {
                // Same window as approx, but full Beer (no Appendix-A).
                T = beerT * winT;
                if (probeProf) {
                  if (winLen > probeProf.interactMax)
                    probeProf.interactMax = winLen;
                }
              } else if (winLen === 0) {
                T = beerT;
              } else {
                // Appendix-A on hard-capped window; unrolled for N≤4.
                let Tl;
                if (winLen === 1) {
                  Tl = T_appendixOne(
                    hitPool[nHits - 1],
                    curMu,
                    zf,
                    cdf,
                    approxProf,
                  );
                } else if (winLen === 2) {
                  Tl = T_appendixTwo(
                    hitPool,
                    nHits,
                    curMu,
                    zf,
                    cdf,
                    approxProf,
                  );
                } else if (winLen === 3) {
                  Tl = T_appendixThree(
                    hitPool,
                    nHits,
                    curMu,
                    zf,
                    cdf,
                    approxProf,
                  );
                } else if (winLen === 4) {
                  Tl = T_appendixFour(
                    hitPool,
                    nHits,
                    curMu,
                    zf,
                    cdf,
                    approxProf,
                  );
                } else {
                  Tl = T_appendixRange(
                    hitPool,
                    winLo,
                    nHits,
                    curMu,
                    zf,
                    cdf,
                    approxProf,
                  );
                }
                T = beerT * clampAppendixT(Tl);
                if (approxProf) {
                  approxProf.interactPairs += winLen;
                  if (winLen > approxProf.interactMax)
                    approxProf.interactMax = winLen;
                }
              }
              lastT = T;

              if (probeAlphaT) {
                r += s.color[0] * alpha * T;
                gch += s.color[1] * alpha * T;
                b += s.color[2] * alpha * T;
                Talpha *= 1 - alpha;
                lastT = Talpha;
              } else {
                r += s.color[0] * alpha * T;
                gch += s.color[1] * alpha * T;
                b += s.color[2] * alpha * T;
              }

              // Append (Appendix-A prep is lazy on first Φ use).
              const h = hitPool[nHits++];
              h.mu = curMu;
              h.sigma = curSigma;
              h.w = w;
              h.surv = 1 - alpha;
              h.eW = 1 / Math.max(h.surv, 1e-6);
              h.color = s.color;
              h.invSigma = 1 / curSigma;
              h.cdfZf = -1;
              h.appendixReady = false;
              winT *= h.surv;
              hits++;
              splatEvals++;

              const Texit = probeAlphaT ? Talpha : T;
              if (Texit < eps || nHits >= maxPerPixel) {
                exited = true;
                // Include last splat’s Beer mass in exit T (probeBeer / bg).
                if (probeBeer) lastT = beerT * winT;
                break;
              }
            }

            const Tbg = probeAlphaT
              ? exited
                ? lastT
                : Talpha
              : exited
                ? lastT
                : beerT * winT;
            r += bg[0] * Tbg;
            gch += bg[1] * Tbg;
            b += bg[2] * Tbg;

            if (hits > 0) {
              prof.pixelsComposited++;
              prof.hitSum += hits;
              if (hits > prof.hitMax) prof.hitMax = hits;
            }
          } else {
            // Classic α path (+ optional dead −log for probeLog ablation)
            let T = 1.0;
            for (let k = 0; k < list.length; k++) {
              if (detailProf) prof.listTests++;
              const s = list[k];
              const dx = x + 0.5 - s.sx;
              const dy = y + 0.5 - s.sy;
              const invR = 1 / s.radius;
              const u = dx * invR;
              const v = dy * invR;
              const r2 = u * u + v * v;
              if (r2 > 9.0) continue;

              let alpha;
              if (probeLogOnly) {
                // Fused w(r²,o) → α (no separate gauss exp) — fair fused ablation
                const w = fusedOpticalDepth(r2, s.opacity, s.negLogO);
                if (w < 1e-4) continue;
                alpha = alphaFromW(w);
                probeLogSink = w;
              } else {
                const gauss = Math.exp(-0.5 * r2);
                alpha = Math.min(0.99, s.opacity * gauss);
                if (alpha < 1e-4) continue;
              }

              splatEvals++;
              hits++;
              const weight = alpha * T;
              r += s.color[0] * weight;
              gch += s.color[1] * weight;
              b += s.color[2] * weight;
              T *= 1.0 - alpha;

              if (T < eps) {
                exited = true;
                break;
              }
              if (hits >= maxPerPixel) {
                exited = true;
                break;
              }
            }
            r += bg[0] * T;
            gch += bg[1] * T;
            b += bg[2] * T;

            if (probeLogOnly && hits > 0) {
              prof.pixelsComposited++;
              prof.hitSum += hits;
              if (hits > prof.hitMax) prof.hitMax = hits;
            }
          }

          if (exited) earlyOutPixels++;
          if (hits > 0) pixelsFilled++;

          const idx = (y * width + x) * 4;
          rgba[idx] = Math.min(255, Math.round(r * 255));
          rgba[idx + 1] = Math.min(255, Math.round(gch * 255));
          rgba[idx + 2] = Math.min(255, Math.round(b * 255));
          rgba[idx + 3] = 255;
        }
      }
    }
  }

  const ms = performance.now() - t0;
  if (true) {
    prof.rasterMs = ms;
    // Fused gather+composite — no separate phase split.
    prof.gatherMs = 0;
    prof.compositeMs = useWindowLoop ? ms : 0;
    prof.phaseTimed = false;
    prof.fused = useWindowLoop;
  }

  const hitMean = prof.pixelsComposited ? prof.hitSum / prof.pixelsComposited : 0;
  // Expected T_slice iters ≈ Σ n(n+1)/2 over pixels (upper bound without early-out)
  const expectedQuadratic = prof.pixelsComposited
    ? 0.5 * prof.hitSum * (hitMean + 1)
    : 0;

  return {
    rgba,
    width,
    height,
    stats: {
      ms: prof.projectMs + prof.sortMs + prof.tileMs + prof.clearMs + prof.rasterMs,
      rasterMs: ms,
      projected: projected.length,
      splatEvals,
      earlyOutPixels,
      pixelsFilled,
      eps,
      blendMode,
      maxInteract,
      kSigma,
      cdfMode,
      logMode,
      evalsPerPixel: pixelsFilled ? splatEvals / pixelsFilled : 0,
      radiusPx: projected.length
        ? {
            min: radiusMin,
            max: radiusMax,
            mean: radiusSum / projected.length,
          }
        : null,
      profile: {
            ...prof,
            hitMean,
            expectedQuadratic,
            detail: detailProf,
            msPerMIter: prof.tSliceIters
              ? (prof.compositeMs / prof.tSliceIters) * 1e6
              : 0,
          },
    },
  };
}

export function orbitEye(yaw, pitch, radius, target = [0, 0, 0]) {
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw);
  const cp = Math.cos(pitch),
    sp = Math.sin(pitch);
  return [
    target[0] + radius * cp * sy,
    target[1] + radius * sp,
    target[2] + radius * cp * cy,
  ];
}
