/**
 * Kerbl-style 3DGS software rasterizer (tile + sorted front-to-back)
 * with tunable transmittance early-out and optional Appendix-A T blend.
 */

const TILE = 16;
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const INV_SQRT2 = Math.SQRT1_2;

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

/** Abramowitz & Stegun erf → standard-normal CDF. */
function normCdf(x) {
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

function normPdf(x) {
  return Math.exp(-0.5 * x * x) / SQRT_2PI;
}

/** Appendix A: exp(w Φ) ≈ 1+(e^w-1) Φ((z-μ-δ)/ν). */
function appendixAParams(w, sigma) {
  if (Math.abs(w) < 1e-8) {
    return { delta: (sigma * w) / SQRT_2PI, nu: sigma };
  }
  const delta = (sigma / w) * (-SQRT_2PI + Math.sqrt(2 * Math.PI + 2 * w * w));
  const u = delta / sigma;
  const phi = Math.max(normPdf(u), 1e-12);
  const nu =
    (sigma * Math.abs(Math.expm1(w)) * Math.exp(-w * normCdf(u))) /
    (w * phi * SQRT_2PI);
  return { delta, nu: Math.max(nu, 1e-8) };
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
 * Approximate-transmittance composite for one pixel's depth-ordered hits.
 * Maps each 2D α → w=−log(1−α). For splat i, T is Appendix-A collapsed
 * transmittance from *front* Gaussians only (μ_j > μ_i) — self excluded —
 * so an isolated splat matches classic C=albedo·α (bright center).
 *
 * Depth convention matches notes: μ = −viewDepth (front = larger μ), zf = 0.
 */
function compositeApprox(hits, bg, eps) {
  const n = hits.length;
  if (!n) {
    return { r: bg[0], g: bg[1], b: bg[2], T: 1, evals: 0, exited: false };
  }

  // Sort μ ascending (far → near). Front of i = indices j > i.
  hits.sort((a, b) => a.mu - b.mu);
  const zf = 0;

  for (let i = 0; i < n; i++) {
    const p = appendixAParams(hits[i].w, hits[i].sigma);
    hits[i].delta = p.delta;
    hits[i].nu = p.nu;
  }

  /** Appendix-A T(z) using only hits with indices in [lo, hi). */
  function T_slice(z, lo, hi) {
    if (hi <= lo) return 1;
    let tauFar = 0;
    let logPrefix = 0;
    let sum = 1;
    for (let j = lo; j < hi; j++) {
      const h = hits[j];
      const B = Math.expm1(h.w) * Math.exp(logPrefix);
      logPrefix += h.w;
      tauFar += h.w * normCdf((zf - h.mu) / h.sigma);
      sum += B * normCdf((z - h.mu - h.delta) / h.nu);
    }
    return Math.exp(-tauFar) * sum;
  }

  let r = 0,
    gch = 0,
    b = 0;
  let exited = false;

  // Front-to-back (largest μ first)
  for (let i = n - 1; i >= 0; i--) {
    const h = hits[i];
    // Front mixture only — do NOT include self (that caused dark cores)
    const T = Math.max(0, Math.min(1, T_slice(h.mu, i + 1, n)));
    const alpha = 1 - Math.exp(-h.w);
    r += h.color[0] * alpha * T;
    gch += h.color[1] * alpha * T;
    b += h.color[2] * alpha * T;
    if (T < eps) {
      exited = true;
      // Still count remaining as occluded; stop adding farther (smaller μ)
      break;
    }
  }

  // Background sees all Gaussians
  const zBack = hits[0].mu - 4 * hits[0].sigma;
  const Tbg = Math.max(0, Math.min(1, T_slice(zBack, 0, n)));
  r += bg[0] * Tbg;
  gch += bg[1] * Tbg;
  b += bg[2] * Tbg;
  return { r, g: gch, b, T: Tbg, evals: n, exited };
}

/**
 * @param {object} scene
 * @param {object} camera { eye, target, up, fovy, width, height }
 * @param {object} opts   { eps, background, maxPerPixel, blendMode }
 *   blendMode: "alpha" | "approx"
 */
export function renderFrame(scene, camera, opts = {}) {
  const width = camera.width | 0;
  const height = camera.height | 0;
  const eps = opts.eps ?? 1e-4;
  const bg = opts.background ?? [0.02, 0.03, 0.05];
  const maxPerPixel = opts.maxPerPixel ?? 64;
  const blendMode = opts.blendMode === "approx" ? "approx" : "alpha";
  const near = 0.05;
  const far = 200;
  const fovy = camera.fovy;

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
    });
  }

  // Front-to-back: smaller view-depth first
  projected.sort((a, b) => a.depth - b.depth);

  const tilesX = Math.ceil(width / TILE);
  const tilesY = Math.ceil(height / TILE);
  const tileLists = Array.from({ length: tilesX * tilesY }, () => []);

  for (const g of projected) {
    // ~3σ footprint
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

  let splatEvals = 0;
  let earlyOutPixels = 0;
  let pixelsFilled = 0;
  const hitScratch = [];

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
          let r = 0,
            gch = 0,
            b = 0;
          let exited = false;
          let hits = 0;

          if (blendMode === "approx") {
            hitScratch.length = 0;
            for (let k = 0; k < list.length; k++) {
              const s = list[k];
              const dx = x + 0.5 - s.sx;
              const dy = y + 0.5 - s.sy;
              const invR = 1 / s.radius;
              const u = dx * invR;
              const v = dy * invR;
              const r2 = u * u + v * v;
              if (r2 > 9.0) continue; // outside ~3σ

              const gauss = Math.exp(-0.5 * r2);
              const alpha = Math.min(0.999, s.opacity * gauss);
              if (alpha < 1e-4) continue;

              const w = -Math.log(1 - alpha);
              hitScratch.push({
                mu: -s.depth, // notes convention: larger μ = closer
                sigma: s.sigma,
                w,
                color: s.color,
              });
              if (hitScratch.length >= maxPerPixel) break;
            }
            const out = compositeApprox(hitScratch, bg, eps);
            r = out.r;
            gch = out.g;
            b = out.b;
            splatEvals += out.evals;
            hits = out.evals;
            exited = out.exited;
          } else {
            let T = 1.0;
            for (let k = 0; k < list.length; k++) {
              const s = list[k];
              const dx = x + 0.5 - s.sx;
              const dy = y + 0.5 - s.sy;
              const invR = 1 / s.radius;
              const u = dx * invR;
              const v = dy * invR;
              const r2 = u * u + v * v;
              if (r2 > 9.0) continue;

              const gauss = Math.exp(-0.5 * r2);
              const alpha = Math.min(0.99, s.opacity * gauss);
              if (alpha < 1e-4) continue;

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
  return {
    rgba,
    width,
    height,
    stats: {
      ms,
      projected: projected.length,
      splatEvals,
      earlyOutPixels,
      pixelsFilled,
      eps,
      blendMode,
      evalsPerPixel: pixelsFilled ? splatEvals / pixelsFilled : 0,
      radiusPx: projected.length
        ? {
            min: radiusMin,
            max: radiusMax,
            mean: radiusSum / projected.length,
          }
        : null,
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
