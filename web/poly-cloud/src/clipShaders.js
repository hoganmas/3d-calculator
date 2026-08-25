/** Fullscreen clip-grid raymarch from prebaked NDC γ(u) fibers. */

export const clipGridVertex = /* glsl */ `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/**
 * Atlas: RedFormat, width=gridW, height=gridH*nAlpha (flipY=false).
 * Stores γ_k for f = Σ γ_k u^k on u∈[-1,1]. Bake is often coarser than the
 * framebuffer — bilinearly interpolate γ so we don't show atlas cell blocks.
 */
export const clipGridFragment = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform sampler2D uAlphaTex;
uniform float uGridW;
uniform float uGridH;
uniform float uFbW;
uniform float uFbH;
uniform float uNAlpha;
uniform float uMax1d;
uniform float uHalf;
uniform float uScale;
uniform int uSteps;
uniform vec3 uCameraPos;
uniform mat3 uDirM;
uniform vec3 uAbsorbColor;
uniform vec3 uEmitColor;

float safeInv(float x) {
  float ax = abs(x);
  float sgn = x >= 0.0 ? 1.0 : -1.0;
  return sgn / max(ax, 1e-8);
}

bool intersectBox(vec3 ro, vec3 rd, float h, out float t0, out float t1) {
  vec3 invRd = vec3(safeInv(rd.x), safeInv(rd.y), safeInv(rd.z));
  vec3 tA = (-vec3(h) - ro) * invRd;
  vec3 tB = (vec3(h) - ro) * invRd;
  vec3 tmin = min(tA, tB);
  vec3 tmax = max(tA, tB);
  t0 = max(max(tmin.x, tmin.y), tmin.z);
  t1 = min(min(tmax.x, tmax.y), tmax.z);
  return t1 > max(t0, 0.0);
}

int clampi(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

float gammaTexel(int px, int py, int k) {
  float tu = (float(px) + 0.5) / uGridW;
  float tv = (float(k) * uGridH + float(py) + 0.5) / (uGridH * uNAlpha);
  return texture2D(uAlphaTex, vec2(tu, tv)).r;
}

void main() {
  // Full-res primary ray (not snapped to atlas centers)
  float ndcX = -1.0 + 2.0 * gl_FragCoord.x / uFbW;
  float ndcY = -1.0 + 2.0 * gl_FragCoord.y / uFbH;

  vec3 dirRaw = uDirM * vec3(ndcX, ndcY, 1.0);
  vec3 ro = uCameraPos;

  float tEnter, tExit;
  if (!intersectBox(ro, dirRaw, uHalf, tEnter, tExit)) discard;
  tEnter = max(tEnter, 0.0);
  if (tExit <= tEnter + 1e-6) discard;

  float tMid = 0.5 * (tEnter + tExit);
  float tHw = 0.5 * (tExit - tEnter);
  if (tHw < 1e-8) discard;

  // Continuous atlas coords; texel centers at integers 0..W-1
  float fx = (ndcX + 1.0) * 0.5 * uGridW - 0.5;
  float fy = (ndcY + 1.0) * 0.5 * uGridH - 0.5;
  int x0 = int(floor(fx));
  int y0 = int(floor(fy));
  float tx = clamp(fx - float(x0), 0.0, 1.0);
  float ty = clamp(fy - float(y0), 0.0, 1.0);
  int xMax = int(uGridW) - 1;
  int yMax = int(uGridH) - 1;
  int x1 = clampi(x0 + 1, 0, xMax);
  int y1 = clampi(y0 + 1, 0, yMax);
  x0 = clampi(x0, 0, xMax);
  y0 = clampi(y0, 0, yMax);

  float gamma[25];
  int M = int(uMax1d);
  for (int k = 0; k <= 24; k++) {
    if (k > M) {
      gamma[k] = 0.0;
      continue;
    }
    float g00 = gammaTexel(x0, y0, k);
    float g10 = gammaTexel(x1, y0, k);
    float g01 = gammaTexel(x0, y1, k);
    float g11 = gammaTexel(x1, y1, k);
    gamma[k] = mix(mix(g00, g10, tx), mix(g01, g11, tx), ty);
  }

  int steps = uSteps;
  if (steps < 8) steps = 8;
  if (steps > 96) steps = 96;
  float dt = (tExit - tEnter) / float(steps);
  float ds = length(dirRaw) * dt;

  vec3 rgb = vec3(0.0);
  float T = 1.0;
  float s = tEnter + 0.5 * dt;

  for (int i = 0; i < 96; i++) {
    if (i >= steps) break;
    if (T < 0.002) break;

    float u = (s - tMid) / tHw;
    float dens = 0.0;
    for (int k = 24; k >= 0; k--) {
      if (k > M) continue;
      dens = dens * u + gamma[k];
    }

    float sigma = max(0.0, uScale * dens);
    float absorb = exp(-sigma * ds);
    float opacity = 1.0 - absorb;
    rgb += T * opacity * (uEmitColor * sigma + uAbsorbColor * 0.15);
    T *= absorb;
    s += dt;
  }

  float a = 1.0 - T;
  if (a < 0.001) discard;
  gl_FragColor = vec4(rgb, a);
}
`;
