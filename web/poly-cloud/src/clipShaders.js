/** Fullscreen clip-grid Beer raymarch from prebaked NDC density samples.
 *  Golden path marcher.
 */

export const clipGridVertex = /* glsl */ `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/**
 * Atlas: RedFormat, width=gridW, height=gridH*nAlpha (flipY=false).
 * Block j stores dens at Chebyshev-root node u_j on the view-fixed fiber
 * t = tMid + u·tHw. Bilinear in screen; barycentric in u.
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
uniform float uTMid;
uniform float uTHw;
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

float densTexel(int px, int py, int j) {
  float tu = (float(px) + 0.5) / uGridW;
  float tv = (float(j) * uGridH + float(py) + 0.5) / (uGridH * uNAlpha);
  return texture2D(uAlphaTex, vec2(tu, tv)).r;
}

// Piecewise-linear in u between adjacent Chebyshev roots.
// Global barycentric through zeroed/exterior nodes overshoots → white fireflies.
#ifndef CLIP_1D_N
#define CLIP_1D_N 49
#endif

float densAtU(float u, float dens[CLIP_1D_N], int M) {
  if (M <= 1) return dens[0];
  float invM = 1.0 / float(M);
  float uFirst = cos(3.141592653589793 * 0.5 * invM);
  float uLast = cos(3.141592653589793 * (float(M) - 0.5) * invM);
  if (u >= uFirst) return dens[0];
  if (u <= uLast) return dens[M - 1];
  for (int j = 0; j < CLIP_1D_N - 1; j++) {
    if (j + 1 >= M) break;
    float u0 = cos(3.141592653589793 * (float(j) + 0.5) * invM);
    float u1 = cos(3.141592653589793 * (float(j) + 1.5) * invM);
    if (u <= u0 && u >= u1) {
      float t = (u0 - u) / max(u0 - u1, 1e-12);
      return mix(dens[j], dens[j + 1], clamp(t, 0.0, 1.0));
    }
  }
  return dens[M - 1];
}

void main() {
  float ndcX = -1.0 + 2.0 * gl_FragCoord.x / uFbW;
  float ndcY = -1.0 + 2.0 * gl_FragCoord.y / uFbH;

  vec3 dirRaw = uDirM * vec3(ndcX, ndcY, 1.0);
  vec3 ro = uCameraPos;

  float tEnter, tExit;
  if (!intersectBox(ro, dirRaw, uHalf, tEnter, tExit)) discard;
  tEnter = max(tEnter, 0.0);
  if (tExit <= tEnter + 1e-6) discard;
  if (uTHw < 1e-8) discard;

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

  float dens[CLIP_1D_N];
  int M = int(uNAlpha);
  for (int k = 0; k < CLIP_1D_N; k++) {
    if (k >= M) {
      dens[k] = 0.0;
      continue;
    }
    float g00 = densTexel(x0, y0, k);
    float g10 = densTexel(x1, y0, k);
    float g01 = densTexel(x0, y1, k);
    float g11 = densTexel(x1, y1, k);
    dens[k] = mix(mix(g00, g10, tx), mix(g01, g11, tx), ty);
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

    float u = (s - uTMid) / uTHw;
    float dval = densAtU(u, dens, M);
    // Sample point outside the fit box → no density (and kill NaNs/spikes).
    vec3 p = ro + s * dirRaw;
    if (abs(p.x) > uHalf || abs(p.y) > uHalf || abs(p.z) > uHalf) dval = 0.0;
    if (!(dval == dval)) dval = 0.0;
    dval = clamp(dval, -4.0, 8.0);

    float sigma = max(0.0, uScale * dval);
    sigma = min(sigma, 40.0);
    float absorb = exp(-sigma * ds);
    float opacity = 1.0 - absorb;
    rgb += T * opacity * (uEmitColor * sigma + uAbsorbColor * 0.15);
    T *= absorb;
    s += dt;
  }

  float a = 1.0 - T;
  if (a < 0.001) discard;
  gl_FragColor = vec4(rgb * a, a);
}
`;
