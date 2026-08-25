export const volumeVertex = /* glsl */ `
varying vec3 vWorldPos;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorldPos = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

/**
 * Camera-centered monomial density tensor → 1D α(t) along the ray.
 *
 * uMode 0 — raymarch baseline: T *= exp(−σ Δt)
 * uMode 1 — Path C: Chebyshev-fit T(t)=exp(−∫_{tEnter}^t σ), then I=∫ σ T_cheb dt
 */
export const volumeFragment = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform sampler2D uCoeffTex;
uniform float uCoeffSize;
uniform int uDeg;
uniform float uHalf;
uniform float uScale;
uniform int uSteps;
uniform int uMode;
uniform int uTDeg;
uniform vec3 uCameraPos;
uniform vec3 uAbsorbColor;
uniform vec3 uEmitColor;

varying vec3 vWorldPos;

float coeffAt(int idx) {
  float f = (float(idx) + 0.5) / uCoeffSize;
  return texture2D(uCoeffTex, vec2(f, 0.5)).r;
}

float ipow(float b, int n) {
  float r = 1.0;
  for (int i = 0; i < 18; i++) {
    if (i >= n) break;
    r *= b;
  }
  return r;
}

bool intersectBox(vec3 ro, vec3 rd, float h, out float t0, out float t1) {
  vec3 invRd = 1.0 / rd;
  vec3 tA = (-vec3(h) - ro) * invRd;
  vec3 tB = (vec3(h) - ro) * invRd;
  vec3 tmin = min(tA, tB);
  vec3 tmax = max(tA, tB);
  t0 = max(max(tmin.x, tmin.y), tmin.z);
  t1 = min(min(tmax.x, tmax.y), tmax.z);
  return t1 > max(t0, 0.0);
}

void clear19(inout float p[19]) {
  for (int i = 0; i < 19; i++) p[i] = 0.0;
}

float horner(float a[19], int deg, float t) {
  float s = 0.0;
  for (int i = 18; i >= 0; i--) {
    if (i > deg) continue;
    s = s * t + a[i];
  }
  return s;
}

float integPoly(float a[19], int deg, float t0, float t1) {
  float s = 0.0;
  for (int m = 0; m <= 18; m++) {
    if (m > deg) break;
    float c = a[m] / float(m + 1);
    s += c * (ipow(t1, m + 1) - ipow(t0, m + 1));
  }
  return s;
}

float chebT(int k, float u) {
  if (k == 0) return 1.0;
  if (k == 1) return u;
  float t0 = 1.0;
  float t1 = u;
  for (int j = 2; j <= 8; j++) {
    if (j > k) break;
    float t2 = 2.0 * u * t1 - t0;
    t0 = t1;
    t1 = t2;
  }
  return t1;
}

void mul19(inout float a[19], float b[19]) {
  float r[19];
  clear19(r);
  for (int i = 0; i <= 18; i++) {
    if (a[i] == 0.0) continue;
    for (int j = 0; j <= 18 - i; j++) {
      if (b[j] == 0.0) continue;
      r[i + j] += a[i] * b[j];
    }
  }
  for (int i = 0; i < 19; i++) a[i] = r[i];
}

void scale19(inout float a[19], float s) {
  for (int i = 0; i < 19; i++) a[i] *= s;
}

// Build monomials of T_0..T_n in u (rows packed as deg+1 consecutive coeffs, max deg 8)
void chebBasisMono(int n, out float basis[81]) {
  for (int i = 0; i < 81; i++) basis[i] = 0.0;
  // T0 = 1, T1 = u
  basis[0] = 1.0;
  if (n >= 1) basis[1 * 9 + 1] = 1.0;
  for (int d = 2; d <= 8; d++) {
    if (d > n) break;
    // T_d = 2u T_{d-1} - T_{d-2}
    for (int i = 0; i <= 8; i++) {
      float prev = basis[(d - 1) * 9 + i];
      if (i + 1 <= 8) basis[d * 9 + (i + 1)] += 2.0 * prev;
      basis[d * 9 + i] -= basis[(d - 2) * 9 + i];
    }
  }
}

void main() {
  vec3 ro = uCameraPos;
  vec3 rd = normalize(vWorldPos - uCameraPos);

  float tEnter, tExit;
  if (!intersectBox(ro, rd, uHalf, tEnter, tExit)) discard;
  tEnter = max(tEnter, 0.0);
  if (tExit <= tEnter) discard;

  int n = uDeg + 1;
  int max1d = min(3 * uDeg, 18);

  float alpha[19];
  clear19(alpha);

  for (int i = 0; i <= 6; i++) {
    if (i > uDeg) break;
    float dxI = ipow(rd.x, i);
    for (int j = 0; j <= 6; j++) {
      if (j > uDeg) break;
      float dyJ = ipow(rd.y, j);
      for (int k = 0; k <= 6; k++) {
        if (k > uDeg) break;
        int m = i + j + k;
        if (m > 18) continue;
        float c = coeffAt(i + j * n + k * n * n);
        alpha[m] += c * dxI * dyJ * ipow(rd.z, k);
      }
    }
  }

  // σ(t) = max(0, scale * α(t)) — poly uses unclamped α; clamp at eval / τ
  float sigmaPoly[19];
  for (int i = 0; i < 19; i++) sigmaPoly[i] = uScale * alpha[i];

  if (uMode == 1) {
    // ── Path C: Chebyshev fit of T(t)=exp(−∫_{tEnter}^t σ) ───────────────
    int TD = uTDeg;
    if (TD < 1) TD = 1;
    if (TD > 8) TD = 8;

    float tMid = 0.5 * (tEnter + tExit);
    float tHw = 0.5 * (tExit - tEnter);
    if (tHw < 1e-8) discard;

    float uNode[9];
    float Tnode[9];
    for (int j = 0; j <= 8; j++) {
      if (j > TD) break;
      float u = cos(3.141592653589793 * (2.0 * float(j) + 1.0) / (2.0 * float(TD + 1)));
      uNode[j] = u;
      float tj = tMid + tHw * u;
      float tau = integPoly(sigmaPoly, max1d, tEnter, tj);
      tau = max(0.0, tau);
      if (tau > 80.0) tau = 80.0;
      Tnode[j] = exp(-tau);
    }

    float ck[9];
    for (int k = 0; k <= 8; k++) {
      if (k > TD) break;
      float s = 0.0;
      for (int j = 0; j <= 8; j++) {
        if (j > TD) break;
        s += Tnode[j] * chebT(k, uNode[j]);
      }
      ck[k] = (k == 0 ? 1.0 : 2.0) * s / float(TD + 1);
    }

    // Σ ck T_k(u) → monomials in u
    float basis[81];
    chebBasisMono(TD, basis);
    float polyU[19];
    clear19(polyU);
    for (int k = 0; k <= 8; k++) {
      if (k > TD) break;
      if (abs(ck[k]) < 1e-12) continue;
      for (int i = 0; i <= 8; i++) {
        if (i > k) break;
        polyU[i] += ck[k] * basis[k * 9 + i];
      }
    }

    // u = (t - tMid)/tHw = a t + b → monomials in t
    float aMap = 1.0 / tHw;
    float bMap = -tMid / tHw;
    float polyT[19];
    clear19(polyT);
    // Build powers of (a t + b) and accumulate
    float powLin[19];
    clear19(powLin);
    powLin[0] = 1.0; // (at+b)^0
    for (int p = 0; p <= 8; p++) {
      if (p > TD) break;
      float c = polyU[p];
      if (abs(c) > 1e-12) {
        for (int i = 0; i <= 8; i++) {
          if (i > p) break;
          polyT[i] += c * powLin[i];
        }
      }
      if (p == TD) break;
      // powLin *= (a t + b)
      float next[19];
      clear19(next);
      for (int i = 0; i <= 8; i++) {
        if (powLin[i] == 0.0) continue;
        next[i] += powLin[i] * bMap;
        if (i + 1 < 19) next[i + 1] += powLin[i] * aMap;
      }
      for (int i = 0; i < 19; i++) powLin[i] = next[i];
    }

    // I = ∫ σ(t) T_cheb(t) dt
    float integrand[19];
    for (int i = 0; i < 19; i++) integrand[i] = sigmaPoly[i];
    mul19(integrand, polyT);
    float I = integPoly(integrand, 18, tEnter, tExit);
    I = max(0.0, I);

    float Texit = max(0.0, horner(polyT, TD, tExit));
    float opacity = clamp(1.0 - Texit, 0.0, 1.0);
    if (opacity < 0.001 && I < 1e-6) discard;

    vec3 rgb = uEmitColor * I + uAbsorbColor * (0.15 * I);
    gl_FragColor = vec4(rgb, max(opacity, min(1.0, I)));
    return;
  }

  // ── Raymarch baseline ────────────────────────────────────────────────────
  int steps = uSteps;
  if (steps < 8) steps = 8;
  if (steps > 128) steps = 128;
  float dt = (tExit - tEnter) / float(steps);

  vec3 rgb = vec3(0.0);
  float T = 1.0;
  float t = tEnter + 0.5 * dt;

  for (int s = 0; s < 128; s++) {
    if (s >= steps) break;
    if (T < 0.002) break;

    float sigma = max(0.0, horner(sigmaPoly, max1d, t));
    float optical = sigma * dt;
    float absorb = exp(-optical);
    float opacity = 1.0 - absorb;
    rgb += T * opacity * (uEmitColor * sigma + uAbsorbColor * 0.15);
    T *= absorb;
    t += dt;
  }

  float a = 1.0 - T;
  if (a < 0.001) discard;
  gl_FragColor = vec4(rgb, a);
}
`;
