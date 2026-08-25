export const volumeVertex = /* glsl */ `
varying vec3 vWorldPos;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorldPos = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

/**
 * World monomials. FIT_DEG / FIT_N / FIT_1D / FIT_1D_N are compile-time
 * defines (set on fit) so loops and arrays scale with degree — no fixed
 * max-N tax from unrolled deg-8 bodies or a fixed local coeff buffer.
 */
export const volumeFragment = /* glsl */ `
precision highp float;
precision highp sampler2D;

#ifndef FIT_DEG
#define FIT_DEG 4
#endif
#ifndef FIT_N
#define FIT_N 5
#endif
#ifndef FIT_1D
#define FIT_1D 12
#endif
#ifndef FIT_1D_N
#define FIT_1D_N 13
#endif

uniform sampler2D uCoeffTex;
uniform float uCoeffTexW;
uniform float uCoeffTexH;
uniform float uHalf;
uniform float uScale;
uniform int uSteps;
uniform int uMode;
uniform int uTDeg;
uniform int uProfileStage;
uniform vec3 uCameraPos;
uniform vec3 uAbsorbColor;
uniform vec3 uEmitColor;

varying vec3 vWorldPos;

float coeffAt(int idx) {
  float w = uCoeffTexW;
  float x = mod(float(idx), w);
  float y = floor(float(idx) / w);
  return texture2D(uCoeffTex, vec2((x + 0.5) / w, (y + 0.5) / uCoeffTexH)).r;
}

float safeInv(float x) {
  float ax = abs(x);
  float s = x >= 0.0 ? 1.0 : -1.0;
  return s / max(ax, 1e-8);
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

void clear1d(inout float p[FIT_1D_N]) {
  for (int i = 0; i < FIT_1D_N; i++) p[i] = 0.0;
}

float horner(float a[FIT_1D_N], float x) {
  float s = 0.0;
  for (int i = FIT_1D; i >= 0; i--) {
    s = s * x + a[i];
  }
  return s;
}

void mulLinear(inout float p[FIT_1D_N], float a, float b) {
  float n[FIT_1D_N];
  clear1d(n);
  for (int i = 0; i < FIT_1D_N; i++) {
    float v = p[i];
    if (v == 0.0) continue;
    n[i] += v * a;
    if (i + 1 < FIT_1D_N) n[i + 1] += v * b;
  }
  for (int i = 0; i < FIT_1D_N; i++) p[i] = n[i];
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

float clenshaw9(float ck[9], int deg, float u) {
  float b1 = 0.0;
  float b2 = 0.0;
  for (int k = 8; k >= 1; k--) {
    if (k > deg) continue;
    float b0 = 2.0 * u * b1 - b2 + ck[k];
    b2 = b1;
    b1 = b0;
  }
  return u * b1 - b2 + ck[0];
}

float sigmaU(float gamma[FIT_1D_N], float u) {
  return max(0.0, uScale * horner(gamma, u));
}

void main() {
  vec3 ro = uCameraPos;
  vec3 rd = normalize(vWorldPos - uCameraPos);

  float tEnter, tExit;
  if (!intersectBox(ro, rd, uHalf, tEnter, tExit)) discard;
  tEnter = max(tEnter, 0.0);
  if (tExit <= tEnter + 1e-5) discard;

  float tMid = 0.5 * (tEnter + tExit);
  float tHw = 0.5 * (tExit - tEnter);
  if (tHw < 1e-8) discard;

  vec3 P0 = ro + rd * tMid;
  vec3 Du = rd * tHw;

  // (P0.z + Du.z·u)^k — only FIT_DEG+1 powers
  float zPow[FIT_N * FIT_1D_N];
  {
    float pk[FIT_1D_N];
    clear1d(pk);
    pk[0] = 1.0;
    for (int k = 0; k <= FIT_DEG; k++) {
      for (int m = 0; m < FIT_1D_N; m++) zPow[k * FIT_1D_N + m] = pk[m];
      if (k == FIT_DEG) break;
      mulLinear(pk, P0.z, Du.z);
    }
  }

  // Exact LOS γ(u); loops are compile-time FIT_DEG (no max-N dead work)
  float gamma[FIT_1D_N];
  clear1d(gamma);

  for (int i = FIT_DEG; i >= 0; i--) {
    float si[FIT_1D_N];
    clear1d(si);

    for (int j = FIT_DEG; j >= 0; j--) {
      float row[FIT_1D_N];
      clear1d(row);
      for (int k = 0; k <= FIT_DEG; k++) {
        float c = coeffAt(i + j * FIT_N + k * FIT_N * FIT_N);
        if (abs(c) < 1e-20) continue;
        for (int m = 0; m < FIT_1D_N; m++) {
          row[m] += c * zPow[k * FIT_1D_N + m];
        }
      }

      if (j < FIT_DEG) {
        mulLinear(si, P0.y, Du.y);
      }
      for (int m = 0; m < FIT_1D_N; m++) si[m] += row[m];
    }

    if (i < FIT_DEG) {
      mulLinear(gamma, P0.x, Du.x);
    }
    for (int m = 0; m < FIT_1D_N; m++) gamma[m] += si[m];
  }

  if (uProfileStage == 1) {
    float s0 = abs(uScale * horner(gamma, 0.0));
    gl_FragColor = vec4(uEmitColor * s0, clamp(s0, 0.05, 1.0));
    return;
  }

  if (uMode == 1) {
    int TD = uTDeg;
    if (TD < 1) TD = 1;
    if (TD > 8) TD = 8;
    int nNodes = TD + 1;

    float uNode[9];
    float tauNode[9];
    float got[9];
    for (int j = 0; j <= 8; j++) {
      if (j >= nNodes) break;
      float theta = 3.141592653589793 * (2.0 * float(j) + 1.0) / (2.0 * float(nNodes));
      uNode[j] = cos(theta);
      tauNode[j] = 0.0;
      got[j] = 0.0;
    }

    int nGrid = 32;
    float duG = 2.0 / float(nGrid);
    float prevS = sigmaU(gamma, -1.0);
    float tau = 0.0;
    for (int s = 1; s <= 64; s++) {
      if (s > nGrid) break;
      float u = -1.0 + duG * float(s);
      float sg = sigmaU(gamma, u);
      tau += 0.5 * (prevS + sg) * duG * tHw;
      prevS = sg;
      for (int j = 0; j <= 8; j++) {
        if (j >= nNodes) break;
        if (got[j] < 0.5 && uNode[j] <= u) {
          tauNode[j] = tau;
          got[j] = 1.0;
        }
      }
    }
    for (int j = 0; j <= 8; j++) {
      if (j >= nNodes) break;
      if (got[j] < 0.5) tauNode[j] = tau;
    }

    float tauCk[9];
    for (int k = 0; k <= 8; k++) {
      if (k > TD) {
        tauCk[k] = 0.0;
        continue;
      }
      float s = 0.0;
      for (int j = 0; j <= 8; j++) {
        if (j >= nNodes) break;
        s += tauNode[j] * chebT(k, uNode[j]);
      }
      tauCk[k] = (k == 0 ? 1.0 : 2.0) * s / float(nNodes);
    }

    if (uProfileStage == 2) {
      float s = 0.0;
      for (int j = 0; j <= 8; j++) {
        if (j >= nNodes) break;
        s += exp(-min(tauNode[j], 80.0));
      }
      s /= float(nNodes);
      gl_FragColor = vec4(uEmitColor * s, clamp(1.0 - s, 0.05, 1.0));
      return;
    }

    if (uProfileStage == 3) {
      float te = exp(-min(max(0.0, clenshaw9(tauCk, TD, 1.0)), 80.0));
      gl_FragColor = vec4(uEmitColor * te, clamp(1.0 - te, 0.05, 1.0));
      return;
    }

    int steps = uSteps;
    if (steps < 16) steps = 16;
    if (steps > 96) steps = 96;
    float dt = (tExit - tEnter) / float(steps);
    float du = dt / tHw;

    vec3 rgb = vec3(0.0);
    float u = -1.0 + 0.5 * du;
    for (int s = 0; s < 96; s++) {
      if (s >= steps) break;
      float tauHat = max(0.0, clenshaw9(tauCk, TD, u));
      if (tauHat > 80.0) tauHat = 80.0;
      float T = exp(-tauHat);
      if (T < 0.002) break;

      float sigma = sigmaU(gamma, u);
      float absorb = exp(-sigma * dt);
      float opacity = 1.0 - absorb;
      rgb += T * opacity * (uEmitColor * sigma + uAbsorbColor * 0.15);
      u += du;
    }

    float tauExit = max(0.0, clenshaw9(tauCk, TD, 1.0));
    float opacity = clamp(1.0 - exp(-min(tauExit, 80.0)), 0.0, 1.0);
    if (opacity < 0.001 && dot(rgb, rgb) < 1e-10) discard;
    gl_FragColor = vec4(rgb, opacity);
    return;
  }

  int steps = uSteps;
  if (steps < 8) steps = 8;
  if (steps > 96) steps = 96;
  float dt = (tExit - tEnter) / float(steps);
  float du = dt / tHw;

  vec3 rgb = vec3(0.0);
  float T = 1.0;
  float u = -1.0 + 0.5 * du;
  for (int s = 0; s < 96; s++) {
    if (s >= steps) break;
    if (T < 0.002) break;

    float sigma = sigmaU(gamma, u);
    float absorb = exp(-sigma * dt);
    float opacity = 1.0 - absorb;
    rgb += T * opacity * (uEmitColor * sigma + uAbsorbColor * 0.15);
    T *= absorb;
    u += du;
  }

  float a = 1.0 - T;
  if (a < 0.001) discard;
  gl_FragColor = vec4(rgb, a);
}
`;
