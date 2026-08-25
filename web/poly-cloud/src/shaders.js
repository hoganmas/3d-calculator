export const volumeVertex = /* glsl */ `
varying vec3 vWorldPos;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorldPos = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

/**
 * LOS reference raymarch: nested Horner → γ(u) → Beer–Lambert.
 * FIT_DEG / FIT_N / FIT_1D / FIT_1D_N are compile-time defines (set on fit).
 * Golden path is clip-grid (clipBakeGpu.js), not this shader.
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
