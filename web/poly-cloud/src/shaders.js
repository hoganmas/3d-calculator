export const volumeVertex = /* glsl */ `
varying vec3 vWorldPos;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorldPos = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`;

/**
 * World monomials (uploaded once). Per pixel:
 *
 *   Exact LOS restriction f(P0 + Du·u) → univariate γ(u) via nested Horner
 *   composition in u (tensor deg N → 1D deg ≤ 3N). Touches each coeff once
 *   in structured O(N⁴) work — not (3N+1) separate 3D evals (= worse O(N⁴)).
 *
 *   Then raymarch / Path C are O(steps · deg) Horner/Clenshaw on γ — linear in N.
 *
 * Note: dense (N+1)³ tensors cannot be O(N) per ray; Ω(N³) to read coeffs.
 * In the N=3…6 range even Θ(N³) still rises ~1.7–2× per +1 degree.
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
uniform int uProfileStage;
uniform vec3 uCameraPos;
uniform vec3 uAbsorbColor;
uniform vec3 uEmitColor;

varying vec3 vWorldPos;

float coeffAt(int idx) {
  float f = (float(idx) + 0.5) / uCoeffSize;
  return texture2D(uCoeffTex, vec2(f, 0.5)).r;
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

void clear19(inout float p[19]) {
  for (int i = 0; i < 19; i++) p[i] = 0.0;
}

float horner(float a[19], int deg, float x) {
  float s = 0.0;
  for (int i = 18; i >= 0; i--) {
    if (i > deg) continue;
    s = s * x + a[i];
  }
  return s;
}

/** p(u) ← p(u) · (a + b u) */
void mulLinear(inout float p[19], float a, float b) {
  float n[19];
  clear19(n);
  for (int i = 0; i <= 18; i++) {
    float v = p[i];
    if (v == 0.0) continue;
    n[i] += v * a;
    if (i + 1 < 19) n[i + 1] += v * b;
  }
  for (int i = 0; i < 19; i++) p[i] = n[i];
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

float sigmaU(float gamma[19], int deg, float u) {
  return max(0.0, uScale * horner(gamma, deg, u));
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

  // p = P0 + Du·u,  u ∈ [-1,1] on the segment (stable powers)
  vec3 P0 = ro + rd * tMid;
  vec3 Du = rd * tHw;

  int n = uDeg + 1;
  int n3 = n * n * n;
  int max1d = min(3 * uDeg, 18);

  float C[343];
  for (int i = 0; i < 343; i++) {
    if (i >= n3) {
      C[i] = 0.0;
      continue;
    }
    C[i] = coeffAt(i);
  }

  // Precompute (P0.z + Du.z·u)^k as univariate polys — O(N²)
  float zPow[133]; // 7 * 19
  {
    float pk[19];
    clear19(pk);
    pk[0] = 1.0;
    for (int k = 0; k <= 6; k++) {
      if (k > uDeg) break;
      for (int m = 0; m < 19; m++) zPow[k * 19 + m] = pk[m];
      if (k == uDeg) break;
      mulLinear(pk, P0.z, Du.z);
    }
  }

  // Nested Horner in x,y with univariate arithmetic in u — exact γ(u)
  //   γ = (...(s_N · Lx + s_{N-1}) · Lx + … );  s_i from y,z slices
  float gamma[19];
  clear19(gamma);

  for (int i = 6; i >= 0; i--) {
    if (i > uDeg) continue;

    float si[19];
    clear19(si);

    for (int j = 6; j >= 0; j--) {
      if (j > uDeg) continue;

      // row(u) = Σ_k C_ijk (P0z + Duz u)^k
      float row[19];
      clear19(row);
      for (int k = 0; k <= 6; k++) {
        if (k > uDeg) break;
        float c = C[i + j * n + k * n * n];
        if (abs(c) < 1e-20) continue;
        for (int m = 0; m <= 18; m++) {
          row[m] += c * zPow[k * 19 + m];
        }
      }

      // si ← si · Ly + row  (Horner in y)
      if (j < uDeg) {
        mulLinear(si, P0.y, Du.y);
      }
      for (int m = 0; m < 19; m++) si[m] += row[m];
    }

    // gamma ← gamma · Lx + si  (Horner in x)
    if (i < uDeg) {
      mulLinear(gamma, P0.x, Du.x);
    }
    for (int m = 0; m < 19; m++) gamma[m] += si[m];
  }

  if (uProfileStage == 1) {
    float s0 = abs(uScale * horner(gamma, max1d, 0.0));
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
    float prevS = sigmaU(gamma, max1d, -1.0);
    float tau = 0.0;
    for (int s = 1; s <= 64; s++) {
      if (s > nGrid) break;
      float u = -1.0 + duG * float(s);
      float sg = sigmaU(gamma, max1d, u);
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

      float sigma = sigmaU(gamma, max1d, u);
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

  // ── Raymarch: O(steps · 3N) Horner — linear in degree ───────────────────
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

    float sigma = sigmaU(gamma, max1d, u);
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
