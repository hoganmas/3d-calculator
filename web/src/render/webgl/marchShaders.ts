/** WebGL fallback: fullscreen Beer march on a single summed dens volume (IDCT bake). */

export const clipGridVertex = /* glsl */ `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/**
 * Volume: RedFormat DataTexture width=M, height=M*M (slices stacked in Y).
 * dens[ix + iy*M + iz*M*M] at texel (ix, iy + iz*M).
 */
export const clipGridFragment = /* glsl */ `
precision highp float;
precision highp sampler2D;

uniform sampler2D uVolumeTex;
uniform float uGridM;
uniform float uFbW;
uniform float uFbH;
/** Framebuffer viewport (x, y, width, height) — required for stereo XR eyes. */
uniform vec4 uViewport;
uniform float uHalf;
uniform float uScale;
uniform int uSteps;
uniform vec3 uCameraPos;
uniform mat3 uDirM;
/** Inverse of xrWorld matrixWorld — volume box stays local to the grab root. */
uniform mat4 uInvXrWorld;
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

float densAt(int ix, int iy, int iz) {
  int M = int(uGridM);
  ix = clamp(ix, 0, M - 1);
  iy = clamp(iy, 0, M - 1);
  iz = clamp(iz, 0, M - 1);
  float tu = (float(ix) + 0.5) / uGridM;
  float tv = (float(iy + iz * M) + 0.5) / (uGridM * uGridM);
  return texture2D(uVolumeTex, vec2(tu, tv)).r;
}

float chebIndex(float xi) {
  float x = clamp(xi, -1.0, 1.0);
  return uGridM / 3.141592653589793 * acos(x) - 0.5;
}

float sampleVolume(vec3 p) {
  vec3 xi = clamp(p / uHalf, vec3(-1.0), vec3(1.0));
  float fx = chebIndex(xi.x);
  float fy = chebIndex(xi.y);
  float fz = chebIndex(xi.z);
  int x0 = int(floor(fx));
  int y0 = int(floor(fy));
  int z0 = int(floor(fz));
  float tx = clamp(fx - float(x0), 0.0, 1.0);
  float ty = clamp(fy - float(y0), 0.0, 1.0);
  float tz = clamp(fz - float(z0), 0.0, 1.0);
  float c00 = mix(densAt(x0, y0, z0), densAt(x0 + 1, y0, z0), tx);
  float c10 = mix(densAt(x0, y0 + 1, z0), densAt(x0 + 1, y0 + 1, z0), tx);
  float c01 = mix(densAt(x0, y0, z0 + 1), densAt(x0 + 1, y0, z0 + 1), tx);
  float c11 = mix(densAt(x0, y0 + 1, z0 + 1), densAt(x0 + 1, y0 + 1, z0 + 1), tx);
  return mix(mix(c00, c10, ty), mix(c01, c11, ty), tz);
}

void main() {
  float vw = max(uViewport.z, 1.0);
  float vh = max(uViewport.w, 1.0);
  float ndcX = -1.0 + 2.0 * (gl_FragCoord.x - uViewport.x) / vw;
  float ndcY = -1.0 + 2.0 * (gl_FragCoord.y - uViewport.y) / vh;
  vec3 dirWorld = uDirM * vec3(ndcX, ndcY, 1.0);
  vec3 ro = (uInvXrWorld * vec4(uCameraPos, 1.0)).xyz;
  vec3 dirRaw = mat3(uInvXrWorld) * dirWorld;

  float tEnter, tExit;
  if (!intersectBox(ro, dirRaw, uHalf, tEnter, tExit)) discard;
  tEnter = max(tEnter, 0.0);
  if (tExit <= tEnter + 1e-6) discard;

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

    vec3 p = ro + s * dirRaw;
    float dval = sampleVolume(p);
    if (!(dval == dval)) dval = 0.0;
    dval = clamp(dval, -4.0, 8.0);

    float sigma = max(0.0, uScale * dval);
    sigma = min(sigma, 40.0);
    float absorb = exp(-sigma * ds);
    float opacity = 1.0 - absorb;
    float gt = clamp(p.y / uHalf * 0.5 + 0.5, 0.0, 1.0);
    gt = clamp(mix(gt, length(p) / max(uHalf * 1.414214, 1e-6), 0.38), 0.0, 1.0);
    vec3 emitGrad = mix(uAbsorbColor, uEmitColor, gt);
    rgb += T * opacity * emitGrad * (1.0 + 0.42);
    T *= absorb;
    s += dt;
  }

  float a = 1.0 - T;
  if (a < 0.001) discard;
  gl_FragColor = vec4(rgb, a);
}
`;
