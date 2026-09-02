/**
 * View-dependent lava-lamp skybox (interior sphere, solid-angle blob field).
 */
import * as THREE from "three";
import type { Camera } from "three";

const SKY_RADIUS = 80;
const BLOB_COUNT = 10;

const vertexShader = /* glsl */ `
varying vec3 vWorldDir;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldDir = normalize(worldPos.xyz - cameraPosition);
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const fragmentShader = /* glsl */ `
uniform float uTime;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;

varying vec3 vWorldDir;

const float PI = 3.141592653589793;
const float BLOB_COUNT = ${BLOB_COUNT}.0;

float hash11(float n) {
  return fract(sin(n * 127.1) * 43758.5453);
}

vec3 hashUnit(float seed) {
  float id = seed * 17.3;
  float u = hash11(id) * 6.2831853;
  float v = hash11(id + 1.0) * 2.0 - 1.0;
  // Keep away from poles slightly for nicer motion.
  v = clamp(v, -0.88, 0.88);
  float r = sqrt(max(0.0, 1.0 - v * v));
  return normalize(vec3(cos(u) * r, v, sin(u) * r));
}

// Even-ish blob homes on S² (Fibonacci lattice).
vec3 fiboDir(float i) {
  float n = BLOB_COUNT;
  float phi = 1.618033988749;
  float theta = 2.0 * PI * fract(i / phi);
  float z = 1.0 - 2.0 * (i + 0.5) / n;
  float r = sqrt(max(0.0, 1.0 - z * z));
  return vec3(cos(theta) * r, z, sin(theta) * r);
}

vec3 rotateAxis(vec3 v, vec3 axis, float angle) {
  axis = normalize(axis);
  float c = cos(angle);
  float s = sin(angle);
  return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// Metaball in solid angle: peaks when dir aligns with center.
float metaballSolid(vec3 dir, vec3 center, float angRadius) {
  float d = clamp(dot(dir, center), -1.0, 1.0);
  float ang = acos(d);
  float ar2 = angRadius * angRadius;
  return ar2 / (ang * ang + 0.0018);
}

// One blob: drifts on a great circle around its home direction.
float blobField(vec3 dir, float t, float seed) {
  vec3 home = mix(fiboDir(seed), hashUnit(seed), 0.35);
  home = normalize(home);

  float phase = hash11(seed * 19.1) * 6.2831853;
  float speed = mix(0.1, 0.18, hash11(seed + 2.0));
  float drift = sin(t * speed + phase) * mix(0.38, 0.62, hash11(seed + 3.0));

  vec3 pole = normalize(cross(home, vec3(0.0, 1.0, 0.0)));
  if (dot(pole, pole) < 1e-4) {
    pole = normalize(cross(home, vec3(1.0, 0.0, 0.0)));
  }
  vec3 center = rotateAxis(home, pole, drift);

  // Small secondary wobble — still on S².
  vec3 wobbleAxis = normalize(cross(center, hashUnit(seed + 9.0)));
  center = rotateAxis(center, wobbleAxis, sin(t * speed * 1.25 + phase * 1.6) * 0.1);
  center = normalize(center);

  float angRadius = mix(0.22, 0.31, hash11(seed + 7.0));
  angRadius *= 1.0 + 0.05 * sin(t * 0.85 + phase);

  return metaballSolid(dir, center, angRadius);
}

void main() {
  vec3 dir = normalize(vWorldDir);
  float t = uTime;

  float f = 0.0;
  for (float i = 0.0; i < BLOB_COUNT; i += 1.0) {
    f += blobField(dir, t, i);
  }

  // Threshold tuned for angular metaballs (~50% wax coverage).
  float wax = smoothstep(0.7, 1.4, f);
  float hot = smoothstep(1.2, 2.55, f);

  // Subtle tint toward the upper hemisphere (world +Y).
  float heightGrad = smoothstep(-0.35, 0.9, dir.y);

  vec3 col = uColor1;
  col = mix(col, uColor2, wax * 0.28);
  col = mix(col, uColor3, hot * 0.16);
  col = mix(col, uColor3 * 1.04, hot * hot * 0.06);
  col = mix(col * 0.97, col, heightGrad * 0.1 + 0.9);

  gl_FragColor = vec4(col, 1.0);
}
`;

export interface LavaColors {
  lava1: string;
  lava2: string;
  lava3: string;
}

function hexToVec3(hex: string): THREE.Vector3 {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(full, 16) || 0;
  return new THREE.Vector3(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
}

export interface LavaBackground {
  mesh: THREE.Mesh;
  material: THREE.ShaderMaterial;
  syncCamera: (camera: Camera) => void;
  setTime: (timeSec: number) => void;
  setColors: (c: LavaColors) => void;
}

export function createLavaBackground(colors: LavaColors): LavaBackground {
  const geometry = new THREE.SphereGeometry(SKY_RADIUS, 24, 16);
  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
    uniforms: {
      uTime: { value: 0 },
      uColor1: { value: hexToVec3(colors.lava1) },
      uColor2: { value: hexToVec3(colors.lava2) },
      uColor3: { value: hexToVec3(colors.lava3) },
    },
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;

  return {
    mesh,
    material,
    syncCamera(camera) {
      mesh.position.copy(camera.position);
    },
    setTime(timeSec) {
      material.uniforms.uTime.value = timeSec;
    },
    setColors(c) {
      material.uniforms.uColor1.value.copy(hexToVec3(c.lava1));
      material.uniforms.uColor2.value.copy(hexToVec3(c.lava2));
      material.uniforms.uColor3.value.copy(hexToVec3(c.lava3));
    },
  };
}
