struct FlowParticleParams {
  viewProj: mat4x4f,
  ro: vec3f,
  half: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  fbW: f32,
  fbH: f32,
  flowLayerStart: f32,
  flowAgeMax: f32,
  flowOpacity: f32,
  _pad0: f32,
  particleCount: u32,
  trailSteps: u32,
  trailSegCount: u32,
  segStride: u32,
  trailWidth: f32,
  flowVMin: f32,
  flowVMax: f32,
  maxSegLen: f32,
  ribbonPxToWorld: f32,
  _padGrid: u32,
  _padVelBase: f32,
  flowCol1: vec3f,
  _padC1: f32,
  flowCol2: vec3f,
  _padC2: f32,
}

@group(0) @binding(0) var<uniform> u: FlowParticleParams;
@group(0) @binding(2) var<storage, read> particleLayers: array<u32>;
@group(0) @binding(3) var<storage, read> layerGrads: array<vec4f>;
@group(0) @binding(4) var occlTex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> trailHist: array<f32>;
@group(0) @binding(6) var<storage, read> sortOrder: array<u32>;

const MAX_GRAD_STOPS: u32 = {{MAX_GRAD_STOPS}}u;
const MAX_TRAIL_STEPS: u32 = {{MAX_FLOW_TRAIL_STEPS}}u;
const TRAIL_STRIDE: u32 = 5u;

fn sampleGradStopsLayer(layer: u32, t: f32) -> vec3f {
  let base = layer * MAX_GRAD_STOPS;
  var stops: array<vec4f, MAX_GRAD_STOPS>;
  for (var i: u32 = 0u; i < MAX_GRAD_STOPS; i++) {
    stops[i] = layerGrads[base + i];
  }
  let n = max(min(u32(stops[0].w), MAX_GRAD_STOPS), 1u);
  if (n <= 1u) { return stops[0].xyz; }
  let x = clamp(t, 0.0, 1.0) * f32(n - 1u);
  let i = min(u32(floor(x)), n - 2u);
  let f = x - f32(i);
  return mix(stops[i].xyz, stops[i + 1u].xyz, f);
}

fn speedColor(speed: f32, flowIdx: u32) -> vec3f {
  let densLayer = u32(max(u.flowLayerStart, 0.0)) + flowIdx;
  let col1 = sampleGradStopsLayer(densLayer, 0.0);
  let col2 = sampleGradStopsLayer(densLayer, 1.0);
  let vSpan = max(u.flowVMax - u.flowVMin, max(u.flowVMax, 1e-6) * 0.08);
  let speedNorm = clamp((speed - u.flowVMin) / vSpan, 0.0, 1.0);
  return mix(col1, col2, speedNorm);
}

fn trailSlotBase(pIdx: u32, slot: u32) -> u32 {
  return (pIdx * MAX_TRAIL_STEPS + slot) * TRAIL_STRIDE;
}

fn trailPosAge(pIdx: u32, slot: u32) -> vec4f {
  let o = trailSlotBase(pIdx, slot);
  return vec4f(trailHist[o], trailHist[o + 1u], trailHist[o + 2u], trailHist[o + 3u]);
}

fn trailSpeed(pIdx: u32, slot: u32) -> f32 {
  return trailHist[trailSlotBase(pIdx, slot) + 4u];
}

fn boxClipAlpha(world: vec3f) -> f32 {
  let half = u.half;
  let m = max(max(abs(world.x), abs(world.y)), abs(world.z));
  // Hard clip at the fit-box faces (hairline AA). Particles may still advect outside;
  // only the portion inside the box is shaded.
  return 1.0 - smoothstep(half, half * 1.012, m);
}

// slot 0 = newest (leading edge), slot N-1 = oldest (trailing edge); u=0 newest, u=1 oldest.
fn ribbonU(slot: u32) -> f32 {
  let n = max(u.trailSteps, 2u);
  return f32(slot) / f32(n - 1u);
}

fn ribbonWidthEnvelope(u: f32) -> f32 {
  // Bulge along the trail body; the leading cap on segment 0 supplies the rounded head.
  return sin(3.14159265 * clamp(u, 0.0, 1.0));
}

fn ribbonAlphaEnvelope(u: f32) -> f32 {
  let width = ribbonWidthEnvelope(u);
  // Fade opacity toward the oldest end (u→1).
  let tailFade = 1.0 - smoothstep(0.62, 1.0, u);
  return width * tailFade;
}

// Trail slot age encodes life phase (CPU syncFlowParticleTrailLife):
//   age in [-2, -1] → ghost fade-out (−1 full … −2 invisible); trail geometry frozen
//   age in [-0.5, 0) → spawn fade-in
//   age >= 0 → live (full opacity; decay happens after death as a ghost)
fn trailLifeAlpha(age: f32) -> f32 {
  // Ghost range includes −1 (full). Must not fall through to spawn branch.
  if (age <= -1.0) {
    return smoothstep(-2.0, -1.0, age);
  }
  if (age < 0.0) {
    return smoothstep(-0.5, 0.0, age);
  }
  return 1.0;
}

// Trail centerline tangent: from newest toward older samples (trail behind the head).
fn trailTangent(pNew: vec3f, pOld: vec3f) -> vec3f {
  let t = pOld - pNew;
  let len = length(t);
  if (len < 1e-6) { return vec3f(0.0); }
  return t / len;
}

// Camera-facing ribbon side: width axis perpendicular to both view and flow tangent.
fn ribbonSide(tangent: vec3f, world: vec3f) -> vec3f {
  let view = normalize(u.ro - world);
  var side = cross(view, tangent);
  let len = length(side);
  if (len < 1e-5) {
    var axis = vec3f(0.0, 0.0, 1.0);
    if (abs(dot(tangent, axis)) > 0.92) { axis = vec3f(0.0, 1.0, 0.0); }
    side = cross(tangent, axis);
  }
  return normalize(side);
}

fn ribbonHalfWidthWorld(world: vec3f, widthMix: f32) -> f32 {
  let dist = max(length(world - u.ro), u.half * 0.06);
  let halfWPx = max(0.5 * u.trailWidth * u.half / dist * widthMix, 0.25);
  return halfWPx * dist * u.ribbonPxToWorld;
}

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) color: vec4f,
  @location(2) uv: vec2f,
}

const SEG_QUAD: array<vec2f, 6> = array<vec2f, 6>(
  vec2f(0.0, -1.0), vec2f(0.0, 1.0), vec2f(1.0, -1.0),
  vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0),
);

@vertex
fn vsMain(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) pIdx: u32,
) -> VSOut {
  var o: VSOut;
  o.clip = vec4f(0.0, 0.0, -2.0, 1.0);
  o.world = vec3f(0.0);
  o.color = vec4f(0.0);
  o.uv = vec2f(0.0);

  if (pIdx >= u.particleCount) { return o; }

  let particleIdx = sortOrder[pIdx];
  if (particleIdx >= u.particleCount) { return o; }

  let segCount = max(u.trailSegCount, 1u);
  let seg = vi / 6u;
  if (seg >= segCount) { return o; }

  let stride = max(u.segStride, 1u);
  let slotNew = seg * stride;
  let slotOld = slotNew + stride;
  if (slotOld >= u.trailSteps) { return o; }

  let pNew = trailPosAge(particleIdx, slotNew);
  let pOld = trailPosAge(particleIdx, slotOld);
  let segLen = length(pNew.xyz - pOld.xyz);
  var segFade = 1.0;
  if (segLen > u.maxSegLen * 0.75) {
    segFade = 1.0 - smoothstep(u.maxSegLen * 0.85, u.maxSegLen * 1.05, segLen);
  }
  if (segFade < 1e-4 && slotNew != 0u) { return o; }
  if (segLen < 1e-5 && slotNew != 0u) { return o; }

  let corner = SEG_QUAD[vi % 6u];
  let tAlong = corner.x;
  let tSide = corner.y;

  let uNew = ribbonU(slotNew);
  let uOld = ribbonU(slotOld);
  var ribbonPos = mix(uNew, uOld, tAlong);

  let wNew = ribbonWidthEnvelope(uNew);
  let wOld = ribbonWidthEnvelope(uOld);
  var widthMix = mix(wNew, wOld, tAlong);

  var world = mix(pNew.xyz, pOld.xyz, tAlong);
  var tangent = trailTangent(pNew.xyz, pOld.xyz);
  if (length(tangent) < 1e-6) {
    if (slotNew != 0u) { return o; }
    tangent = normalize(vec3f(u.ro - pNew.xyz));
  }

  // Rounded leading cap: extend slightly forward of the head along motion.
  if (slotNew == 0u) {
    let capFrac = 0.22;
    if (tAlong < capFrac) {
      let capT = tAlong / capFrac;
      let capLen = max(segLen * 0.45, u.half * 0.004);
      world = pNew.xyz - tangent * capLen * (1.0 - capT);
      widthMix = sin(1.5707963 * capT) * max(wNew, 0.35);
      ribbonPos = capT * uNew;
    }
  }

  let side = ribbonSide(tangent, world);
  let halfW = ribbonHalfWidthWorld(world, widthMix);
  let worldOffset = world + side * (tSide * halfW);
  let clipPos = u.viewProj * vec4f(worldOffset, 1.0);

  let flowIdx = particleLayers[particleIdx];
  let spdNew = trailSpeed(particleIdx, slotNew);
  let spdOld = trailSpeed(particleIdx, slotOld);
  let spd = mix(spdNew, spdOld, tAlong);
  let rgb = speedColor(spd, flowIdx);
  let lifeAge = mix(pNew.w, pOld.w, tAlong);
  let lifeFade = trailLifeAlpha(lifeAge);
  let alpha = u.flowOpacity * max(ribbonAlphaEnvelope(ribbonPos), widthMix * 0.15) * boxClipAlpha(world) * lifeFade * segFade;

  o.clip = clipPos;
  o.world = worldOffset;
  o.color = vec4f(rgb * alpha, alpha);
  o.uv = vec2f(tSide, ribbonPos);
  return o;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  if (in.color.a < 1e-6) { discard; }

  // Clip any ribbon surface that lies outside the fit box.
  let half = u.half;
  let m = max(max(abs(in.world.x), abs(in.world.y)), abs(in.world.z));
  if (m > half * 1.012) { discard; }

  let edge = abs(in.uv.x);
  if (edge > 1.0) { discard; }
  let widthSoft = 1.0 - smoothstep(0.55, 1.0, edge);

  let fbW = u.fbW; let fbH = u.fbH;
  let ndcX = -1.0 + 2.0 * in.clip.x / fbW;
  let ndcY = 1.0 - 2.0 * in.clip.y / fbH;
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(u.m0.xyz, xy1), dot(u.m1.xyz, xy1), dot(u.m2.xyz, xy1));
  let ro = u.ro;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB); let tmax = max(tA, tB);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  let far = max(tExit, half * 4.0);
  let rd2 = max(dot(rd, rd), 1e-20);
  let t = dot(in.world - ro, rd) / rd2;
  if (!(t > 0.0)) { discard; }
  let myD = clamp(t / far, 0.0, 0.999);
  let dims = vec2f(textureDimensions(occlTex));
  let mx = i32(clamp(floor(in.clip.x * dims.x / fbW), 0.0, dims.x - 1.0));
  let my = i32(clamp(floor(in.clip.y * dims.y / fbH), 0.0, dims.y - 1.0));
  let isoD = textureLoad(occlTex, vec2i(mx, my), 0).r;
  if (myD >= isoD - 1e-4) { discard; }

  // Soften alpha right at the clip plane to reduce stair-steps.
  let boxEdge = 1.0 - smoothstep(half, half * 1.012, m);
  let a = in.color.a * widthSoft * boxEdge;
  return vec4f(in.color.rgb * widthSoft * boxEdge, a);
}
