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
  _pad1: u32,
  trailWidth: f32,
  flowVRef: f32,
}

@group(0) @binding(0) var<uniform> u: FlowParticleParams;
@group(0) @binding(2) var<storage, read> particleLayers: array<u32>;
@group(0) @binding(3) var<storage, read> layerGrads: array<vec4f>;
@group(0) @binding(4) var occlTex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read> trailHist: array<f32>;

const MAX_GRAD_STOPS: u32 = {{MAX_GRAD_STOPS}}u;
const MAX_TRAIL_STEPS: u32 = {{MAX_FLOW_TRAIL_STEPS}}u;
const TRAIL_STRIDE: u32 = 5u;

fn sampleGradStopsLayer(layer: u32, t: f32) -> vec3f {
  let base = layer * MAX_GRAD_STOPS;
  let n = max(min(u32(layerGrads[base].w), MAX_GRAD_STOPS), 1u);
  if (n <= 1u) { return layerGrads[base].xyz; }
  let x = clamp(t, 0.0, 1.0) * f32(n - 1u);
  let i = min(u32(floor(x)), n - 2u);
  let f = fract(x);
  return mix(layerGrads[base + i].xyz, layerGrads[base + i + 1u].xyz, f);
}

fn speedColor(layer: u32, speed: f32) -> vec3f {
  let speedNorm = clamp(speed / max(u.flowVRef, 1e-6), 0.0, 1.0);
  let col1 = sampleGradStopsLayer(layer, 0.0);
  let col2 = sampleGradStopsLayer(layer, 1.0);
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

// slot 0 = newest (leading edge), slot N-1 = oldest (trailing edge); u=0 newest, u=1 oldest.
fn ribbonU(slot: u32) -> f32 {
  let n = max(u.trailSteps, 2u);
  return f32(slot) / f32(n - 1u);
}

fn ribbonWidthEnvelope(u: f32) -> f32 {
  return sin(3.14159265 * clamp(u, 0.0, 1.0));
}

fn ribbonAlphaEnvelope(u: f32) -> f32 {
  let width = ribbonWidthEnvelope(u);
  // Fade opacity toward the oldest end (u→1); keep newest end visible via width envelope only.
  let tailFade = 1.0 - smoothstep(0.62, 1.0, u);
  return width * tailFade;
}

fn clipToPixel(c: vec4f) -> vec2f {
  let iw = 1.0 / max(abs(c.w), 1e-6);
  return vec2f(
    (c.x * iw * 0.5 + 0.5) * u.fbW,
    (0.5 - c.y * iw * 0.5) * u.fbH,
  );
}

fn pixelToClip(px: vec2f, clipW: f32) -> vec2f {
  let ndc = vec2f(px.x / max(u.fbW, 1.0) * 2.0 - 1.0, 1.0 - px.y / max(u.fbH, 1.0) * 2.0);
  return ndc * clipW;
}

fn screenPerp(p0: vec3f, p1: vec3f) -> vec2f {
  let px0 = clipToPixel(u.viewProj * vec4f(p0, 1.0));
  let px1 = clipToPixel(u.viewProj * vec4f(p1, 1.0));
  let t = px1 - px0;
  let len = length(t);
  if (len < 1e-4) { return vec2f(0.0, 1.0); }
  let dir = t / len;
  return vec2f(-dir.y, dir.x);
}

fn distPxScale(world: vec3f) -> f32 {
  let dist = max(length(world - u.ro), u.half * 0.06);
  return (u.trailWidth * u.half) / dist;
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

  let segCount = max(u.trailSegCount, 1u);
  let seg = vi / 6u;
  if (seg >= segCount) { return o; }

  let slotNew = seg;
  let slotOld = seg + 1u;
  if (slotOld >= u.trailSteps) { return o; }

  let pNew = trailPosAge(pIdx, slotNew);
  let pOld = trailPosAge(pIdx, slotOld);
  if (length(pNew.xyz - pOld.xyz) < 1e-5) { return o; }

  let corner = SEG_QUAD[vi % 6u];
  let tAlong = corner.x;
  let tSide = corner.y;

  let uNew = ribbonU(slotNew);
  let uOld = ribbonU(slotOld);
  let ribbonPos = mix(uNew, uOld, tAlong);

  let wNew = ribbonWidthEnvelope(uNew);
  let wOld = ribbonWidthEnvelope(uOld);
  let widthMix = mix(wNew, wOld, tAlong);
  let halfWPx = max(0.5 * distPxScale(mix(pNew.xyz, pOld.xyz, tAlong)) * widthMix, 0.25);

  let world = mix(pNew.xyz, pOld.xyz, tAlong);
  let clipCenter = u.viewProj * vec4f(world, 1.0);

  let perp = screenPerp(pNew.xyz, pOld.xyz);
  let centerPx = clipToPixel(clipCenter);
  let offsetPx = centerPx + perp * (tSide * halfWPx);
  let clipOffset = pixelToClip(offsetPx, clipCenter.w);

  let layer = u32(u.flowLayerStart) + particleLayers[pIdx];
  let spd = mix(trailSpeed(pIdx, slotNew), trailSpeed(pIdx, slotOld), tAlong);
  let rgb = speedColor(layer, spd);
  let alpha = u.flowOpacity * max(ribbonAlphaEnvelope(ribbonPos), widthMix * 0.15) * clamp(length(rgb), 0.0, 1.0);

  o.clip = vec4f(clipOffset, clipCenter.z, clipCenter.w);
  o.world = world;
  o.color = vec4f(rgb * alpha, alpha);
  o.uv = vec2f(tSide, ribbonPos);
  return o;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  if (in.color.a < 1e-6) { discard; }

  let edge = abs(in.uv.x);
  if (edge > 1.0) { discard; }
  let widthSoft = 1.0 - smoothstep(0.55, 1.0, edge);

  let fbW = u.fbW; let fbH = u.fbH;
  let ndcX = -1.0 + 2.0 * in.clip.x / fbW;
  let ndcY = 1.0 - 2.0 * in.clip.y / fbH;
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(u.m0.xyz, xy1), dot(u.m1.xyz, xy1), dot(u.m2.xyz, xy1));
  let ro = u.ro; let half = u.half;
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

  return vec4f(in.color.rgb * widthSoft, in.color.a * widthSoft);
}
