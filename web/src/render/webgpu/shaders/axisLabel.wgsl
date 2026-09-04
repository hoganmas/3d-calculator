struct GridParams {
  viewProj: mat4x4f,
  ro: vec3f,
  half: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  fbW: f32,
  fbH: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: GridParams;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlasSamp: sampler;
@group(0) @binding(3) var occlTex: texture_2d<f32>;

struct VSIn {
  @location(0) pos: vec3f,
  @location(1) uv: vec2f,
}

struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) uv: vec2f,
}

@vertex
fn vsMain(v: VSIn) -> VSOut {
  var o: VSOut;
  o.clip = u.viewProj * vec4f(v.pos, 1.0);
  o.world = v.pos;
  o.uv = v.uv;
  return o;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let tex = textureSample(atlas, atlasSamp, in.uv);
  if (tex.a < 0.02) { discard; }
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
  let tEnter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  // Must match isoHermite.wgsl's normalization exactly — this is compared
  // directly against occl.r below.
  let far = max(tExit - tEnter, half * 4.0);
  let rd2 = max(dot(rd, rd), 1e-20);
  let t = dot(in.world - ro, rd) / rd2;
  if (!(t > 0.0)) { discard; }
  let myD = clamp((t - tEnter) / far, 0.0, 0.999);
  let dims = vec2f(textureDimensions(occlTex));
  let mx = i32(clamp(floor(in.clip.x * dims.x / fbW), 0.0, dims.x - 1.0));
  let my = i32(clamp(floor(in.clip.y * dims.y / fbH), 0.0, dims.y - 1.0));
  let isoD = textureLoad(occlTex, vec2i(mx, my), 0).r;
  if (myD >= isoD - 1e-4) { discard; }
  return vec4f(tex.rgb * tex.a, tex.a);
}
