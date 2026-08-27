struct SsaoParams {
  fbW: u32,
  fbH: u32,
  _pad0: u32,
  _pad1: u32,
  half: f32,
  radius: f32,
  strength: f32,
  bias: f32,
  ro: vec3f,
  _p1: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
}

@group(0) @binding(0) var<uniform> p: SsaoParams;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var occlTex: texture_2d<f32>;
@group(0) @binding(3) var normalTex: texture_2d<f32>;

struct VSOut { @builtin(position) pos: vec4f, }

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var pts = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(pts[vi], 0.0, 1.0);
  return o;
}

fn rayDir(ndcX: f32, ndcY: f32) -> vec3f {
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  return vec3f(dot(p.m0.xyz, xy1), dot(p.m1.xyz, xy1), dot(p.m2.xyz, xy1));
}

fn boxFar(ro: vec3f, rd: vec3f) -> f32 {
  let half = p.half;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmax = max(tA, tB);
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  return max(tExit, half * 4.0);
}

fn hash2(pxy: vec2f) -> f32 {
  return fract(sin(dot(pxy, vec2f(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let fbW = f32(p.fbW);
  let fbH = f32(p.fbH);
  let px = u32(clamp(floor(in.pos.x), 0.0, fbW - 1.0));
  let py = u32(clamp(floor(in.pos.y), 0.0, fbH - 1.0));
  let uv = vec2u(px, py);

  let color = textureLoad(sceneTex, uv, 0);
  let depthN = textureLoad(occlTex, uv, 0).r;
  let nEnc = textureLoad(normalTex, uv, 0);
  if (depthN >= 0.999 || nEnc.a < 0.5) {
    return color;
  }

  let ndcX = -1.0 + 2.0 * (f32(px) + 0.5) / fbW;
  let ndcY = 1.0 - 2.0 * (f32(py) + 0.5) / fbH;
  let rd = rayDir(ndcX, ndcY);
  let ro = p.ro;
  let far = boxFar(ro, rd);
  let t0 = depthN * far;
  let pos0 = ro + rd * t0;
  let n = normalize(nEnc.xyz * 2.0 - 1.0);

  // Screen-space radius shrinks with distance (approx).
  let radPx = clamp(p.radius * 0.5 * min(fbW, fbH) / max(t0, 0.2), 2.0, 28.0);
  let rot = hash2(vec2f(f32(px), f32(py))) * 6.2831853;
  let cR = cos(rot);
  let sR = sin(rot);

  var occ = 0.0;
  var wSum = 0.0;
  for (var i: u32 = 0u; i < 16u; i++) {
    let fi = f32(i);
    let ang = fi * 2.3999632 + rot; // golden angle
    let r = sqrt((fi + 0.5) / 16.0) * radPx;
    let off = vec2f(cos(ang) * r, sin(ang) * r);
    // rotate slightly with noise
    let ox = off.x * cR - off.y * sR;
    let oy = off.x * sR + off.y * cR;

    let sx = i32(px) + i32(round(ox));
    let sy = i32(py) + i32(round(oy));
    if (sx < 0 || sy < 0 || sx >= i32(p.fbW) || sy >= i32(p.fbH)) { continue; }
    let suv = vec2u(u32(sx), u32(sy));
    let dS = textureLoad(occlTex, suv, 0).r;
    if (dS >= 0.999) { continue; }

    let sndcX = -1.0 + 2.0 * (f32(sx) + 0.5) / fbW;
    let sndcY = 1.0 - 2.0 * (f32(sy) + 0.5) / fbH;
    let srd = rayDir(sndcX, sndcY);
    let sFar = boxFar(ro, srd);
    let tS = dS * sFar;
    let posS = ro + srd * tS;
    let v = posS - pos0;
    let dist = length(v);
    if (dist < 1e-5) { continue; }
    // Reject silhouette / box-edge samples (large ray-t discontinuity).
    if (abs(tS - t0) > p.radius * 2.5) { continue; }
    let vn = v / dist;
    // Hemisphere contribution + range falloff (Alchemy-ish).
    let nd = max(dot(n, vn) - p.bias, 0.0);
    let fall = 1.0 - smoothstep(0.0, p.radius, dist);
    let w = fall;
    occ += nd * fall;
    wSum += w;
  }

  var ao = 1.0;
  if (wSum > 1e-4) {
    ao = clamp(1.0 - (occ / wSum), 0.0, 1.0);
    // Mild contrast so creases read without crushing flats.
    ao = pow(ao, 1.25);
  }
  let factor = mix(1.0, ao, p.strength);
  // Occlusion: slide toward shadow gradient stop (c0-like), not grey darkening
  let tOcc = (1.0 - factor) * 0.55;
  let peak = max(max(color.r, color.g), color.b);
  let hue = color.rgb / max(peak, 1e-4);
  let shadowStop = hue * peak * 0.62;
  let rgb = mix(color.rgb, mix(shadowStop, color.rgb, factor), tOcc);
  return vec4f(rgb, color.a);
}
