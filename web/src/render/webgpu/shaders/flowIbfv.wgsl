struct FlowIbfvParams {
  flowGridM: u32,
  velGridM: u32,
  half: f32,
  alpha: f32,
  gridSpacing: f32,
  dt: f32,
  vMax: f32,
  frameIdx: u32,
  velBase: u32,
  dyeLayerOff: u32,
  gridPoints: f32,
}

@group(0) @binding(0) var<uniform> params: FlowIbfvParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var<storage, read> dyeIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> dyeOut: array<f32>;

// ch0 = dye density; ch1 = age (seconds since injection, advected).
const DYE_CH: u32 = 2u;
const CH_TOTAL: u32 = 0u;
const CH_AGE: u32 = 1u;

fn chebIndex(xi: f32) -> f32 {
  let x = clamp(xi, -1.0, 1.0);
  return f32(params.velGridM) / 3.141592653589793 * acos(x) - 0.5;
}

fn gridLineDist(coord: f32, spacing: f32) -> f32 {
  let f = fract(coord / spacing);
  return min(f, 1.0 - f) * spacing;
}

fn backgroundGridlines(p: vec3f) -> f32 {
  let s = max(params.gridSpacing, 1e-4);
  let w = s * 0.22;
  let dx = gridLineDist(p.x, s);
  let dy = gridLineDist(p.y, s);
  let dz = gridLineDist(p.z, s);
  let lx = 1.0 - smoothstep(0.0, w, dx);
  let ly = 1.0 - smoothstep(0.0, w, dy);
  let lz = 1.0 - smoothstep(0.0, w, dz);
  return max(lx, max(ly, lz));
}

fn backgroundGridPoints(p: vec3f) -> f32 {
  let s = max(params.gridSpacing, 1e-4);
  let w = s * 0.22;
  let dx = gridLineDist(p.x, s);
  let dy = gridLineDist(p.y, s);
  let dz = gridLineDist(p.z, s);
  let lx = 1.0 - smoothstep(0.0, w, dx);
  let ly = 1.0 - smoothstep(0.0, w, dy);
  let lz = 1.0 - smoothstep(0.0, w, dz);
  return lx * ly * lz;
}

fn backgroundGrid(p: vec3f) -> f32 {
  if (params.gridPoints > 0.5) {
    return backgroundGridPoints(p);
  }
  return backgroundGridlines(p);
}

fn densAtBase(base: u32, ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(params.velGridM);
  let x = clamp(ix, 0, M - 1);
  let y = clamp(iy, 0, M - 1);
  let z = clamp(iz, 0, M - 1);
  let M2 = params.velGridM * params.velGridM;
  return volume[base + u32(x) + u32(y) * params.velGridM + u32(z) * M2];
}

fn sampleVelLayer(base: u32, p: vec3f) -> vec3f {
  let half = params.half;
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let volN = params.velGridM * params.velGridM * params.velGridM;
  var v = vec3f(0.0);
  for (var c: u32 = 0u; c < 3u; c++) {
    let compBase = base + c * volN;
    let fx = chebIndex(xi.x);
    let fy = chebIndex(xi.y);
    let fz = chebIndex(xi.z);
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let z0 = i32(floor(fz));
    let tx = clamp(fx - f32(x0), 0.0, 1.0);
    let ty = clamp(fy - f32(y0), 0.0, 1.0);
    let tz = clamp(fz - f32(z0), 0.0, 1.0);
    let c000 = densAtBase(compBase, x0, y0, z0);
    let c100 = densAtBase(compBase, x0 + 1, y0, z0);
    let c010 = densAtBase(compBase, x0, y0 + 1, z0);
    let c110 = densAtBase(compBase, x0 + 1, y0 + 1, z0);
    let c001 = densAtBase(compBase, x0, y0, z0 + 1);
    let c101 = densAtBase(compBase, x0 + 1, y0, z0 + 1);
    let c011 = densAtBase(compBase, x0, y0 + 1, z0 + 1);
    let c111 = densAtBase(compBase, x0 + 1, y0 + 1, z0 + 1);
    v[c] = mix(mix(mix(c000, c100, tx), mix(c010, c110, tx), ty),
               mix(mix(c001, c101, tx), mix(c011, c111, tx), ty), tz);
  }
  return v;
}

fn dyeIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  let M = params.flowGridM;
  return ix + iy * M + iz * M * M;
}

fn worldToDyeFrac(p: vec3f) -> vec3f {
  let half = params.half;
  return (p + vec3f(half)) / (2.0 * half);
}

fn sampleDyeChannel(p: vec3f, layerOff: u32, ch: u32) -> f32 {
  let half = params.half;
  if (abs(p.x) > half || abs(p.y) > half || abs(p.z) > half) {
    return 0.0;
  }
  let M = i32(params.flowGridM);
  let f = worldToDyeFrac(p) * f32(M) - 0.5;
  let x0 = i32(floor(f.x));
  let y0 = i32(floor(f.y));
  let z0 = i32(floor(f.z));
  let tx = clamp(f.x - f32(x0), 0.0, 1.0);
  let ty = clamp(f.y - f32(y0), 0.0, 1.0);
  let tz = clamp(f.z - f32(z0), 0.0, 1.0);
  let vi = dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0, 0, M - 1)));
  let base = layerOff + vi * DYE_CH + ch;
  let c000 = dyeIn[base];
  let c100 = dyeIn[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0, 0, M - 1))) * DYE_CH + ch];
  let c010 = dyeIn[layerOff + dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0, 0, M - 1))) * DYE_CH + ch];
  let c110 = dyeIn[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0, 0, M - 1))) * DYE_CH + ch];
  let c001 = dyeIn[layerOff + dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * DYE_CH + ch];
  let c101 = dyeIn[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * DYE_CH + ch];
  let c011 = dyeIn[layerOff + dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * DYE_CH + ch];
  let c111 = dyeIn[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * DYE_CH + ch];
  return mix(mix(mix(c000, c100, tx), mix(c010, c110, tx), ty),
             mix(mix(c001, c101, tx), mix(c011, c111, tx), ty), tz);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let M = params.flowGridM;
  if (gid.x >= M || gid.y >= M || gid.z >= M) {
    return;
  }
  let half = params.half;
  let p = (vec3f(gid) + 0.5) * (2.0 * half / f32(M)) - vec3f(half);
  let V = sampleVelLayer(params.velBase, p);
  let speed = length(V);
  var v = V;
  if (speed > params.vMax && params.vMax > 1e-8) {
    v = V * (params.vMax / speed);
  }
  let pPrev = p - v * params.dt;
  let vi = dyeIndex(gid.x, gid.y, gid.z);
  let outBase = params.dyeLayerOff + vi * DYE_CH;
  if (speed <= 1e-5) {
    dyeOut[outBase + CH_TOTAL] = 0.0;
    dyeOut[outBase + CH_AGE] = 0.0;
    return;
  }
  let totalPrev = sampleDyeChannel(pPrev, params.dyeLayerOff, CH_TOTAL);
  let agePrev = sampleDyeChannel(pPrev, params.dyeLayerOff, CH_AGE) + params.dt;
  var G = 0.0;
  if (params.alpha > 1e-6) {
    G = backgroundGrid(pPrev);
  }
  let totalNew = (1.0 - params.alpha) * totalPrev + params.alpha * G;
  var ageNew = 0.0;
  if (totalNew > 1e-6) {
    ageNew = (1.0 - params.alpha) * totalPrev * agePrev / totalNew;
  }
  dyeOut[outBase + CH_TOTAL] = totalNew;
  dyeOut[outBase + CH_AGE] = ageNew;
}
