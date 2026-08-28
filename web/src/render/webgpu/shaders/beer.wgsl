struct DrawParams {
  fbW: u32,
  fbH: u32,
  gridM: u32,
  steps: u32,
  half: f32,
  scale: f32,
  densBase: f32,
  layerCount: u32,
  ro: vec3f,
  _p1: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  flowLayerStart: f32,
  flowVelBase: f32,
  flowGridM: u32,
  flowOpacity: f32,
  flowAlpha: f32,
  flowVRef: f32,
  flowAgeMax: f32,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var occlIsoTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> layerGrads: array<vec4f>;
@group(0) @binding(4) var<storage, read> flowDye: array<f32>;

{{GRADIENT_WGSL}}

fn sampleLayerGradAt(L: u32, t: f32) -> vec3f {
  let base = L * MAX_GRAD_STOPS;
  var stops: array<vec4f, {{MAX_GRAD_STOPS}}>;
  for (var i: u32 = 0u; i < MAX_GRAD_STOPS; i++) {
    stops[i] = layerGrads[base + i];
  }
  return sampleGradStops(&stops, t);
}

fn sampleLayerGrad(L: u32, t: f32) -> vec3f {
  let base = L * MAX_GRAD_STOPS;
  var stops: array<vec4f, {{MAX_GRAD_STOPS}}>;
  for (var i: u32 = 0u; i < MAX_GRAD_STOPS; i++) {
    stops[i] = layerGrads[base + i];
  }
  return sampleGradStops(&stops, t);
}

struct VSOut { @builtin(position) pos: vec4f, }

struct FSOut {
  @location(0) color: vec4f,
  @location(1) occl: vec4f,
}

const OCCL_ALPHA: f32 = 0.15;

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}

fn densAtBase(base: u32, ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(draw.gridM);
  let x = clamp(ix, 0, M - 1); let y = clamp(iy, 0, M - 1); let z = clamp(iz, 0, M - 1);
  return volume[base + u32(x) + u32(y) * draw.gridM + u32(z) * draw.gridM * draw.gridM];
}
fn chebIndex(xi: f32) -> f32 {
  let x = clamp(xi, -1.0, 1.0);
  return f32(draw.gridM) / 3.141592653589793 * acos(x) - 0.5;
}
fn sampleLayer(base: u32, p: vec3f) -> f32 {
  let half = draw.half;
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let fx = chebIndex(xi.x); let fy = chebIndex(xi.y); let fz = chebIndex(xi.z);
  let x0 = i32(floor(fx)); let y0 = i32(floor(fy)); let z0 = i32(floor(fz));
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let tz = clamp(fz - f32(z0), 0.0, 1.0);
  let c000 = densAtBase(base, x0, y0, z0); let c100 = densAtBase(base, x0 + 1, y0, z0);
  let c010 = densAtBase(base, x0, y0 + 1, z0); let c110 = densAtBase(base, x0 + 1, y0 + 1, z0);
  let c001 = densAtBase(base, x0, y0, z0 + 1); let c101 = densAtBase(base, x0 + 1, y0, z0 + 1);
  let c011 = densAtBase(base, x0, y0 + 1, z0 + 1); let c111 = densAtBase(base, x0 + 1, y0 + 1, z0 + 1);
  return mix(mix(mix(c000, c100, tx), mix(c010, c110, tx), ty),
              mix(mix(c001, c101, tx), mix(c011, c111, tx), ty), tz);
}

fn dyeIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  let M = draw.flowGridM;
  return ix + iy * M + iz * M * M;
}

fn sampleFlowDyePair(flowIdx: u32, p: vec3f) -> vec2f {
  let half = draw.half;
  if (abs(p.x) > half || abs(p.y) > half || abs(p.z) > half) {
    return vec2f(0.0);
  }
  let M = i32(draw.flowGridM);
  if (M <= 0) { return vec2f(0.0); }
  let layerVolN = draw.flowGridM * draw.flowGridM * draw.flowGridM;
  let layerOff = flowIdx * layerVolN * 2u;
  let f = (p + vec3f(half)) / (2.0 * half) * f32(M) - 0.5;
  let x0 = i32(floor(f.x));
  let y0 = i32(floor(f.y));
  let z0 = i32(floor(f.z));
  let tx = clamp(f.x - f32(x0), 0.0, 1.0);
  let ty = clamp(f.y - f32(y0), 0.0, 1.0);
  let tz = clamp(f.z - f32(z0), 0.0, 1.0);
  var out = vec2f(0.0);
  for (var ch: u32 = 0u; ch < 2u; ch++) {
    let c000 = flowDye[layerOff + dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0, 0, M - 1))) * 2u + ch];
    let c100 = flowDye[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0, 0, M - 1))) * 2u + ch];
    let c010 = flowDye[layerOff + dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0, 0, M - 1))) * 2u + ch];
    let c110 = flowDye[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0, 0, M - 1))) * 2u + ch];
    let c001 = flowDye[layerOff + dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * 2u + ch];
    let c101 = flowDye[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * 2u + ch];
    let c011 = flowDye[layerOff + dyeIndex(u32(clamp(x0, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * 2u + ch];
    let c111 = flowDye[layerOff + dyeIndex(u32(clamp(x0 + 1, 0, M - 1)), u32(clamp(y0 + 1, 0, M - 1)), u32(clamp(z0 + 1, 0, M - 1))) * 2u + ch];
    out[ch] = mix(mix(mix(c000, c100, tx), mix(c010, c110, tx), ty),
                  mix(mix(c001, c101, tx), mix(c011, c111, tx), ty), tz);
  }
  return out;
}

fn sampleVelLayer(velBase: u32, p: vec3f) -> vec3f {
  let half = draw.half;
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let M = draw.gridM;
  let M2 = M * M;
  let volN = M2 * M;
  var v = vec3f(0.0);
  for (var c: u32 = 0u; c < 3u; c++) {
    let compBase = velBase + c * volN;
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

@fragment
fn fsMain(in: VSOut) -> FSOut {
  var out: FSOut;
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);

  let fbW = f32(draw.fbW); let fbH = f32(draw.fbH);
  let ndcX = -1.0 + 2.0 * in.pos.x / fbW;
  let ndcY = 1.0 - 2.0 * in.pos.y / fbH;
  let xy1 = vec3f(ndcX, ndcY, 1.0);
  let rd = vec3f(dot(draw.m0.xyz, xy1), dot(draw.m1.xyz, xy1), dot(draw.m2.xyz, xy1));
  let ro = draw.ro; let half = draw.half;
  let invRd = vec3f(
    select(1e15, 1.0 / rd.x, abs(rd.x) >= 1e-15),
    select(1e15, 1.0 / rd.y, abs(rd.y) >= 1e-15),
    select(1e15, 1.0 / rd.z, abs(rd.z) >= 1e-15),
  );
  let tA = (-vec3f(half) - ro) * invRd;
  let tB = (vec3f(half) - ro) * invRd;
  let tmin = min(tA, tB); let tmax = max(tA, tB);
  var tEnter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  var tExit = min(min(tmax.x, tmax.y), tmax.z);
  if (!(tExit > tEnter + 1e-6)) {
    out.color = vec4f(0.0);
    return out;
  }

  let far = max(tExit, half * 4.0);
  let px = u32(clamp(floor(in.pos.x), 0.0, fbW - 1.0));
  let py = u32(clamp(floor(in.pos.y), 0.0, fbH - 1.0));
  let isoD = textureLoad(occlIsoTex, vec2u(px, py), 0).r;
  if (isoD < 0.999) { tExit = min(tExit, isoD * far); }
  if (!(tExit > tEnter + 1e-6)) {
    out.color = vec4f(0.0);
    out.occl = vec4f(isoD, 0.0, 0.0, 1.0);
    return out;
  }

  var steps = draw.steps;
  if (steps < 8u) { steps = 8u; }
  if (steps > 96u) { steps = 96u; }
  let dt = (tExit - tEnter) / f32(steps);
  let ds = length(rd) * dt;
  let volN = draw.gridM * draw.gridM * draw.gridM;
  let densBase = u32(draw.densBase);
  let nLay = min(draw.layerCount, {{MAX_DENS_LAYERS}}u);

  var rgb = vec3f(0.0); var T = 1.0; var s = tEnter + 0.5 * dt;
  var densD = 1.0;
  for (var i: u32 = 0u; i < 96u; i++) {
    if (i >= steps) { break; }
    if (T < 0.002) { break; }
    let p = ro + rd * s;
    var sigma = 0.0; var emitAcc = vec3f(0.0);
    for (var L: u32 = 0u; L < {{MAX_DENS_LAYERS}}u; L++) {
      if (L >= nLay) { break; }
      var dval = sampleLayer(densBase + L * volN, p);
      if (dval != dval) { dval = 0.0; }
      var col: vec3f;
      let isFlow = f32(L) >= draw.flowLayerStart && draw.flowLayerStart >= 0.0;
      if (isFlow) {
        let flowIdx = L - u32(draw.flowLayerStart);
        let presence = dval;
        let velBase = u32(draw.flowVelBase) + flowIdx * volN * 3u;
        let V = sampleVelLayer(velBase, p);
        let speedNorm = clamp(length(V) / max(draw.flowVRef, 1e-6), 0.0, 1.0);
        let dyePair = sampleFlowDyePair(flowIdx, p);
        let totalAmt = dyePair.x;
        let age = dyePair.y;
        dval = draw.flowOpacity * presence * speedNorm * totalAmt;
        let tAge = clamp(age / max(draw.flowAgeMax, 1e-4), 0.0, 1.0);
        let col1 = sampleLayerGradAt(L, 0.0);
        let col2 = sampleLayerGradAt(L, 1.0);
        col = mix(col1, col2, tAge);
      } else {
        let gt = gradientT(p, half);
        col = sampleLayerGrad(L, gt);
      }
      dval = clamp(dval, -4.0, 8.0);
      let sig = min(max(0.0, draw.scale * dval), 40.0);
      if (sig > 1e-8) {
        sigma += sig;
        emitAcc += col * sig;
      }
    }
    if (sigma > 1e-8) {
      let absorb = exp(-sigma * ds);
      let opacity = 1.0 - absorb;
      let col = emitAcc / sigma;
      // Beer emission + soft ambient so wispy low-density regions stay luminous.
      rgb += T * opacity * col * (1.0 + 0.42);
      T *= absorb;
      let alpha = 1.0 - T;
      if (densD >= 0.999 && alpha >= OCCL_ALPHA) {
        densD = clamp(s / far, 0.0, 0.999);
      }
    }
    s += dt;
  }
  let a = 1.0 - T;
  out.occl = vec4f(min(isoD, densD), 0.0, 0.0, 1.0);
  if (a < 0.001) {
    out.color = vec4f(0.0);
    return out;
  }
  out.color = vec4f(rgb, a);
  return out;
}
