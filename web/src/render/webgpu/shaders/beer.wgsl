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
  _p2: f32,
  _p3: f32,
  _p4: f32,
  densBlend: array<vec4f, 8>,
  densT: array<vec4f, 2>,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;
@group(0) @binding(2) var occlIsoTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> layerGrads: array<vec4f>;

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

fn densBlendT(L: u32) -> f32 {
  let v = draw.densT[select(0u, 1u, L >= 4u)];
  let i = L % 4u;
  if (i == 0u) { return v.x; }
  if (i == 1u) { return v.y; }
  if (i == 2u) { return v.z; }
  return v.w;
}

fn isoOcclusionForVolumePixel(pos: vec2f, fbW: f32, fbH: f32) -> f32 {
  let isoDims = textureDimensions(occlIsoTex);
  let isoW = f32(isoDims.x);
  let isoH = f32(isoDims.y);
  let ux = clamp(pos.x / fbW, 0.0, 1.0);
  let uy = clamp(pos.y / fbH, 0.0, 1.0);
  if (isoW <= fbW + 0.5 && isoH <= fbH + 0.5) {
    // Same-res compose, or coarser occupancy (16× interiors / 4× mid beer).
    let ix = u32(min(floor(ux * isoW), isoW - 1.0));
    let iy = u32(min(floor(uy * isoH), isoH - 1.0));
    return textureLoad(occlIsoTex, vec2u(ix, iy), 0).r;
  }
  // Clip tex finer than beer: clip only fully-covered interior tiles (min depth).
  // Mixed footprints stay unclipped and remarch at compose res.
  let x0 = i32(clamp(floor(pos.x / fbW * isoW), 0.0, isoW - 1.0));
  let y0 = i32(clamp(floor(pos.y / fbH * isoH), 0.0, isoH - 1.0));
  var x1 = i32(clamp(ceil((pos.x + 1.0) / fbW * isoW) - 1.0, 0.0, isoW - 1.0));
  var y1 = i32(clamp(ceil((pos.y + 1.0) / fbH * isoH) - 1.0, 0.0, isoH - 1.0));
  x1 = max(x0, x1);
  y1 = max(y0, y1);
  var dMin = 1.0;
  var dMax = 0.0;
  let spanX = min(x1 - x0, 7);
  let spanY = min(y1 - y0, 7);
  for (var iy = 0; iy < 8; iy++) {
    if (iy > spanY) { break; }
    for (var ix = 0; ix < 8; ix++) {
      if (ix > spanX) { break; }
      let d = textureLoad(occlIsoTex, vec2u(u32(x0 + ix), u32(y0 + iy)), 0).r;
      dMin = min(dMin, d);
      dMax = max(dMax, d);
    }
  }
  if (dMin >= 0.999 || dMax >= 0.999) { return 1.0; }
  return dMin;
}

fn marchBeer(pos: vec2f) -> FSOut {
  var out: FSOut;
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);

  let fbW = f32(draw.fbW); let fbH = f32(draw.fbH);
  let ndcX = -1.0 + 2.0 * pos.x / fbW;
  let ndcY = 1.0 - 2.0 * pos.y / fbH;
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
  // Same-res: clip beer to iso. Finer iso: leave tExit alone (isoD == 1).
  let isoD = isoOcclusionForVolumePixel(pos, fbW, fbH);
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
      var dval = 0.0;
      let packed = draw.densBlend[L];
      let stride = u32(packed.y);
      if (stride > 0u) {
        let base0 = u32(packed.x) + u32(packed.z) * stride;
        let bt = densBlendT(L);
        if (bt <= 1e-5 || packed.z == packed.w) {
          dval = sampleLayer(base0, p);
        } else if (bt >= 0.999) {
          dval = sampleLayer(u32(packed.x) + u32(packed.w) * stride, p);
        } else {
          let base1 = u32(packed.x) + u32(packed.w) * stride;
          dval = mix(sampleLayer(base0, p), sampleLayer(base1, p), bt);
        }
      } else {
        dval = sampleLayer(densBase + L * volN, p);
      }
      if (dval != dval) { dval = 0.0; }
      var col: vec3f;
      let isFlow = f32(L) >= draw.flowLayerStart && draw.flowLayerStart >= 0.0;
      if (isFlow) { continue; }
      let gt = gradientT(p, half);
      col = sampleLayerGrad(L, gt);
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
  // Deferred iso clip stores volume depth only; same-res still mins with isoD.
  out.occl = vec4f(min(isoD, densD), 0.0, 0.0, 1.0);
  if (a < 0.001) {
    out.color = vec4f(0.0);
    return out;
  }
  out.color = vec4f(rgb, a);
  return out;
}

@fragment
fn fsMain(in: VSOut) -> FSOut {
  return marchBeer(in.pos.xy);
}
