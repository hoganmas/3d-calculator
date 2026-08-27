struct FxaaParams {
  invRes: vec2f,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(0) var<uniform> u: FxaaParams;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var srcSamp: sampler;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  // WebGPU: texel (0,0) is top-left; clip +Y is up — flip V so the overlay matches Three.
  o.uv = vec2f(p[vi].x * 0.5 + 0.5, -p[vi].y * 0.5 + 0.5);
  return o;
}

fn luma(rgb: vec3f) -> f32 {
  return dot(rgb, vec3f(0.299, 0.587, 0.114));
}

fn samplePremul(uv: vec2f) -> vec4f {
  return textureSampleLevel(srcTex, srcSamp, uv, 0.0);
}

fn unpremul(c: vec4f) -> vec3f {
  return c.xyz / max(c.a, 1e-4);
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let rcp = u.invRes;
  let uv = in.uv;

  let rgbM = samplePremul(uv);
  // Skip empty pixels — keep overlay holes crisp for Three underneath.
  if (rgbM.a < 0.001) { return vec4f(0.0); }

  let rgbNW = samplePremul(uv + vec2f(-rcp.x, -rcp.y));
  let rgbNE = samplePremul(uv + vec2f( rcp.x, -rcp.y));
  let rgbSW = samplePremul(uv + vec2f(-rcp.x,  rcp.y));
  let rgbSE = samplePremul(uv + vec2f( rcp.x,  rcp.y));

  // Edge detect on straight alpha + luma (silhouettes are mostly alpha edges).
  let lM = luma(unpremul(rgbM)) * rgbM.a + rgbM.a;
  let lNW = luma(unpremul(rgbNW)) * rgbNW.a + rgbNW.a;
  let lNE = luma(unpremul(rgbNE)) * rgbNE.a + rgbNE.a;
  let lSW = luma(unpremul(rgbSW)) * rgbSW.a + rgbSW.a;
  let lSE = luma(unpremul(rgbSE)) * rgbSE.a + rgbSE.a;

  let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  let range = lMax - lMin;
  // No edge → passthrough.
  if (range < max(0.0312, lMax * 0.125)) {
    return rgbM;
  }

  var dir = vec2f(
    -((lNW + lNE) - (lSW + lSE)),
    ((lNW + lSW) - (lNE + lSE)),
  );
  let dirReduce = max((lNW + lNE + lSW + lSE) * 0.03125, 0.0078125);
  let rcpDir = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDir, vec2f(-8.0), vec2f(8.0)) * rcp;

  let rgbA = 0.5 * (
    samplePremul(uv + dir * (1.0 / 3.0 - 0.5)) +
    samplePremul(uv + dir * (2.0 / 3.0 - 0.5))
  );
  let rgbB = rgbA * 0.5 + 0.25 * (
    samplePremul(uv + dir * -0.5) +
    samplePremul(uv + dir * 0.5)
  );

  let lB = luma(unpremul(rgbB)) * rgbB.a + rgbB.a;
  if (lB < lMin || lB > lMax) {
    return rgbA;
  }
  return rgbB;
}
