const MAX_GRAD_STOPS: u32 = {{MAX_GRAD_STOPS}}u;

fn gradientT(p: vec3f, half: f32) -> f32 {
  let u = p / max(half, 1e-6);
  let ty = clamp(u.y * 0.5 + 0.5, 0.0, 1.0);
  let tx = clamp(u.x * 0.5 + 0.5, 0.0, 1.0);
  let tz = clamp(u.z * 0.5 + 0.5, 0.0, 1.0);
  let tr = clamp(length(u) * 0.70710678, 0.0, 1.0);
  return clamp(0.36 * ty + 0.24 * tx + 0.24 * tz + 0.16 * tr, 0.0, 1.0);
}

fn isoGradientT(p: vec3f, n: vec3f, half: f32) -> f32 {
  let tp = gradientT(p, half);
  let gdir = normalize(vec3f(0.12, 0.94, 0.32));
  let tn = clamp(0.5 + 0.5 * dot(n, gdir), 0.0, 1.0);
  let tb = clamp(0.5 + 0.35 * n.y, 0.0, 1.0);
  return clamp(mix(tp, tn, 0.58) * 0.82 + tb * 0.18, 0.0, 1.0);
}

/** Piecewise-linear sample across up to MAX_GRAD_STOPS colors. Count in stops[0].w */
fn sampleGradStops(stops: ptr<function, array<vec4f, {{MAX_GRAD_STOPS}}>>, t: f32) -> vec3f {
  let n = max(min(u32((*stops)[0].w), MAX_GRAD_STOPS), 1u);
  if (n <= 1u) { return (*stops)[0].xyz; }
  let x = clamp(t, 0.0, 1.0) * f32(n - 1u);
  let i = min(u32(floor(x)), n - 2u);
  let f = fract(x);
  return mix((*stops)[i].xyz, (*stops)[i + 1u].xyz, f);
}
