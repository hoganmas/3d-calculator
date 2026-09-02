struct IsoUpParams {
  fineW: u32,
  fineH: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<uniform> u: IsoUpParams;
@group(0) @binding(1) var colorTex: texture_2d<f32>;
@group(0) @binding(2) var occlTex: texture_2d<f32>;
@group(0) @binding(3) var normalTex: texture_2d<f32>;

{{ISO_EDGE_WGSL}}

struct VSOut {
  @builtin(position) pos: vec4f,
}

struct FSOut {
  @location(0) color: vec4f,
  @location(1) occl: vec4f,
  @location(2) normal: vec4f,
  @builtin(frag_depth) depth: f32,
}

fn loadBilinearAt(tex: texture_2d<f32>, src: vec2f) -> vec4f {
  let dims = textureDimensions(tex);
  let x0 = i32(floor(src.x));
  let y0 = i32(floor(src.y));
  let tx = clamp(src.x - f32(x0), 0.0, 1.0);
  let ty = clamp(src.y - f32(y0), 0.0, 1.0);
  let maxX = i32(dims.x) - 1;
  let maxY = i32(dims.y) - 1;
  let t00 = textureLoad(tex, vec2u(u32(clamp(x0, 0, maxX)), u32(clamp(y0, 0, maxY))), 0);
  let t10 = textureLoad(tex, vec2u(u32(clamp(x0 + 1, 0, maxX)), u32(clamp(y0, 0, maxY))), 0);
  let t01 = textureLoad(tex, vec2u(u32(clamp(x0, 0, maxX)), u32(clamp(y0 + 1, 0, maxY))), 0);
  let t11 = textureLoad(tex, vec2u(u32(clamp(x0 + 1, 0, maxX)), u32(clamp(y0 + 1, 0, maxY))), 0);
  return mix(mix(t00, t10, tx), mix(t01, t11, tx), ty);
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}

@fragment
fn fsMain(in: VSOut) -> FSOut {
  var out: FSOut;
  let fineW = f32(max(u.fineW, 1u));
  let fineH = f32(max(u.fineH, 1u));
  // Framebuffer pixels, same space as marchPixel / the box overlay — not clip UVs.
  let src = isoCoarseTexel(occlTex, fineW, fineH, in.pos.xy);
  let c = loadBilinearAt(colorTex, src);
  let n = loadBilinearAt(normalTex, src);
  let ocl = loadBilinearAt(occlTex, src);
  if (isoNeedRefine(occlTex, fineW, fineH, in.pos.xy)) {
    out.color = vec4f(0.0);
    out.occl = vec4f(1.0, 0.0, 0.0, 1.0);
    out.normal = vec4f(0.0);
    out.depth = 1.0;
    return out;
  }
  out.color = c;
  out.occl = ocl;
  out.normal = n;
  out.depth = clamp(ocl.r, 0.0, 1.0);
  return out;
}
