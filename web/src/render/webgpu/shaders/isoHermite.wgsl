struct DrawParams {
  fbW: u32,
  fbH: u32,
  gridM: u32,
  steps: u32,
  half: f32,
  scale: f32,
  isoLevel: f32,
  debugTint: f32,
  ro: vec3f,
  volBase: f32,
  m0: vec4f,
  m1: vec4f,
  m2: vec4f,
  volBaseB: f32,
  blendT: f32,
  gradCount: f32,
  _padG: f32,
  g0: vec4f,
  g1: vec4f,
  g2: vec4f,
  g3: vec4f,
  g4: vec4f,
  g5: vec4f,
}

@group(0) @binding(0) var<uniform> draw: DrawParams;
@group(0) @binding(1) var<storage, read> volume: array<f32>;

struct VSOut { @builtin(position) pos: vec4f, }

struct FSOut {
  @location(0) color: vec4f,
  @location(1) occl: vec4f,
  @location(2) normal: vec4f,
  @builtin(frag_depth) depth: f32,
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VSOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  return o;
}

fn densAt(ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(draw.gridM);
  let base = u32(draw.volBase);
  let x = clamp(ix, 0, M - 1);
  let y = clamp(iy, 0, M - 1);
  let z = clamp(iz, 0, M - 1);
  return volume[base + u32(x) + u32(y) * draw.gridM + u32(z) * draw.gridM * draw.gridM];
}

fn densAtBase(base: u32, ix: i32, iy: i32, iz: i32) -> f32 {
  let M = i32(draw.gridM);
  let x = clamp(ix, 0, M - 1);
  let y = clamp(iy, 0, M - 1);
  let z = clamp(iz, 0, M - 1);
  return volume[base + u32(x) + u32(y) * draw.gridM + u32(z) * draw.gridM * draw.gridM];
}

fn chebIndex(xi: f32) -> f32 {
  let x = clamp(xi, -1.0, 1.0);
  return f32(draw.gridM) / 3.141592653589793 * acos(x) - 0.5;
}

fn sampleFieldBase(base: u32, p: vec3f) -> f32 {
  let half = draw.half;
  // Clamp to the fit domain instead of returning 0 outside — a hard exterior
  // zero creates discontinuous normals / grainy isos on the box faces.
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let fx = chebIndex(xi.x); let fy = chebIndex(xi.y); let fz = chebIndex(xi.z);
  let x0 = i32(floor(fx)); let y0 = i32(floor(fy)); let z0 = i32(floor(fz));
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let tz = clamp(fz - f32(z0), 0.0, 1.0);
  let c00 = mix(densAtBase(base, x0, y0, z0), densAtBase(base, x0 + 1, y0, z0), tx);
  let c10 = mix(densAtBase(base, x0, y0 + 1, z0), densAtBase(base, x0 + 1, y0 + 1, z0), tx);
  let c01 = mix(densAtBase(base, x0, y0, z0 + 1), densAtBase(base, x0 + 1, y0, z0 + 1), tx);
  let c11 = mix(densAtBase(base, x0, y0 + 1, z0 + 1), densAtBase(base, x0 + 1, y0 + 1, z0 + 1), tx);
  return mix(mix(c00, c10, ty), mix(c01, c11, ty), tz);
}

/** Chebyshev-root ξ at dens-grid index i. */
fn nodeXi(i: i32) -> f32 {
  return cos((f32(i) + 0.5) * 3.141592653589793 / f32(draw.gridM));
}

fn h00(t: f32) -> f32 { let t2 = t * t; return (2.0 * t - 3.0) * t2 + 1.0; }
fn h10(t: f32) -> f32 { let t2 = t * t; return (t - 2.0) * t2 + t; }
fn h01(t: f32) -> f32 { let t2 = t * t; return (-2.0 * t + 3.0) * t2; }
fn h11(t: f32) -> f32 { let t2 = t * t; return (t - 1.0) * t2; }
fn h00d(t: f32) -> f32 { return 6.0 * t * (t - 1.0); }
fn h10d(t: f32) -> f32 { return 3.0 * t * t - 4.0 * t + 1.0; }
fn h01d(t: f32) -> f32 { return 6.0 * t * (1.0 - t); }
fn h11d(t: f32) -> f32 { return 3.0 * t * t - 2.0 * t; }

fn hermiteVal(t: f32, f0: f32, f1: f32, d0: f32, d1: f32) -> f32 {
  return h00(t) * f0 + h10(t) * d0 + h01(t) * f1 + h11(t) * d1;
}
fn hermiteDt(t: f32, f0: f32, f1: f32, d0: f32, d1: f32) -> f32 {
  return h00d(t) * f0 + h10d(t) * d0 + h01d(t) * f1 + h11d(t) * d1;
}

/**
 * Tricubic Hermite (tensor-product) using dens + ∂f/∂ξ slabs at the 8 cell corners.
 * Derivatives are scaled to the unit-cell parameter t∈[0,1]. Mixed partials are
 * estimated by finite differences of ∇f across the cell (C¹ Hermite).
 * Returns (f, ∂f/∂x, ∂f/∂y, ∂f/∂z) in world space.
 */
fn sampleFieldHermite4(base: u32, p: vec3f) -> vec4f {
  let half = draw.half;
  let M = i32(draw.gridM);
  let last = max(M - 2, 0);
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let fx = chebIndex(xi.x); let fy = chebIndex(xi.y); let fz = chebIndex(xi.z);
  let x0 = clamp(i32(floor(fx)), 0, last);
  let y0 = clamp(i32(floor(fy)), 0, last);
  let z0 = clamp(i32(floor(fz)), 0, last);
  let tx = clamp(fx - f32(x0), 0.0, 1.0);
  let ty = clamp(fy - f32(y0), 0.0, 1.0);
  let tz = clamp(fz - f32(z0), 0.0, 1.0);

  let volN = draw.gridM * draw.gridM * draw.gridM;
  let bGx = base + volN;
  let bGy = base + 2u * volN;
  let bGz = base + 3u * volN;
  let dltX = nodeXi(x0 + 1) - nodeXi(x0);
  let dltY = nodeXi(y0 + 1) - nodeXi(y0);
  let dltZ = nodeXi(z0 + 1) - nodeXi(z0);

  var F: array<f32, 8>;
  var DX: array<f32, 8>;
  var DY: array<f32, 8>;
  var DZ: array<f32, 8>;
  for (var k: i32 = 0; k < 2; k++) {
    for (var j: i32 = 0; j < 2; j++) {
      for (var i: i32 = 0; i < 2; i++) {
        let idx = i + j * 2 + k * 4;
        let ix = x0 + i; let iy = y0 + j; let iz = z0 + k;
        F[idx] = densAtBase(base, ix, iy, iz);
        DX[idx] = densAtBase(bGx, ix, iy, iz) * dltX;
        DY[idx] = densAtBase(bGy, ix, iy, iz) * dltY;
        DZ[idx] = densAtBase(bGz, ix, iy, iz) * dltZ;
      }
    }
  }

  // Cell-wide mixed partials from FD of first derivatives (same for all corners).
  var dxy = 0.0;
  var dxz = 0.0;
  var dyz = 0.0;
  for (var j: i32 = 0; j < 2; j++) {
    for (var k: i32 = 0; k < 2; k++) {
      let a = j * 2 + k * 4;
      dxy += 0.25 * (DY[1 + a] - DY[0 + a]);
      dxz += 0.25 * (DZ[1 + a] - DZ[0 + a]);
    }
  }
  for (var i: i32 = 0; i < 2; i++) {
    for (var k: i32 = 0; k < 2; k++) {
      dyz += 0.25 * (DZ[i + 2 + k * 4] - DZ[i + k * 4]);
    }
  }
  // Also average the dual FD for dxy/dxz for symmetry.
  var dxy2 = 0.0;
  var dxz2 = 0.0;
  var dyz2 = 0.0;
  for (var i: i32 = 0; i < 2; i++) {
    for (var k: i32 = 0; k < 2; k++) {
      dxy2 += 0.25 * (DX[i + 2 + k * 4] - DX[i + k * 4]);
    }
    for (var j: i32 = 0; j < 2; j++) {
      dxz2 += 0.25 * (DX[i + j * 2 + 4] - DX[i + j * 2]);
      dyz2 += 0.25 * (DY[i + j * 2 + 4] - DY[i + j * 2]);
    }
  }
  dxy = 0.5 * (dxy + dxy2);
  dxz = 0.5 * (dxz + dxz2);
  dyz = 0.5 * (dyz + dyz2);
  let dxyz = 0.5 * (
    (DZ[1 + 2 + 0] - DZ[0 + 2 + 0]) - (DZ[1 + 0 + 0] - DZ[0 + 0 + 0])
    + (DZ[1 + 2 + 4] - DZ[0 + 2 + 4]) - (DZ[1 + 0 + 4] - DZ[0 + 0 + 4])
  ) * 0.5;

  // X pass → 4 values on the yz face (pack j+2*k).
  var U: array<f32, 4>;
  var UY: array<f32, 4>;
  var UZ: array<f32, 4>;
  var UYZ: array<f32, 4>;
  var UX: array<f32, 4>;
  var UXY: array<f32, 4>;
  var UXZ: array<f32, 4>;
  var UXYZ: array<f32, 4>;
  for (var k: i32 = 0; k < 2; k++) {
    for (var j: i32 = 0; j < 2; j++) {
      let pjk = j + k * 2;
      let i0 = 0 + j * 2 + k * 4;
      let i1 = 1 + j * 2 + k * 4;
      U[pjk] = hermiteVal(tx, F[i0], F[i1], DX[i0], DX[i1]);
      UY[pjk] = hermiteVal(tx, DY[i0], DY[i1], dxy, dxy);
      UZ[pjk] = hermiteVal(tx, DZ[i0], DZ[i1], dxz, dxz);
      UYZ[pjk] = hermiteVal(tx, dyz, dyz, dxyz, dxyz);
      UX[pjk] = hermiteDt(tx, F[i0], F[i1], DX[i0], DX[i1]);
      UXY[pjk] = hermiteDt(tx, DY[i0], DY[i1], dxy, dxy);
      UXZ[pjk] = hermiteDt(tx, DZ[i0], DZ[i1], dxz, dxz);
      UXYZ[pjk] = hermiteDt(tx, dyz, dyz, dxyz, dxyz);
    }
  }

  // Y pass → 2 values on the z edge.
  let V0 = hermiteVal(ty, U[0], U[1], UY[0], UY[1]);
  let V1 = hermiteVal(ty, U[2], U[3], UY[2], UY[3]);
  let VZ0 = hermiteVal(ty, UZ[0], UZ[1], UYZ[0], UYZ[1]);
  let VZ1 = hermiteVal(ty, UZ[2], UZ[3], UYZ[2], UYZ[3]);
  let VX0 = hermiteVal(ty, UX[0], UX[1], UXY[0], UXY[1]);
  let VX1 = hermiteVal(ty, UX[2], UX[3], UXY[2], UXY[3]);
  let VY0 = hermiteDt(ty, U[0], U[1], UY[0], UY[1]);
  let VY1 = hermiteDt(ty, U[2], U[3], UY[2], UY[3]);
  let VYZ0 = hermiteDt(ty, UZ[0], UZ[1], UYZ[0], UYZ[1]);
  let VYZ1 = hermiteDt(ty, UZ[2], UZ[3], UYZ[2], UYZ[3]);
  let VXZ0 = hermiteVal(ty, UXZ[0], UXZ[1], UXYZ[0], UXYZ[1]);
  let VXZ1 = hermiteVal(ty, UXZ[2], UXZ[3], UXYZ[2], UXYZ[3]);

  // Z pass → value + parametric derivatives.
  let f = hermiteVal(tz, V0, V1, VZ0, VZ1);
  let dF_dtz = hermiteDt(tz, V0, V1, VZ0, VZ1);
  let dF_dty = hermiteVal(tz, VY0, VY1, VYZ0, VYZ1);
  let dF_dtx = hermiteVal(tz, VX0, VX1, VXZ0, VXZ1);

  let invH = select(0.0, 1.0 / half, abs(half) > 1e-12);
  let invDltX = select(0.0, 1.0 / dltX, abs(dltX) > 1e-15);
  let invDltY = select(0.0, 1.0 / dltY, abs(dltY) > 1e-15);
  let invDltZ = select(0.0, 1.0 / dltZ, abs(dltZ) > 1e-15);
  return vec4f(
    f,
    dF_dtx * invDltX * invH,
    dF_dty * invDltY * invH,
    dF_dtz * invDltZ * invH,
  );
}

/** Ray-walk interpolant: cheap trilinear. Hermite is reserved for hit normals. */
fn sampleVolume(p: vec3f) -> f32 {
  let baseA = u32(draw.volBase);
  let t = draw.blendT;
  if (t <= 1e-5) {
    return sampleFieldBase(baseA, p);
  }
  let baseB = u32(draw.volBaseB);
  if (t >= 0.999) {
    return sampleFieldBase(baseB, p);
  }
  return mix(sampleFieldBase(baseA, p), sampleFieldBase(baseB, p), t);
}

fn fieldAt(p: vec3f) -> f32 {
  var d = sampleVolume(p);
  if (d != d) { d = 0.0; }
  return d - draw.isoLevel;
}

/** Blended dens at an integer Chebyshev-grid vertex (pre-iso). */
fn blendedDens(ix: i32, iy: i32, iz: i32) -> f32 {
  let a = densAtBase(u32(draw.volBase), ix, iy, iz);
  let b = densAtBase(u32(draw.volBaseB), ix, iy, iz);
  return mix(a, b, draw.blendT);
}

/** Lower corner of the trilinear cell containing p (in Chebyshev index space). */
fn cellIndexAt(p: vec3f) -> vec3i {
  let half = draw.half;
  let M = i32(draw.gridM);
  let last = max(M - 2, 0);
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let x0 = clamp(i32(floor(chebIndex(xi.x))), 0, last);
  let y0 = clamp(i32(floor(chebIndex(xi.y))), 0, last);
  let z0 = clamp(i32(floor(chebIndex(xi.z))), 0, last);
  return vec3i(x0, y0, z0);
}

/**
 * Final-hit filter: reject hits in cells whose 8 corner dens values do not
 * bracket isoLevel. Hermite can overshoot, so this is conservative (may drop
 * some valid zeros) but still kills most ghost-halo near-misses.
 */
fn cellBracketsIso(p: vec3f) -> bool {
  let c = cellIndexAt(p);
  let iso = draw.isoLevel;
  var lo = 1e30;
  var hi = -1e30;
  for (var dz: i32 = 0; dz < 2; dz++) {
    for (var dy: i32 = 0; dy < 2; dy++) {
      for (var dx: i32 = 0; dx < 2; dx++) {
        let v = blendedDens(c.x + dx, c.y + dy, c.z + dz) - iso;
        lo = min(lo, v);
        hi = max(hi, v);
      }
    }
  }
  return lo <= 0.0 && hi >= 0.0;
}

/** ∇f of the active iso interpolant (world space). */
fn fieldGrad(p: vec3f) -> vec3f {
  let t = draw.blendT;
  if (t <= 1e-5) {
    return sampleFieldHermite4(u32(draw.volBase), p).yzw;
  }
  if (t >= 0.999) {
    return sampleFieldHermite4(u32(draw.volBaseB), p).yzw;
  }
  let a = sampleFieldHermite4(u32(draw.volBase), p);
  let b = sampleFieldHermite4(u32(draw.volBaseB), p);
  return mix(a.yzw, b.yzw, t);
}

{{GRADIENT_WGSL}}

fn shadeIso(p: vec3f, rd: vec3f, n: vec3f) -> vec4f {
  var stops: array<vec4f, {{MAX_GRAD_STOPS}}>;
  stops[0] = vec4f(draw.g0.xyz, draw.gradCount);
  stops[1] = draw.g1;
  stops[2] = draw.g2;
  stops[3] = draw.g3;
  stops[4] = draw.g4;
  stops[5] = draw.g5;

  let L = normalize(vec3f(0.35, 0.85, 0.45));
  let V = normalize(-rd);
  let ndotl = max(dot(n, L), 0.0);
  let ndotv = max(dot(n, V), 0.0);

  let tBase = isoGradientT(p, n, draw.half);
  // Full-hemisphere wrap so the dark side still has variation (not clamped to 0).
  let wrapped = clamp(0.5 + 0.5 * dot(n, L), 0.0, 1.0);
  // Fold: midtones → one gradient end; darkest + lightest → the other.
  let lit = pow(wrapped, 0.85);
  let tFold = 1.0 - abs(2.0 * lit - 1.0);
  let tLight = mix(0.04, 0.96, tFold);
  // Lean a bit more on spatial/normal base in deep shadow so backsides aren't flat.
  let shadeW = mix(0.48, 0.72, ndotl);
  let t = clamp(mix(tBase, tLight, shadeW), 0.0, 1.0);
  var rgb = sampleGradStops(&stops, t);

  let nStops = max(min(u32(draw.gradCount), MAX_GRAD_STOPS), 1u);
  let c0 = stops[0].xyz;
  let c1 = stops[nStops - 1u].xyz;

  // Mild fresnel — keep some rim without making silhouettes scream with view angle.
  let fresnel = pow(1.0 - ndotv, 2.4);
  rgb += c1 * fresnel * 0.2;

  // Specular on blended albedo
  let H = normalize(L + V);
  let spec = pow(max(dot(n, H), 0.0), 40.0);
  rgb += mix(c0, c1, 0.7) * spec * 0.45;

  return vec4f(rgb, 1.0);
}

fn isoNormal(p: vec3f, rd: vec3f) -> vec3f {
  // Evaluate ∇f slightly inside the fit domain — endpoint clamp makes
  // boundary gradients noisy (grain where isos meet the box faces).
  let half = draw.half;
  let inset = max(1e-4, 2e-3 * half);
  let pIn = clamp(p, vec3f(-half + inset), vec3f(half - inset));
  var g = fieldGrad(pIn);
  let gl = length(g);
  var n = select(vec3f(0.0, 1.0, 0.0), g / gl, gl > 1e-8);
  if (dot(n, -rd) < 0.0) { n = -n; }
  return n;
}

/**
 * Walk [t0,t1] and return the FIRST sign-change sub-bracket (near zero along the ray).
 * Returns (lo, hi, flo, fAtT1). Empty when hi < lo; fAtT1 is always fieldAt(t1).
 */
fn firstZeroBracket(ro: vec3f, rd: vec3f, t0: f32, t1: f32, f0: f32, samples: u32) -> vec4f {
  var prevT = t0;
  var prevF = f0;
  let n = max(samples, 1u);
  var fEnd = f0;
  for (var s: u32 = 1u; s <= n; s++) {
    let t = mix(t0, t1, f32(s) / f32(n));
    let f = fieldAt(ro + rd * t);
    fEnd = f;
    if (prevF * f <= 0.0) {
      return vec4f(prevT, t, prevF, f); // w unused on hit path
    }
    prevT = t;
    prevF = f;
  }
  return vec4f(t1, t0, f0, fEnd); // empty: hi < lo
}

/** Bisect an already-isolated nearest-zero bracket (HEAD-cost refine). */
fn bisectIso(ro: vec3f, rd: vec3f, lo: f32, hi: f32, flo: f32) -> f32 {
  var a = lo;
  var b = hi;
  var fa = flo;
  for (var k: u32 = 0u; k < 12u; k++) {
    let mid = 0.5 * (a + b);
    let fm = fieldAt(ro + rd * mid);
    if (fa * fm <= 0.0) { b = mid; } else { a = mid; fa = fm; }
  }
  return 0.5 * (a + b);
}

fn marchIso(ro: vec3f, rd: vec3f, tEnter: f32, tExit: f32) -> FSOut {
  var out: FSOut;
  out.color = vec4f(0.0);
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);
  out.normal = vec4f(0.0);
  out.depth = 1.0;

  var steps = draw.steps;
  if (steps < 16u) { steps = 16u; }
  if (steps > 192u) { steps = 192u; }
  let dt = (tExit - tEnter) / f32(steps);
  var s0 = tEnter;
  var f0 = fieldAt(ro + rd * s0);
  let far = max(tExit, draw.half * 4.0);
  var dfdT = 0.0;
  var hasDfdT = false;

  for (var i: u32 = 0u; i < 192u; i++) {
    if (i >= steps) { break; }
    let s1 = min(s0 + dt, tExit);
    let ds = max(s1 - s0, 1e-8);

    // Far from iso along the ray → HEAD path: one endpoint sample only.
    let distT = select(0.0, abs(f0) / max(abs(dfdT), 1e-8), hasDfdT);
    let farFromIso = hasDfdT && (distT > 2.0 * dt);

    var f1: f32;
    var hit = -1.0;
    if (farFromIso) {
      f1 = fieldAt(ro + rd * s1);
      if (f0 * f1 < 0.0) {
        hit = bisectIso(ro, rd, s0, s1, f0);
      }
    } else {
      // Near band (or first step): densify; reuse endpoint from the scan.
      let nDense = select(
        4u,
        u32(clamp(8.0 * dt / max(distT, 1e-6), 1.0, 8.0)),
        hasDfdT,
      );
      let br = firstZeroBracket(ro, rd, s0, s1, f0, nDense);
      f1 = br.w;
      if (br.y > br.x) {
        hit = bisectIso(ro, rd, br.x, br.y, br.z);
      }
    }

    if (hit >= 0.0) {
      let p = ro + rd * hit;
      if (cellBracketsIso(p)) {
        let d = clamp(hit / far, 0.0, 0.999);
        let n = isoNormal(p, rd);
        out.color = shadeIso(p, rd, n);
        out.occl = vec4f(d, 0.0, 0.0, 1.0);
        out.normal = vec4f(n * 0.5 + 0.5, 1.0);
        out.depth = d;
        return out;
      }
    }

    dfdT = (f1 - f0) / ds;
    hasDfdT = true;
    s0 = s1;
    f0 = f1;
    if (s0 >= tExit - 1e-6) { break; }
  }
  return out;
}

fn marchPixel(fragPos: vec2f) -> FSOut {
  var out: FSOut;
  out.color = vec4f(0.0);
  out.occl = vec4f(1.0, 0.0, 0.0, 1.0);
  out.normal = vec4f(0.0);
  out.depth = 1.0;

  let fbW = f32(draw.fbW); let fbH = f32(draw.fbH);
  let ndcX = -1.0 + 2.0 * fragPos.x / fbW;
  let ndcY = 1.0 - 2.0 * fragPos.y / fbH;
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
  let tExit = min(min(tmax.x, tmax.y), tmax.z);
  // Stay clear of the discontinuous clamped boundary.
  let edgeEps = (tExit - tEnter) * 2e-4 + 5e-5 * half;
  tEnter = tEnter + edgeEps;
  let tExitIn = tExit - edgeEps;
  if (!(tExitIn > tEnter + 1e-6)) {
    discard;
    return out;
  }
  let hit = marchIso(ro, rd, tEnter, tExitIn);
  // Miss: discard so we don't overwrite a closer prior iso's color/depth.
  if (hit.color.a < 0.5) {
    discard;
    return out;
  }
  return hit;
}

@fragment
fn fsMain(in: VSOut) -> FSOut {
  return marchPixel(in.pos.xy);
}
