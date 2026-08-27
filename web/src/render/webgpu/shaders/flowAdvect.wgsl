struct FlowParams {
  gridM: u32,
  half: f32,
  dt: f32,
  dissipation: f32,
}

@group(0) @binding(0) var<uniform> params: FlowParams;
@group(0) @binding(1) var<storage, read> velocity: array<f32>;
@group(0) @binding(2) var<storage, read> dyeIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> dyeOut: array<f32>;

fn chebIndex(xi: f32) -> f32 {
  let x = clamp(xi, -1.0, 1.0);
  return f32(params.gridM) / 3.141592653589793 * acos(x) - 0.5;
}

fn sampleDye(p: vec3f) -> f32 {
  let half = params.half;
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let fx = chebIndex(xi.x);
  let fy = chebIndex(xi.y);
  let fz = chebIndex(xi.z);
  let M = i32(params.gridM);
  let x0 = clamp(i32(floor(fx)), 0, M - 2);
  let y0 = clamp(i32(floor(fy)), 0, M - 2);
  let z0 = clamp(i32(floor(fz)), 0, M - 2);
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let tz = fz - f32(z0);
  var s: f32 = 0.0;
  for (var dz: i32 = 0; dz <= 1; dz++) {
    for (var dy: i32 = 0; dy <= 1; dy++) {
      for (var dx: i32 = 0; dx <= 1; dx++) {
        let ix = x0 + dx;
        let iy = y0 + dy;
        let iz = z0 + dz;
        let idx = u32(ix) + u32(iy) * params.gridM + u32(iz) * params.gridM * params.gridM;
        let w = mix(1.0 - tx, tx, f32(dx)) * mix(1.0 - ty, ty, f32(dy)) * mix(1.0 - tz, tz, f32(dz));
        s += dyeIn[idx] * w;
      }
    }
  }
  return s;
}

fn sampleVel(p: vec3f) -> vec3f {
  let half = params.half;
  let xi = clamp(p / half, vec3f(-1.0), vec3f(1.0));
  let fx = chebIndex(xi.x);
  let fy = chebIndex(xi.y);
  let fz = chebIndex(xi.z);
  let M = i32(params.gridM);
  let x0 = clamp(i32(floor(fx)), 0, M - 2);
  let y0 = clamp(i32(floor(fy)), 0, M - 2);
  let z0 = clamp(i32(floor(fz)), 0, M - 2);
  let tx = fx - f32(x0);
  let ty = fy - f32(y0);
  let tz = fz - f32(z0);
  var v = vec3f(0.0);
  let volN = params.gridM * params.gridM * params.gridM;
  for (var dz: i32 = 0; dz <= 1; dz++) {
    for (var dy: i32 = 0; dy <= 1; dy++) {
      for (var dx: i32 = 0; dx <= 1; dx++) {
        let ix = x0 + dx;
        let iy = y0 + dy;
        let iz = z0 + dz;
        let idx = u32(ix) + u32(iy) * params.gridM + u32(iz) * params.gridM * params.gridM;
        let w = mix(1.0 - tx, tx, f32(dx)) * mix(1.0 - ty, ty, f32(dy)) * mix(1.0 - tz, tz, f32(dz));
        v += vec3f(velocity[idx], velocity[volN + idx], velocity[2u * volN + idx]) * w;
      }
    }
  }
  return v;
}

@compute @workgroup_size(4, 4, 4)
fn csMain(@builtin(global_invocation_id) gid: vec3u) {
  let M = params.gridM;
  if (gid.x >= M || gid.y >= M || gid.z >= M) {
    return;
  }
  let idx = gid.x + gid.y * M + gid.z * M * M;
  let half = params.half;
  let xi = cos(3.141592653589793 * (f32(gid.x) + 0.5) / f32(M));
  let yi = cos(3.141592653589793 * (f32(gid.y) + 0.5) / f32(M));
  let zi = cos(3.141592653589793 * (f32(gid.z) + 0.5) / f32(M));
  let p = vec3f(xi, yi, zi) * half;
  let v = sampleVel(p);
  let pBack = clamp(p - v * params.dt, vec3f(-half), vec3f(half));
  var d = sampleDye(pBack);
  d *= max(0.0, 1.0 - params.dissipation);
  dyeOut[idx] = d;
}
