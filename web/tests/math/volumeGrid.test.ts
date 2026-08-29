import { resampleVolumeGrid } from "../../src/math/volumeGrid.ts";
import { runSuite } from "../helpers/runner.ts";

function fillConstant(M: number, v: number): Float32Array {
  const d = new Float32Array(M ** 3);
  d.fill(v);
  return d;
}

function fillRamp(M: number): Float32Array {
  const d = new Float32Array(M ** 3);
  for (let z = 0; z < M; z++) {
    for (let y = 0; y < M; y++) {
      for (let x = 0; x < M; x++) {
        d[z * M * M + y * M + x] = x / Math.max(1, M - 1);
      }
    }
  }
  return d;
}

function sampleAt(d: Float32Array, M: number, x: number, y: number, z: number): number {
  return d[z * M * M + y * M + x] ?? 0;
}

export async function run() {
  return runSuite("volumeGrid", [
    {
      name: "identity when M matches",
      fn() {
        const src = fillConstant(5, 0.42);
        const out = resampleVolumeGrid(src, 5);
        if (out !== src) throw new Error("expected same reference for matching size");
      },
    },
    {
      name: "constant field preserved on upsample",
      fn() {
        const src = fillConstant(3, 1.25);
        const out = resampleVolumeGrid(src, 7);
        if (out.length !== 7 ** 3) throw new Error(`bad length ${out.length}`);
        for (let i = 0; i < out.length; i++) {
          if (Math.abs(out[i]! - 1.25) > 1e-5) throw new Error(`voxel ${i} = ${out[i]}`);
        }
      },
    },
    {
      name: "corners preserved on upsample ramp",
      fn() {
        const src = fillRamp(3);
        const out = resampleVolumeGrid(src, 5);
        const eps = 1e-4;
        if (Math.abs(sampleAt(out, 5, 0, 0, 0) - 0) > eps) throw new Error("min corner");
        if (Math.abs(sampleAt(out, 5, 4, 0, 0) - 1) > eps) throw new Error("max x corner");
      },
    },
    {
      name: "downsample averages neighborhood",
      fn() {
        const src = fillConstant(5, 2);
        const out = resampleVolumeGrid(src, 3);
        if (out.length !== 27) throw new Error(`bad length ${out.length}`);
        for (let i = 0; i < out.length; i++) {
          if (Math.abs(out[i]! - 2) > 1e-5) throw new Error(`voxel ${i} = ${out[i]}`);
        }
      },
    },
  ]);
}
