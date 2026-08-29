import { compileExpr, fitChebyshev3D } from "../../src/math/fit.ts";
import { compileVectorExpr } from "../../src/math/fitVector.ts";
import {
  chebDefiniteInt3D,
  embedReducedCheb3D,
  idctChebCurl3D,
  idctChebDivergence3D,
  idctChebLaplacian3D,
  idctChebPartial3D,
} from "../../src/math/idct.ts";
import { assert } from "../helpers/assert.ts";
import { chebWorld, densIndex } from "../helpers/grid.ts";
import { runSuite } from "../helpers/runner.ts";

function maxGridError(
  dens: ArrayLike<number>,
  M: number,
  half: number,
  truth: (x: number, y: number, z: number) => number,
) {
  let maxErr = 0;
  for (let ix = 0; ix < M; ix++) {
    const x = chebWorld(ix, M, half);
    for (let iy = 0; iy < M; iy++) {
      const y = chebWorld(iy, M, half);
      for (let iz = 0; iz < M; iz++) {
        const z = chebWorld(iz, M, half);
        maxErr = Math.max(maxErr, Math.abs(dens[densIndex(ix, iy, iz, M)]! - truth(x, y, z)));
      }
    }
  }
  return maxErr;
}

export async function run() {
  return runSuite("math / idct-operators", [
    {
      name: "idctChebLaplacian3D of r^2 ≈ 6",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const deg = 8;
        const fit = fitChebyshev3D(fn, half, deg, { skipMono: true });
        const { dens, M } = idctChebLaplacian3D(fit.cheb, deg);
        const maxErr = maxGridError(dens, M, half, () => 6);
        assert(maxErr < 0.15, `laplacian max err ${maxErr}`);
      },
    },
    {
      name: "idctChebPartial3D ∂/∂x of r^2 ≈ 2x",
      fn: () => {
        const fn = compileExpr("x^2+y^2+z^2").bind({});
        const half = 1;
        const deg = 8;
        const fit = fitChebyshev3D(fn, half, deg, { skipMono: true });
        const { dens, M } = idctChebPartial3D(fit.cheb, deg, 0);
        const scaled = new Float32Array(dens.length);
        const scale = 1 / half;
        for (let i = 0; i < scaled.length; i++) scaled[i] = dens[i]! * scale;
        const maxErr = maxGridError(scaled, M, half, (x) => 2 * x);
        assert(maxErr < 0.25, `partial x max err ${maxErr}`);
      },
    },
    {
      name: "idctChebDivergence3D of (x,y,z) ≈ 3",
      fn: () => {
        const half = 1;
        const deg = 8;
        const compiled = compileVectorExpr("(x,y,z)");
        const vectorFn = compiled.bind({});
        const fitX = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[0]!, half, deg, {
          skipMono: true,
        });
        const fitY = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[1]!, half, deg, {
          skipMono: true,
        });
        const fitZ = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[2]!, half, deg, {
          skipMono: true,
        });
        const { dens, M } = idctChebDivergence3D(fitX.cheb, fitY.cheb, fitZ.cheb, deg);
        const scaled = new Float32Array(dens.length);
        const scale = 1 / half;
        for (let i = 0; i < scaled.length; i++) scaled[i] = dens[i]! * scale;
        const maxErr = maxGridError(scaled, M, half, () => 3);
        assert(maxErr < 0.25, `divergence max err ${maxErr}`);
      },
    },
    {
      name: "idctChebCurl3D of (-y,x,0) ≈ (0,0,2)",
      fn: () => {
        const half = 1;
        const deg = 8;
        const compiled = compileVectorExpr("(-y,x,0)");
        const vectorFn = compiled.bind({});
        const fitX = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[0]!, half, deg, {
          skipMono: true,
        });
        const fitY = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[1]!, half, deg, {
          skipMono: true,
        });
        const fitZ = fitChebyshev3D((x, y, z) => vectorFn(x, y, z)[2]!, half, deg, {
          skipMono: true,
        });
        const curl = idctChebCurl3D(fitX.cheb, fitY.cheb, fitZ.cheb, deg);
        const M = curl.M;
        const scale = 1 / half;
        const fx = new Float32Array(curl.fx.length);
        const fy = new Float32Array(curl.fy.length);
        const fz = new Float32Array(curl.fz.length);
        for (let i = 0; i < fx.length; i++) {
          fx[i] = curl.fx[i]! * scale;
          fy[i] = curl.fy[i]! * scale;
          fz[i] = curl.fz[i]! * scale;
        }
        const errX = maxGridError(fx, M, half, () => 0);
        const errY = maxGridError(fy, M, half, () => 0);
        const errZ = maxGridError(fz, M, half, () => 2);
        assert(errX < 0.3 && errY < 0.3 && errZ < 0.35, `curl errs ${errX}, ${errY}, ${errZ}`);
      },
    },
    {
      name: "chebDefiniteInt3D along x smoke",
      fn: () => {
        const fn = compileExpr("x").bind({});
        const half = 1;
        const deg = 6;
        const fit = fitChebyshev3D(fn, half, deg, { skipMono: true });
        const integrated = chebDefiniteInt3D(fit.cheb, deg, 0, -0.5, 0.5, half);
        assert(integrated.length === (deg + 1) ** 3, "coeff length");
        assert(Math.abs(integrated[0]!) > 0, "nonzero integral coeff");
      },
    },
    {
      name: "embedReducedCheb3D broadcasts integrated axis",
      fn: () => {
        const fn = compileExpr("x+y+z").bind({});
        const deg = 4;
        const fit = fitChebyshev3D(fn, 1, deg, { skipMono: true });
        const reduced = chebDefiniteInt3D(fit.cheb, deg, 0, -1, 1, 1);
        const embedded = embedReducedCheb3D(reduced, deg, [0]);
        assert(embedded.length === (deg + 1) ** 3, "embedded size");
        const v00 = embedded[0]!;
        const v10 = embedded[1]!;
        assert(Math.abs(v00 - v10) < 1e-12, "constant along integrated x");
        assert(Math.abs(v00) > 0, "nonzero embed");
      },
    },
    {
      name: "chebDefiniteInt3D along y and z",
      fn: () => {
        const fn = compileExpr("y+z").bind({});
        const deg = 4;
        const fit = fitChebyshev3D(fn, 1, deg, { skipMono: true });
        const alongY = chebDefiniteInt3D(fit.cheb, deg, 1, -0.5, 0.5, 1);
        const alongZ = chebDefiniteInt3D(fit.cheb, deg, 2, -0.5, 0.5, 1);
        assert(alongY.length === (deg + 1) ** 3, "y integral size");
        assert(alongZ.length === (deg + 1) ** 3, "z integral size");
        assert(Math.abs(alongY[0]!) > 0 || Math.abs(alongZ[0]!) > 0, "nonzero coeff");
      },
    },
  ]);
}
