/** Density expression (MathLive/LaTeX via Compute Engine) + 3D Chebyshev fit → world monomials. */

import { compile } from "@cortex-js/compute-engine";
import { MAX_DEG } from "./clipGrid.js";

/** Preset densities as LaTeX (shown in the MathLive field). */
export const PRESETS = {
  blob: {
    label: "Gaussian blob",
    latex: String.raw`\exp(-(x^{2}+y^{2}+z^{2}))`,
  },
  soft: {
    label: "Soft ellipsoid",
    latex: String.raw`\exp(-(x^{2}+0.5y^{2}+2z^{2}))`,
  },
  two: {
    label: "Two blobs",
    latex: String.raw`\exp(-4((x-0.7)^{2}+y^{2}+z^{2}))+\exp(-4((x+0.7)^{2}+y^{2}+z^{2}))`,
  },
  shell: {
    label: "Spherical shell",
    latex: String.raw`\exp(-12(\sqrt{x^{2}+y^{2}+z^{2}}-0.9)^{2})`,
  },
  ridge: {
    label: "Vertical ridge",
    latex: String.raw`\exp(-10x^{2})\exp(-0.4(y^{2}+z^{2}))`,
  },
  pulse: {
    label: "Pulse blob (a)",
    latex: String.raw`\exp(-(x^{2}+y^{2}+z^{2})/a^{2})`,
    params: { a: { value: 1, min: 0.35, max: 1.6, animate: true } },
  },
  twist: {
    label: "Two blobs (d)",
    latex: String.raw`\exp(-4((x-d)^{2}+y^{2}+z^{2}))+\exp(-4((x+d)^{2}+y^{2}+z^{2}))`,
    params: { d: { value: 0.7, min: 0.15, max: 1.2, animate: true } },
  },
};

/** Spatial vars; everything else in freeSymbols is a slider parameter. */
const RESERVED_SYMBOLS = new Set(["x", "y", "z"]);

function coerceNumber(v) {
  if (typeof v === "number") return v;
  if (v && typeof v.valueOf === "function") return Number(v.valueOf());
  return Number(v);
}

/**
 * Compile a density expression.
 *
 * @returns {{
 *   freeParams: string[],
 *   bind: (params?: Record<string, number>) => (x: number, y: number, z: number) => number,
 * }}
 */
export function compileExpr(raw) {
  const src = String(raw ?? "").trim();
  if (!src) throw new Error("Empty expression");

  const result = compile(src);
  if (!result?.success || typeof result.run !== "function") {
    const why = result?.unsupported?.length
      ? `unsupported: ${result.unsupported.join(", ")}`
      : "could not compile";
    throw new Error(`Expression ${why}`);
  }

  const { run, freeSymbols = [] } = result;
  const freeParams = [];
  for (const s of freeSymbols) {
    const id = String(s);
    if (RESERVED_SYMBOLS.has(id)) continue;
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(id)) {
      throw new Error(`Invalid parameter “${id}” (use letters / digits)`);
    }
    freeParams.push(id);
  }
  freeParams.sort();

  return {
    freeParams,
    /** Bind current parameter values → f(x,y,z). */
    bind(params = {}) {
      return (x, y, z) => {
        const scope = { x, y, z };
        for (const name of freeParams) {
          const v = params[name];
          scope[name] = Number.isFinite(v) ? v : 1;
        }
        return coerceNumber(run(scope));
      };
    },
  };
}

function fromUnit(u, a, b) {
  return 0.5 * (a + b) + 0.5 * (b - a) * u;
}

/**
 * DCT matrix W[i,a] = T_i(u_a) for Gauss–Chebyshev nodes.
 * Built via recurrence in O(n²).
 */
function chebWeightMatrix(n, uNodes) {
  const W = new Float64Array(n * n);
  for (let a = 0; a < n; a++) {
    const u = uNodes[a];
    W[a] = 1;
    if (n > 1) W[n + a] = u;
    for (let i = 2; i < n; i++) {
      W[i * n + a] = 2 * u * W[(i - 1) * n + a] - W[(i - 2) * n + a];
    }
  }
  return W;
}

/**
 * Separable 3D Chebyshev DCT of samples on the tensor Chebyshev grid.
 * Same math as the naive O(n⁶) sum, but three 1D passes → O(n⁴).
 *
 * c_ijk = (α_i α_j α_k / n³) Σ_{a,b,c} f_abc T_i(u_a) T_j(u_b) T_k(u_c)
 * with α_0 = 1, α_{>0} = 2.
 *
 * Packing: idx = x + y*n + z*n*n (same as vals / cheb elsewhere).
 */
function chebDCT3DSeparable(vals, n, uNodes) {
  const W = chebWeightMatrix(n, uNodes);
  const scale = new Float64Array(n);
  for (let i = 0; i < n; i++) scale[i] = (i === 0 ? 1 : 2) / n;

  const n2 = n * n;
  const tmp = new Float64Array(n * n * n);
  const tmp2 = new Float64Array(n * n * n);
  const out = new Float32Array(n * n * n);

  // X: vals[a,b,c] → tmp[i,b,c]
  for (let b = 0; b < n; b++) {
    for (let c = 0; c < n; c++) {
      const base = b * n + c * n2;
      for (let i = 0; i < n; i++) {
        let s = 0;
        const Wi = i * n;
        for (let a = 0; a < n; a++) s += vals[a + base] * W[Wi + a];
        tmp[i + base] = s * scale[i];
      }
    }
  }

  // Y: tmp[i,b,c] → tmp2[i,j,c]
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < n; c++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        const Wj = j * n;
        for (let b = 0; b < n; b++) s += tmp[i + b * n + c * n2] * W[Wj + b];
        tmp2[i + j * n + c * n2] = s * scale[j];
      }
    }
  }

  // Z: tmp2[i,j,c] → out[i,j,k]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const ij = i + j * n;
      for (let k = 0; k < n; k++) {
        let s = 0;
        const Wk = k * n;
        for (let c = 0; c < n; c++) s += tmp2[ij + c * n2] * W[Wk + c];
        out[ij + k * n2] = s * scale[k];
      }
    }
  }

  return out;
}

/** T_0..T_deg as monomial coeffs in u (length deg+1 arrays). */
function chebToMonoTable(deg) {
  const T = [[1], [0, 1]];
  for (let n = 2; n <= deg; n++) {
    const prev = T[n - 1];
    const prev2 = T[n - 2];
    const cur = new Array(n + 1).fill(0);
    for (let i = 0; i < prev.length; i++) cur[i + 1] += 2 * prev[i];
    for (let i = 0; i < prev2.length; i++) cur[i] -= prev2[i];
    T[n] = cur;
  }
  return T;
}

/**
 * Chebyshev tensor c_ijk T_i(x/h)T_j(y/h)T_k(z/h)
 * → monomials m_abc for x^a y^b z^c (same packing).
 */
function chebToMonomial3D(chebCoeffs, deg, half) {
  const N = deg;
  const n = N + 1;
  const T = chebToMonoTable(N);
  const invH = 1 / half;
  const mono = new Float64Array(n * n * n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        const c = chebCoeffs[i + j * n + k * n * n];
        if (Math.abs(c) < 1e-18) continue;
        const Ti = T[i];
        const Tj = T[j];
        const Tk = T[k];
        for (let a = 0; a < Ti.length; a++) {
          if (Ti[a] === 0) continue;
          for (let b = 0; b < Tj.length; b++) {
            if (Tj[b] === 0) continue;
            for (let d = 0; d < Tk.length; d++) {
              if (Tk[d] === 0) continue;
              const scale = c * Ti[a] * Tj[b] * Tk[d] * invH ** (a + b + d);
              mono[a + b * n + d * n * n] += scale;
            }
          }
        }
      }
    }
  }
  return Float32Array.from(mono);
}

export function evalMonomial3D(mono, deg, x, y, z) {
  const n = deg + 1;
  let s = 0;
  let xp = 1;
  for (let i = 0; i < n; i++) {
    let yp = 1;
    for (let j = 0; j < n; j++) {
      let zp = 1;
      for (let k = 0; k < n; k++) {
        s += mono[i + j * n + k * n * n] * xp * yp * zp;
        zp *= z;
      }
      yp *= y;
    }
    xp *= x;
  }
  return s;
}

/**
 * Fit f on [-half,half]^3 with tensor Chebyshev, convert to world monomials.
 * @param {{ skipL2?: boolean }} [opts]
 */
export function fitChebyshev3D(fn, half, deg, opts = {}) {
  const tAll = performance.now();
  const N = Math.max(0, Math.min(MAX_DEG, deg | 0));
  const n = N + 1;
  const uNodes = new Array(n);
  const pts = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.cos((Math.PI * (2 * i + 1)) / (2 * n));
    uNodes[i] = u;
    pts[i] = fromUnit(u, -half, half);
  }

  let t0 = performance.now();
  const vals = new Float64Array(n * n * n);
  let fMin = Infinity;
  let fMax = -Infinity;
  for (let ix = 0; ix < n; ix++) {
    for (let iy = 0; iy < n; iy++) {
      for (let iz = 0; iz < n; iz++) {
        const v = fn(pts[ix], pts[iy], pts[iz]);
        if (!Number.isFinite(v)) throw new Error(`f NaN at sample (${pts[ix]}, ${pts[iy]}, ${pts[iz]})`);
        vals[ix + iy * n + iz * n * n] = v;
        fMin = Math.min(fMin, v);
        fMax = Math.max(fMax, v);
      }
    }
  }
  const sampleMs = performance.now() - t0;

  t0 = performance.now();
  // Separable DCT: O(n⁴). Mutates vals as scratch after the X-pass buffer.
  const cheb = chebDCT3DSeparable(vals, n, uNodes);
  const chebMs = performance.now() - t0;

  t0 = performance.now();
  const mono = chebToMonomial3D(cheb, N, half);
  const monoMs = performance.now() - t0;

  let fitRelL2 = NaN;
  let l2Ms = 0;
  if (!opts.skipL2) {
    t0 = performance.now();
    const M = 10;
    let num = 0;
    let den = 0;
    for (let ix = 0; ix < M; ix++) {
      for (let iy = 0; iy < M; iy++) {
        for (let iz = 0; iz < M; iz++) {
          const x = -half + (2 * half * (ix + 0.5)) / M;
          const y = -half + (2 * half * (iy + 0.5)) / M;
          const z = -half + (2 * half * (iz + 0.5)) / M;
          const truth = fn(x, y, z);
          const approx = evalMonomial3D(mono, N, x, y, z);
          const d = approx - truth;
          num += d * d;
          den += truth * truth;
        }
      }
    }
    fitRelL2 = Math.sqrt(num) / (Math.sqrt(den) + 1e-15);
    l2Ms = performance.now() - t0;
  }

  const totalMs = performance.now() - tAll;
  return {
    cheb,
    mono,
    deg: N,
    half,
    fitRelL2,
    fMin,
    fMax,
    timing: {
      sampleMs,
      chebMs,
      monoMs,
      l2Ms,
      totalMs,
    },
  };
}
