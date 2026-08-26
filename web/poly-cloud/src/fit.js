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

function chebT(k, u) {
  if (k === 0) return 1;
  if (k === 1) return u;
  let t0 = 1;
  let t1 = u;
  for (let j = 2; j <= k; j++) {
    const t2 = 2 * u * t1 - t0;
    t0 = t1;
    t1 = t2;
  }
  return t1;
}

function fromUnit(u, a, b) {
  return 0.5 * (a + b) + 0.5 * (b - a) * u;
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
 */
export function fitChebyshev3D(fn, half, deg) {
  const N = Math.max(0, Math.min(MAX_DEG, deg | 0));
  const n = N + 1;
  const uNodes = new Array(n);
  const pts = new Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.cos((Math.PI * (2 * i + 1)) / (2 * n));
    uNodes[i] = u;
    pts[i] = fromUnit(u, -half, half);
  }

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

  const cheb = new Float32Array(n * n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        let s = 0;
        for (let a = 0; a < n; a++) {
          const Ti = chebT(i, uNodes[a]);
          for (let b = 0; b < n; b++) {
            const Tj = chebT(j, uNodes[b]);
            for (let c = 0; c < n; c++) {
              s += vals[a + b * n + c * n * n] * Ti * Tj * chebT(k, uNodes[c]);
            }
          }
        }
        const ai = i === 0 ? 1 : 2;
        const aj = j === 0 ? 1 : 2;
        const ak = k === 0 ? 1 : 2;
        cheb[i + j * n + k * n * n] = (s * ai * aj * ak) / (n * n * n);
      }
    }
  }

  const mono = chebToMonomial3D(cheb, N, half);

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

  return {
    cheb,
    mono,
    deg: N,
    half,
    fitRelL2: Math.sqrt(num) / (Math.sqrt(den) + 1e-15),
    fMin,
    fMax,
  };
}
