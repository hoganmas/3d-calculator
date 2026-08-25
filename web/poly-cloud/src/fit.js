/** Expression parse + 3D Chebyshev fit → world monomials + camera pullback. */

const FN = new Set(["sin", "cos", "tan", "exp", "log", "ln", "sqrt", "abs", "max", "min", "hypot"]);

export const PRESETS = {
  blob: {
    label: "Gaussian blob",
    expr: "exp(-(x^2+y^2+z^2))",
    deg: 4,
    scale: 2.5,
    half: 2,
  },
  soft: {
    label: "Soft ellipsoid",
    expr: "exp(-(x^2+0.5*y^2+2*z^2))",
    deg: 4,
    scale: 3,
    half: 2,
  },
  two: {
    label: "Two blobs",
    expr: "exp(-4*((x-0.7)^2+y^2+z^2))+exp(-4*((x+0.7)^2+y^2+z^2))",
    deg: 5,
    scale: 2.2,
    half: 2,
  },
  shell: {
    label: "Spherical shell",
    expr: "exp(-12*(sqrt(x^2+y^2+z^2)-0.9)^2)",
    deg: 5,
    scale: 3,
    half: 1.8,
  },
  ridge: {
    label: "Vertical ridge",
    expr: "exp(-10*x^2)*exp(-0.4*(y^2+z^2))",
    deg: 4,
    scale: 4,
    half: 2,
  },
};

function normalizeLatex(s) {
  let x = s.trim();
  x = x.replace(/^\$+|\$+$/g, "");
  x = x.replace(/\\\[|\\\]/g, "");
  x = x.replace(/\\\(|\\\)/g, "");
  x = x.replace(/\\left|\\right/g, "");
  x = x.replace(/\\,/g, "");
  x = x.replace(/\\ |~/g, "");
  x = x.replace(/\\times|\\cdot|\\ast|·|×/g, "*");
  x = x.replace(/\^{([^{}]+)}/g, "^$1");
  x = x.replace(/\^\{([^{}]+)\}/g, "^$1");
  for (let k = 0; k < 8; k++) {
    const next = x.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "(($1)/($2))");
    if (next === x) break;
    x = next;
  }
  const map = [
    [/\\sin/g, "sin"],
    [/\\cos/g, "cos"],
    [/\\tan/g, "tan"],
    [/\\exp/g, "exp"],
    [/\\log/g, "log"],
    [/\\ln/g, "ln"],
    [/\\sqrt/g, "sqrt"],
    [/\\abs/g, "abs"],
    [/\\max/g, "max"],
    [/\\min/g, "min"],
    [/\\pi/g, "pi"],
  ];
  for (const [re, rep] of map) x = x.replace(re, rep);
  x = x.replace(/π/g, "pi");
  x = x.replace(/\\([a-zA-Z]+)/g, "$1");
  x = x.replace(/\\/g, "");
  x = x.replace(/\s+/g, "");
  return x;
}

function insertImplicitMult(s) {
  let x = s;
  x = x.replace(/(\d)([a-zA-Z(])/g, "$1*$2");
  x = x.replace(/(\))(\()/g, "$1*$2");
  x = x.replace(/(\))([a-zA-Z0-9])/g, "$1*$2");
  x = x.replace(/([a-zA-Z0-9_]+)\(/g, (m, id) => (FN.has(id) ? m : `${id}*(`));
  return x;
}

function tokenize(src) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (/[0-9.]/.test(ch)) {
      const start = i++;
      while (i < src.length && /[0-9.eE]/.test(src[i])) i++;
      if (src[i] === "e" || src[i] === "E") {
        i++;
        if (src[i] === "+" || src[i] === "-") i++;
        while (i < src.length && /[0-9]/.test(src[i])) i++;
      }
      tokens.push({ kind: "num", text: src.slice(start, i) });
      continue;
    }
    if (/[a-zA-Z_]/.test(ch)) {
      const start = i++;
      while (i < src.length && /[a-zA-Z0-9_]/.test(src[i])) i++;
      tokens.push({ kind: "ident", text: src.slice(start, i) });
      continue;
    }
    if ("+-*/^".includes(ch)) {
      tokens.push({ kind: "op", text: ch });
      i++;
      continue;
    }
    if (ch === "(") {
      tokens.push({ kind: "lp", text: ch });
      i++;
      continue;
    }
    if (ch === ")") {
      tokens.push({ kind: "rp", text: ch });
      i++;
      continue;
    }
    if (ch === ",") {
      tokens.push({ kind: "comma", text: ch });
      i++;
      continue;
    }
    throw new Error(`Unexpected “${ch}”`);
  }
  tokens.push({ kind: "eof", text: "" });
  return tokens;
}

class Parser {
  constructor(toks) {
    this.toks = toks;
    this.i = 0;
  }
  peek() {
    return this.toks[this.i];
  }
  eat(kind) {
    const t = this.peek();
    if (kind && t.kind !== kind) throw new Error(`Expected ${kind}, got “${t.text}”`);
    this.i++;
    return t;
  }
  parse() {
    const e = this.expr();
    if (this.peek().kind !== "eof") throw new Error(`Unexpected “${this.peek().text}”`);
    return e;
  }
  expr() {
    let left = this.term();
    while (this.peek().kind === "op" && (this.peek().text === "+" || this.peek().text === "-")) {
      const op = this.eat("op").text;
      left = { kind: "bin", op, left, right: this.term() };
    }
    return left;
  }
  term() {
    let left = this.power();
    while (this.peek().kind === "op" && (this.peek().text === "*" || this.peek().text === "/")) {
      const op = this.eat("op").text;
      left = { kind: "bin", op, left, right: this.power() };
    }
    return left;
  }
  power() {
    let left = this.unary();
    if (this.peek().kind === "op" && this.peek().text === "^") {
      this.eat("op");
      left = { kind: "bin", op: "^", left, right: this.power() };
    }
    return left;
  }
  unary() {
    if (this.peek().kind === "op" && this.peek().text === "-") {
      this.eat("op");
      return { kind: "unary", op: "-", arg: this.unary() };
    }
    if (this.peek().kind === "op" && this.peek().text === "+") {
      this.eat("op");
      return this.unary();
    }
    return this.primary();
  }
  primary() {
    const t = this.peek();
    if (t.kind === "num") {
      this.eat("num");
      return { kind: "num", v: Number(t.text) };
    }
    if (t.kind === "ident") {
      const name = this.eat("ident").text;
      if (this.peek().kind === "lp") {
        if (!FN.has(name)) throw new Error(`Unknown function “${name}”`);
        this.eat("lp");
        const args = [];
        if (this.peek().kind !== "rp") {
          args.push(this.expr());
          while (this.peek().kind === "comma") {
            this.eat("comma");
            args.push(this.expr());
          }
        }
        this.eat("rp");
        return { kind: "call", fn: name, args };
      }
      return { kind: "ident", name };
    }
    if (t.kind === "lp") {
      this.eat("lp");
      const e = this.expr();
      this.eat("rp");
      return e;
    }
    throw new Error(`Unexpected “${t.text}”`);
  }
}

function realPow(a, b) {
  if (Number.isInteger(b)) {
    const n = b | 0;
    if (n === 0) return 1;
    let r = 1;
    const base = n > 0 ? a : 1 / a;
    const k = Math.abs(n);
    for (let i = 0; i < k; i++) r *= base;
    return r;
  }
  if (a < 0) return NaN;
  return Math.pow(a, b);
}

function evalAst(ast, ctx) {
  switch (ast.kind) {
    case "num":
      return ast.v;
    case "ident": {
      const n = ast.name.toLowerCase();
      if (n === "x") return ctx.x;
      if (n === "y") return ctx.y;
      if (n === "z") return ctx.z;
      if (n === "r") return Math.hypot(ctx.x, ctx.y, ctx.z);
      if (n === "pi") return Math.PI;
      if (n === "e") return Math.E;
      throw new Error(`Unknown identifier “${ast.name}”`);
    }
    case "unary":
      return -evalAst(ast.arg, ctx);
    case "bin": {
      const a = evalAst(ast.left, ctx);
      const b = evalAst(ast.right, ctx);
      if (ast.op === "+") return a + b;
      if (ast.op === "-") return a - b;
      if (ast.op === "*") return a * b;
      if (ast.op === "/") return a / b;
      return realPow(a, b);
    }
    case "call": {
      const args = ast.args.map((a) => evalAst(a, ctx));
      switch (ast.fn) {
        case "sin":
          return Math.sin(args[0]);
        case "cos":
          return Math.cos(args[0]);
        case "tan":
          return Math.tan(args[0]);
        case "exp":
          return Math.exp(Math.max(-80, Math.min(80, args[0])));
        case "log":
        case "ln":
          return Math.log(args[0]);
        case "sqrt":
          return Math.sqrt(Math.max(0, args[0]));
        case "abs":
          return Math.abs(args[0]);
        case "max":
          return Math.max(...args);
        case "min":
          return Math.min(...args);
        case "hypot":
          return Math.hypot(...args);
        default:
          throw new Error(ast.fn);
      }
    }
    default:
      throw new Error("bad ast");
  }
}

export function compileExpr(raw) {
  const norm = insertImplicitMult(normalizeLatex(raw));
  if (!norm) throw new Error("Empty expression");
  const ast = new Parser(tokenize(norm)).parse();
  return (x, y, z) => evalAst(ast, { x, y, z });
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
export function chebToMonomial3D(chebCoeffs, deg, half) {
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
              // T_i(x/h) contrib Ti[a]*(x/h)^a
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

function binomInt(n, k) {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  if (k > n - k) k = n - k;
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}

function intPow(b, e) {
  let r = 1;
  for (let i = 0; i < e; i++) r *= b;
  return r;
}

/**
 * Change of origin to the camera: f_cam(p) = f_world(p + o).
 * Then rays are p(t)=t d (camera at 0) and α_m = Σ_{i+j+k=m} c_ijk d_x^i d_y^j d_z^k.
 */
export function translateMonomial3D(mono, deg, ox, oy, oz) {
  const n = deg + 1;
  const out = new Float64Array(n * n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        const c = mono[i + j * n + k * n * n];
        if (Math.abs(c) < 1e-18) continue;
        for (let a = 0; a <= i; a++) {
          const wx = binomInt(i, a) * intPow(ox, i - a);
          for (let b = 0; b <= j; b++) {
            const wy = binomInt(j, b) * intPow(oy, j - b);
            for (let d = 0; d <= k; d++) {
              const wz = binomInt(k, d) * intPow(oz, k - d);
              out[a + b * n + d * n * n] += c * wx * wy * wz;
            }
          }
        }
      }
    }
  }
  return Float32Array.from(out);
}

/**
 * f_cam(p) = f_world(R p + t), with R 3×3 row-major [r00,r01,r02,r10,...], t vec3.
 * Output is a dense tensor-product of degree outDeg >= deg (use 3*deg).
 * Packing: idx = i + j*(outDeg+1) + k*(outDeg+1)^2 for u^i v^j w^k.
 */
export function pullbackAffine(monoWorld, deg, R, t, outDeg) {
  const nIn = deg + 1;
  const nOut = outDeg + 1;
  const out = new Float64Array(nOut * nOut * nOut);

  // Dense poly multiply by linear form L = a u + b v + c w + d
  const mulL = (poly, a, b, c, d) => {
    const next = new Float64Array(nOut * nOut * nOut);
    for (let i = 0; i < nOut; i++) {
      for (let j = 0; j < nOut; j++) {
        for (let k = 0; k < nOut; k++) {
          const v = poly[i + j * nOut + k * nOut * nOut];
          if (v === 0) continue;
          if (d) next[i + j * nOut + k * nOut * nOut] += v * d;
          if (a && i + 1 < nOut) next[i + 1 + j * nOut + k * nOut * nOut] += v * a;
          if (b && j + 1 < nOut) next[i + (j + 1) * nOut + k * nOut * nOut] += v * b;
          if (c && k + 1 < nOut) next[i + j * nOut + (k + 1) * nOut * nOut] += v * c;
        }
      }
    }
    return next;
  };

  const powLinear = (a, b, c, d, power) => {
    let p = new Float64Array(nOut * nOut * nOut);
    p[0] = 1;
    for (let m = 0; m < power; m++) p = mulL(p, a, b, c, d);
    return p;
  };

  const mulPoly = (A, B) => {
    const next = new Float64Array(nOut * nOut * nOut);
    for (let i1 = 0; i1 < nOut; i1++) {
      for (let j1 = 0; j1 < nOut; j1++) {
        for (let k1 = 0; k1 < nOut; k1++) {
          const va = A[i1 + j1 * nOut + k1 * nOut * nOut];
          if (va === 0) continue;
          for (let i2 = 0; i2 < nOut - i1; i2++) {
            for (let j2 = 0; j2 < nOut - j1; j2++) {
              for (let k2 = 0; k2 < nOut - k1; k2++) {
                const vb = B[i2 + j2 * nOut + k2 * nOut * nOut];
                if (vb === 0) continue;
                next[i1 + i2 + (j1 + j2) * nOut + (k1 + k2) * nOut * nOut] += va * vb;
              }
            }
          }
        }
      }
    }
    return next;
  };

  // Cache powers of Lx, Ly, Lz
  const Lx = [R[0], R[1], R[2], t[0]];
  const Ly = [R[3], R[4], R[5], t[1]];
  const Lz = [R[6], R[7], R[8], t[2]];
  const powX = [];
  const powY = [];
  const powZ = [];
  for (let p = 0; p < nIn; p++) {
    powX[p] = powLinear(Lx[0], Lx[1], Lx[2], Lx[3], p);
    powY[p] = powLinear(Ly[0], Ly[1], Ly[2], Ly[3], p);
    powZ[p] = powLinear(Lz[0], Lz[1], Lz[2], Lz[3], p);
  }

  for (let i = 0; i < nIn; i++) {
    for (let j = 0; j < nIn; j++) {
      for (let k = 0; k < nIn; k++) {
        const c = monoWorld[i + j * nIn + k * nIn * nIn];
        if (Math.abs(c) < 1e-18) continue;
        let term = mulPoly(powX[i], powY[j]);
        term = mulPoly(term, powZ[k]);
        for (let idx = 0; idx < out.length; idx++) out[idx] += c * term[idx];
      }
    }
  }

  return Float32Array.from(out);
}

/**
 * Fit f on [-half,half]^3 with tensor Chebyshev, convert to world monomials.
 */
export function fitChebyshev3D(fn, half, deg) {
  const N = Math.max(0, Math.min(8, deg | 0)); // cap 8 for shader
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
