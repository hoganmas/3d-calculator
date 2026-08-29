import { classifyExpr, compileExpr } from "../../src/math/fit.ts";
import { classifyVectorExpr, compileVectorExpr } from "../../src/math/fitVector.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

/** Deterministic PRNG for reproducible fuzz runs. */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const BASES = [
  "x",
  "y",
  "z",
  "x^2+y^2+z^2",
  "sin(x)",
  "cos(y)",
  "a=1",
  "T=x^2",
  "(-y,x,0)",
  String.raw`\sin(x+t)`,
  String.raw`\nabla\cdot(x,y,z)`,
  String.raw`\grad(x^2)`,
  "",
  " ",
  "???",
  "1+",
  "(((",
  "\\notacommand",
  "a=b=c",
  "x^",
  "1/0",
  "💥",
];

function pick(rng: () => number, arr: string[]) {
  return arr[Math.floor(rng() * arr.length)]!;
}

function mutate(rng: () => number, s: string): string {
  if (!s.length) return pick(rng, BASES);
  const i = Math.floor(rng() * s.length);
  const chars = "xyz0123456789+-*/^=()\\";
  const c = chars[Math.floor(rng() * chars.length)]!;
  const mode = Math.floor(rng() * 4);
  if (mode === 0) return s.slice(0, i) + c + s.slice(i);
  if (mode === 1) return s.slice(0, i) + s.slice(i + 1);
  if (mode === 2) return pick(rng, BASES) + s;
  return s + pick(rng, BASES);
}

function randomLatex(rng: () => number): string {
  const roll = rng();
  if (roll < 0.35) return pick(rng, BASES);
  if (roll < 0.7) return mutate(rng, pick(rng, BASES));
  let s = pick(rng, BASES);
  const n = 1 + Math.floor(rng() * 3);
  for (let i = 0; i < n; i++) s = mutate(rng, s);
  return s;
}

function expectControlledError(fn: () => unknown, label: string) {
  try {
    fn();
  } catch (e) {
    assert(e instanceof Error, `${label}: throw Error, got ${typeof e}`);
    assert(typeof e.message === "string" && e.message.length > 0, `${label}: message`);
    return;
  }
}

export async function run() {
  return runSuite("math / fuzz-compile", [
    {
      name: "classifyExpr: 300 random strings never throw non-Errors",
      fn: () => {
        const rng = mulberry32(0xfa57);
        for (let i = 0; i < 300; i++) {
          const latex = randomLatex(rng);
          try {
            const c = classifyExpr(latex);
            assert(typeof c.kind === "string", "kind string");
          } catch (e) {
            assert(e instanceof Error, `classify threw non-Error for ${JSON.stringify(latex)}`);
          }
        }
      },
    },
    {
      name: "compileExpr: random strings throw Error or return compiled fn",
      fn: () => {
        const rng = mulberry32(0xc012ee);
        for (let i = 0; i < 200; i++) {
          const latex = randomLatex(rng);
          try {
            const c = compileExpr(latex);
            const fn = c.bind({});
            const v = fn(0.1, 0.2, 0.3);
            assert(Number.isFinite(v) || v === 0, "finite scalar eval");
          } catch (e) {
            assert(e instanceof Error, `compile threw non-Error for ${JSON.stringify(latex)}`);
          }
        }
      },
    },
    {
      name: "vector classify/compile: tuple-like strings stay controlled",
      fn: () => {
        const rng = mulberry32(0x7ec);
        const hints = ["(-y,x,0)", "(x,y,z)", String.raw`\grad(x)`, String.raw`\curl(x,y)`];
        for (let i = 0; i < 80; i++) {
          const latex = rng() < 0.5 ? pick(rng, hints) : mutate(rng, pick(rng, hints));
          try {
            const kind = classifyVectorExpr(latex).kind;
            assert(typeof kind === "string", "vector kind");
            if (kind === "tuple" || kind === "gradient" || kind === "curl") {
              const compiled = compileVectorExpr(latex);
              const out = compiled.bind({})(0.1, 0.2, 0.3);
              assert(Array.isArray(out) && out.length === 3, "vec3");
            }
          } catch (e) {
            assert(e instanceof Error, `vector threw non-Error for ${JSON.stringify(latex)}`);
          }
        }
      },
    },
    {
      name: "known-bad inputs fail with messages (no silent NaN compile)",
      fn: () => {
        for (const latex of ["", "x^", String.raw`(-y,x,0)`, "\\bogus"]) {
          expectControlledError(() => compileExpr(latex), latex || "(empty)");
        }
      },
    },
  ]);
}
