/** @typedef {{ name: string; keys: string; expectLatex: string | RegExp; expectNotLatex?: RegExp; curatedOnly?: boolean; rawOnly?: boolean; delay?: number }} KeystrokeCase */

/** @type {KeystrokeCase[]} */
export const KEYSTROKE_CASES = [
  // --- Spaces & implicit multiplication (core laplaci patterns) ---
  {
    name: "space separates b and x",
    keys: "b x",
    expectLatex: /b.*x/i,
  },
  {
    name: "parametric plane a x + b y",
    keys: "a x + b y",
    expectLatex: /a.*x.*\+.*b.*y/i,
  },
  {
    name: "coefficient 2 x",
    keys: "2 x",
    expectLatex: /2.*x/i,
  },
  {
    name: "mixed terms a x^2 + b y",
    keys: "a x^2 + b y",
    expectLatex: /a.*x.*2.*\+.*b.*y/i,
  },
  {
    name: "sin x with implicit argument",
    keys: "sin x",
    expectLatex: /\\sin.*x|sin.*x/i,
  },
  {
    name: "cos y with implicit argument",
    keys: "cos y",
    expectLatex: /\\cos.*y|cos.*y/i,
  },
  {
    name: "sin(x) with explicit parens",
    keys: "sin(x)",
    expectLatex: /\\sin.*x/i,
  },

  // --- Polynomials, exponentials, radial fields ---
  {
    name: "quadratic x^2+y^2",
    keys: "x^2+y^2",
    expectLatex: /x.*\^?\{?2\}?.*\+.*y.*\^?\{?2\}?/i,
  },
  {
    name: "sphere radius squared",
    keys: "x^2+y^2+z^2",
    expectLatex: /x.*2.*\+.*y.*2.*\+.*z.*2/i,
  },
  {
    name: "gaussian exp(-x^2)",
    keys: "exp(-x^2)",
    expectLatex: /exp.*-.*x.*2|\\exp.*-.*x.*2/i,
  },
  {
    name: "euler gaussian e^(-x^2)",
    keys: "e^(-x^2)",
    expectLatex: /e.*\^.*-.*x.*2/i,
  },
  {
    name: "radial decay exp(-r^2)",
    keys: "exp(-r^2)",
    expectLatex: /exp.*-.*r.*2|\\exp.*-.*r.*2/i,
  },

  // --- Greek letters & useful shortcuts ---
  {
    name: "pi shortcut",
    keys: "pi",
    expectLatex: /\\pi/,
  },
  {
    name: "alpha shortcut",
    keys: "alpha",
    expectLatex: /\\alpha/,
  },
  {
    name: "theta shortcut",
    keys: "theta",
    expectLatex: /\\theta/,
  },
  {
    name: "2 pi x wave number",
    keys: "2 pi x",
    expectLatex: /2.*\\pi.*x/i,
  },
  {
    name: "animated sine sin(x+2 pi t)",
    keys: "sin(x+2 pi t)",
    expectLatex: /\\sin.*x.*\\pi.*t/i,
  },

  // --- Relational operators (ASCIIMath shortcuts we keep) ---
  {
    name: "greater equal shortcut",
    keys: ">=",
    expectLatex: /\\ge/,
  },
  {
    name: "less equal shortcut",
    keys: "<=",
    expectLatex: /\\le/,
  },
  {
    name: "not equal shortcut",
    keys: "!=",
    expectLatex: /\\ne/,
  },

  // --- Vector fields & tuples ---
  {
    name: "2d rotation flow (-y,x,0)",
    keys: "(-y,x,0)",
    expectLatex: /-.*y.*,.*x.*,.*0/,
  },
  {
    name: "3d position tuple (x,y,z)",
    keys: "(x,y,z)",
    expectLatex: /x.*,.*y.*,.*z/,
  },
  {
    name: "vector component sum (x,y,x+y)",
    keys: "(x,y,x+y)",
    expectLatex: /x.*,.*y.*,.*x.*\+.*y/,
  },

  // --- Parameter / alias declarations (multi-char lhs avoids shortcut buffers) ---
  {
    name: "alias T=x^2+y^2",
    keys: "T=x^2+y^2",
    expectLatex: /T\s*=.*x.*2.*\+.*y.*2/i,
  },
  {
    name: "function definition f(x)=x^2",
    keys: "f(x)=x^2",
    expectLatex: /f.*x.*=.*x/i,
  },
  {
    name: "parametric polynomial a x^2 + b x + c",
    keys: "a x^2 + b x + c",
    expectLatex: /a.*x.*2.*\+.*b.*x.*\+.*c/i,
  },
  {
    name: "ratio a/b",
    keys: "a/b",
    expectLatex: /a.*\/.*b|\\frac.*a.*b/i,
  },
  {
    name: "clamp x>=0",
    keys: "x>=0",
    expectLatex: /x.*\\ge.*0/i,
  },

  // --- Vector calculus spellings (curated policy keeps macros literal) ---
  {
    name: "grad stays literal under curated policy",
    keys: "grad",
    expectLatex: /grad/i,
    curatedOnly: true,
    expectNotLatex: /\\nabla/,
  },
  {
    name: "del stays literal under curated policy",
    keys: "del",
    expectLatex: /del/i,
    curatedOnly: true,
    expectNotLatex: /\\partial/,
  },
  {
    name: "curl stays literal under curated policy",
    keys: "curl",
    expectLatex: /curl/i,
    curatedOnly: true,
  },
  {
    name: "grad f spell-out",
    keys: "grad f",
    expectLatex: /grad.*f/i,
    curatedOnly: true,
    expectNotLatex: /\\nabla\s*f/i,
  },

  // --- xx → × edge cases (curated policy) ---
  {
    name: "xx after letter stays literal",
    keys: "axx",
    expectLatex: /a.*x.*x|axx/i,
    curatedOnly: true,
    expectNotLatex: /\\times/,
  },
  {
    name: "xxx converts leading xx only (known edge)",
    keys: "xxx",
    expectLatex: /\\times.*x|x.*x.*x/i,
    curatedOnly: true,
  },
  {
    name: "xx at start becomes times",
    keys: "xx",
    expectLatex: /\\times/,
    curatedOnly: true,
  },
  {
    name: "tuple cross product (a,b,c)xx(d,e,f)",
    keys: "(a,b,c)xx(d,e,f)",
    expectLatex: /\\times/,
    curatedOnly: true,
    delay: 25,
  },
  {
    name: "spaced xx between tuples stays literal",
    keys: "(a,b,c) xx (d,e,f)",
    expectLatex: /xx/i,
    expectNotLatex: /\\times/,
    curatedOnly: true,
    delay: 25,
  },
  {
    name: "numeric product 2xx3",
    keys: "2xx3",
    expectLatex: /2.*\\times.*3/,
    curatedOnly: true,
  },

  // --- Common composed surfaces / flows ---
  {
    name: "sin cos product sin(x)cos(y)",
    keys: "sin(x)cos(y)",
    expectLatex: /\\sin.*x.*\\cos.*y/i,
  },
  {
    name: "max clamp max(0,x)",
    keys: "max(0,x)",
    expectLatex: /max.*0.*x/i,
  },
  {
    name: "absolute value abs(x)",
    keys: "abs(x)",
    expectLatex: /abs.*x|\\left\\|.*x|\\vert.*x/i,
  },
  {
    name: "fraction one half 1/2",
    keys: "1/2",
    expectLatex: /frac|\\\/|1.*2/i,
  },
  {
    name: "offset gaussian exp(-4(x^2+y^2))",
    keys: "exp(-4(x^2+y^2))",
    expectLatex: /exp.*-.*4.*x.*2.*y.*2|\\exp.*-.*4.*x.*2.*y.*2/i,
  },
  {
    name: "damped wave sin(t)exp(-t)",
    keys: "sin(t)exp(-t)",
    expectLatex: /\\sin.*t.*exp.*-.*t|\\sin.*t.*\\exp.*-.*t/i,
  },
  {
    name: "negated coordinate -y",
    keys: "-y",
    expectLatex: /-.*y/i,
  },
  {
    name: "curl tuple spell-out",
    keys: "curl(x,y,z)",
    expectLatex: /curl.*x.*y.*z/i,
    curatedOnly: true,
  },
  {
    name: "three-term dot-style sum a x + b y + c z",
    keys: "a x + b y + c z",
    expectLatex: /a.*x.*\+.*b.*y.*\+.*c.*z/i,
  },

  // --- Raw MathLive comparison (dev toggle off) ---
  {
    name: "raw xx after letter may stay literal under Playwright typing",
    keys: "axx",
    expectLatex: /axx|a.*\\times/i,
    rawOnly: true,
  },
];
