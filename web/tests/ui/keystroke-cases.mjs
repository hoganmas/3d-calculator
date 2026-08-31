/** @typedef {{ name: string; keys: string; expectLatex: string | RegExp; curatedOnly?: boolean; rawOnly?: boolean; delay?: number }} KeystrokeCase */

/** @type {KeystrokeCase[]} */
export const KEYSTROKE_CASES = [
  {
    name: "space separates b and x",
    keys: "b x",
    expectLatex: /b.*x/i,
  },
  {
    name: "xx after letter stays literal",
    keys: "axx",
    expectLatex: /a.*x.*x|axx/i,
    curatedOnly: true,
  },
  {
    name: "xx at start becomes times",
    keys: "xx",
    expectLatex: /\\times/,
    curatedOnly: true,
  },
  {
    name: "pi shortcut",
    keys: "pi",
    expectLatex: /\\pi/,
  },
  {
    name: "grad stays literal under curated policy",
    keys: "grad",
    expectLatex: /grad/i,
    curatedOnly: true,
  },
  {
    name: "raw xx after letter may stay literal under Playwright typing",
    keys: "axx",
    expectLatex: /axx|a.*\\times/i,
    rawOnly: true,
  },
];
