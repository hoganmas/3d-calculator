import { formatParamLatexValue } from "./fit.js";

/** Compute Engine symbol ids → LaTeX command for parameter declarations. */
const GREEK_CE_TO_LATEX: Record<string, string> = {
  alpha: "\\alpha",
  beta: "\\beta",
  gamma: "\\gamma",
  delta: "\\delta",
  epsilon: "\\epsilon",
  varepsilon: "\\varepsilon",
  zeta: "\\zeta",
  eta: "\\eta",
  theta: "\\theta",
  vartheta: "\\vartheta",
  iota: "\\iota",
  kappa: "\\kappa",
  lambda: "\\lambda",
  mu: "\\mu",
  nu: "\\nu",
  xi: "\\xi",
  pi: "\\pi",
  varpi: "\\varpi",
  rho: "\\rho",
  varrho: "\\varrho",
  sigma: "\\sigma",
  varsigma: "\\varsigma",
  tau: "\\tau",
  upsilon: "\\upsilon",
  phi: "\\phi",
  varphi: "\\varphi",
  chi: "\\chi",
  psi: "\\psi",
  omega: "\\omega",
  Gamma: "\\Gamma",
  Delta: "\\Delta",
  Theta: "\\Theta",
  Lambda: "\\Lambda",
  Xi: "\\Xi",
  Pi: "\\Pi",
  Sigma: "\\Sigma",
  Upsilon: "\\Upsilon",
  Phi: "\\Phi",
  Psi: "\\Psi",
  Omega: "\\Omega",
};

/** LaTeX symbol token for a CE parameter id (`alpha` → `\alpha`). */
export function paramCeIdToLatexSymbol(ceId: string): string {
  const id = String(ceId ?? "").trim();
  if (!id) return id;
  return GREEK_CE_TO_LATEX[id] ?? id;
}

/** `name=value` LaTeX for a parameter row (Greek-aware). */
export function formatParamDefLatex(name: string, value: number): string {
  return `${paramCeIdToLatexSymbol(name)}=${formatParamLatexValue(value)}`;
}

export function isGreekParamCeId(ceId: string): boolean {
  return Object.prototype.hasOwnProperty.call(GREEK_CE_TO_LATEX, ceId);
}

/** Unicode symbol for plain-text UI (tooltips, aria). */
const GREEK_CE_TO_UNICODE: Record<string, string> = {
  alpha: "α",
  beta: "β",
  gamma: "γ",
  delta: "δ",
  epsilon: "ε",
  varepsilon: "ε",
  zeta: "ζ",
  eta: "η",
  theta: "θ",
  vartheta: "ϑ",
  iota: "ι",
  kappa: "κ",
  lambda: "λ",
  mu: "μ",
  nu: "ν",
  xi: "ξ",
  pi: "π",
  varpi: "ϖ",
  rho: "ρ",
  varrho: "ϱ",
  sigma: "σ",
  varsigma: "ς",
  tau: "τ",
  upsilon: "υ",
  phi: "φ",
  varphi: "ϕ",
  chi: "χ",
  psi: "ψ",
  omega: "ω",
  Gamma: "Γ",
  Delta: "Δ",
  Theta: "Θ",
  Lambda: "Λ",
  Xi: "Ξ",
  Pi: "Π",
  Sigma: "Σ",
  Upsilon: "Υ",
  Phi: "Φ",
  Psi: "Ψ",
  Omega: "Ω",
};

export function paramCeIdToPlainSymbol(ceId: string): string {
  const id = String(ceId ?? "").trim();
  if (!id) return id;
  return GREEK_CE_TO_UNICODE[id] ?? id;
}
