import { classifyExpr } from "../../math/fit.js";
import { getParam } from "../../model/params.js";
import type { ExprItem } from "../../types/models.js";

export const ANIM_OPTS_ICON = `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 3.2a1.1 1.1 0 1 0 0 2.2 1.1 1.1 0 0 0 0-2.2Zm0 3.7a1.1 1.1 0 1 0 0 2.2 1.1 1.1 0 0 0 0-2.2Zm0 3.7a1.1 1.1 0 1 0 0 2.2 1.1 1.1 0 0 0 0-2.2Z"/></svg>`;

export const ICON_EYE = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>`;
export const ICON_EYE_OFF = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3l18 18"/><path d="M10.6 10.6a3 3 0 004.2 4.2"/><path d="M9.9 5.1A11 11 0 0112 5c6.5 0 10 7 10 7a18 18 0 01-4.2 4.8"/><path d="M6.1 6.1C4 7.8 2.5 10 2 12c0 0 3.5 7 10 7a11 11 0 005-.9"/></svg>`;

export const ANIM_SPEED_MIN = 0.05;
export const ANIM_SPEED_MAX = 2;
export const ANIM_SPEED_STEP = 0.05;

export function readFieldLatex(mf: MathfieldElement): string {
  return typeof mf.getValue === "function"
    ? String(mf.getValue("latex") || "")
    : String(mf.value || "");
}

export function getCaretPos(mf: MathfieldElement): number {
  const sel = mf.selection;
  if (sel?.ranges?.length) return sel.ranges[0][0] | 0;
  if (typeof mf.position === "number") return mf.position | 0;
  if (typeof mf.selectionStart === "number") return mf.selectionStart | 0;
  return 0;
}

export function getLastOffset(mf: MathfieldElement): number {
  if (typeof mf.lastOffset === "number") return mf.lastOffset | 0;
  const latex = readFieldLatex(mf);
  return Math.max(getCaretPos(mf), latex.length);
}

export function isCursorAtStart(mf: MathfieldElement): boolean {
  const sel = mf.selection;
  if (sel?.ranges?.length) {
    const [a, b] = sel.ranges[0];
    return a === 0 && b === 0;
  }
  return getCaretPos(mf) === 0;
}

export function setCaretPos(mf: MathfieldElement, pos: number) {
  const end = getLastOffset(mf);
  const p = Math.max(0, Math.min(pos | 0, end));
  if (typeof mf.position === "number") {
    mf.position = p;
    return;
  }
  if (mf.selection) {
    try {
      mf.selection = { ranges: [[p, p]] };
    } catch {
      /* ignore */
    }
  }
}

export function setFieldLatex(mf: MathfieldElement, latex: string, silent = true) {
  if (typeof mf.setValue === "function") {
    mf.setValue(latex, silent ? { silenceNotifications: true } : undefined);
  } else {
    mf.value = latex;
  }
}

export function latexAroundCaret(mf: MathfieldElement) {
  const pos = getCaretPos(mf);
  const end = getLastOffset(mf);
  try {
    if (typeof mf.getValue === "function") {
      if (mf.getValue.length >= 2 || end > 0) {
        const left = String(mf.getValue(0, pos, "latex") ?? "");
        const right = String(mf.getValue(pos, end, "latex") ?? "");
        if (!(left === right && left === readFieldLatex(mf) && pos > 0 && pos < end)) {
          return { left, right, pos };
        }
      }
    }
  } catch {
    /* fall through */
  }
  const full = readFieldLatex(mf);
  const marker = "⟦SPLIT⟧";
  const savedPos = pos;
  try {
    if (typeof mf.executeCommand === "function") {
      mf.executeCommand(["insert", marker]);
      const withMark = readFieldLatex(mf);
      const i = withMark.indexOf(marker);
      if (i >= 0) {
        return {
          left: withMark.slice(0, i),
          right: withMark.slice(i + marker.length),
          pos: savedPos,
        };
      }
    }
  } catch {
    /* ignore */
  }
  return { left: full.slice(0, pos), right: full.slice(pos), pos };
}

export function configureMathField(mf: MathfieldElement) {
  mf.setAttribute("math-virtual-keyboard-policy", "manual");
  mf.setAttribute("virtual-keyboard-mode", "off");
  mf.setAttribute("smart-fence", "");
  mf.setAttribute("smart-superscript", "");
  try {
    mf.mathVirtualKeyboardPolicy = "manual";
  } catch {
    /* ignore */
  }
  try {
    mf.macros = {
      ...mf.macros,
      grad: "\\nabla",
      del: "\\nabla",
      laplacian: "\\operatorname{laplacian}",
      div: "\\operatorname{div}",
      curl: "\\operatorname{curl}",
    };
  } catch {
    /* ignore */
  }
  try {
    mf.menuItems = [];
  } catch {
    /* ignore */
  }
  try {
    mf.inlineShortcutTimeout = 2000;
  } catch {
    /* ignore */
  }
}

export function isSuggestionUiActive(mf: MathfieldElement): boolean {
  const panel = document.getElementById("mathlive-suggestion-popover");
  if (panel?.classList.contains("is-visible")) return true;
  try {
    if (mf?.mode === "latex") return true;
  } catch {
    /* ignore */
  }
  return false;
}

export function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1000 || a < 0.01)) return v.toPrecision(3);
  return String(Math.round(v * 1000) / 1000);
}

export function classifyKind(latex: string) {
  try {
    return classifyExpr(latex);
  } catch {
    return null;
  }
}

export function fmtAnimSpeed(speed: number): string {
  const s = Math.round(speed * 100) / 100;
  return Number.isInteger(s) ? String(s) : s.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

/** Params needed under a row: owning `a=…` declaration only. */
export function neededParamForItem(item: ExprItem): string | null {
  const classified = String(item.latex || "").trim() ? classifyKind(item.latex) : null;
  if (classified?.kind === "parameter" && classified.paramName) {
    const p = getParam(classified.paramName);
    if (p && p.exprId === item.id) return classified.paramName;
  }
  return null;
}

export function itemsStructurallyEqual(a: ExprItem[], b: ExprItem[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (
      x.id !== y.id ||
      x.latex !== y.latex ||
      x.enabled !== y.enabled ||
      x.color !== y.color ||
      x.color2 !== y.color2 ||
      JSON.stringify(x.colors) !== JSON.stringify(y.colors)
    ) {
      return false;
    }
  }
  return true;
}
