import { isCuratedShortcutsEnabled } from "../../app/curatedShortcutsState.js";

/** Thin space inserted when the user presses space in math mode. */
export const LAPLACI_MATH_MODE_SPACE = "\\,";

/** Default MathLive inline-shortcut timing (ms between keys in a sequence). */
export const LAPLACI_INLINE_SHORTCUT_TIMEOUT = 750;

/** Laplaci vector-calculus macros merged into every math-field. */
export const LAPLACI_MATH_MACROS: Record<string, string> = {
  grad: "\\nabla",
  del: "\\nabla",
  laplacian: "\\operatorname{laplacian}",
  div: "\\operatorname{div}",
  curl: "\\operatorname{curl}",
  partial: "\\partial",
  int: "\\int",
};

export type InlineShortcutDefinition =
  | string
  | {
      value: string;
      after?: string;
    };

export type InlineShortcutMap = Record<string, InlineShortcutDefinition>;

/** Shortcuts removed or replaced when the curated whitelist is active. */
export const CURATED_SHORTCUT_OVERRIDES: InlineShortcutMap = {
  xx: {
    after: "digit+binop+relop+punct+openfence+nothing+space",
    value: "\\times",
  },
};

/** Default inline shortcuts suppressed — laplaci macros or literal typing handle these. */
export const CURATED_SHORTCUT_SUPPRESS = ["grad", "del"] as const;

/** Merge MathLive defaults with laplaci curated overrides. Pure for unit tests. */
export function buildCuratedInlineShortcuts(
  defaults: InlineShortcutMap = {},
): InlineShortcutMap {
  const merged: InlineShortcutMap = { ...defaults, ...CURATED_SHORTCUT_OVERRIDES };
  for (const key of CURATED_SHORTCUT_SUPPRESS) {
    delete merged[key];
  }
  return merged;
}

function applyInlineShortcutPolicy(mf: MathfieldElement): void {
  mf.inlineShortcutTimeout = LAPLACI_INLINE_SHORTCUT_TIMEOUT;
  if (!isCuratedShortcutsEnabled()) return;

  const defaults =
    mf.inlineShortcuts && typeof mf.inlineShortcuts === "object"
      ? ({ ...mf.inlineShortcuts } as InlineShortcutMap)
      : {};
  mf.inlineShortcuts = buildCuratedInlineShortcuts(defaults);
}

function applyMathfieldMacros(mf: MathfieldElement): void {
  mf.macros = {
    ...mf.macros,
    ...LAPLACI_MATH_MACROS,
  };
}

/** Apply laplaci keyboard / shortcut policy to a MathLive field. */
export function applyMathfieldInputPolicy(mf: MathfieldElement, label = "Math expression"): void {
  mf.setAttribute("aria-label", label);
  const syncKeyboardSinkLabel = () => {
    const sink = mf.shadowRoot?.querySelector<HTMLElement>(".ML__keyboard-sink");
    if (sink && sink.getAttribute("aria-label") !== label) {
      sink.setAttribute("aria-label", label);
    }
  };
  syncKeyboardSinkLabel();
  queueMicrotask(syncKeyboardSinkLabel);
  requestAnimationFrame(syncKeyboardSinkLabel);
  mf.addEventListener("focusin", syncKeyboardSinkLabel);

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
    mf.smartMode = false;
  } catch {
    /* ignore */
  }
  try {
    mf.mathModeSpace = LAPLACI_MATH_MODE_SPACE;
  } catch {
    /* ignore */
  }

  applyMathfieldMacros(mf);
  applyInlineShortcutPolicy(mf);

  try {
    mf.menuItems = [];
  } catch {
    /* ignore */
  }
}

/** Re-apply input policy after dev toggle changes (existing LaTeX is preserved). */
export function reconfigureAllMathfields(root: ParentNode = document): void {
  for (const mf of root.querySelectorAll("math-field")) {
    const label = mf.getAttribute("aria-label") || "Math expression";
    applyMathfieldInputPolicy(mf, label);
  }
}
