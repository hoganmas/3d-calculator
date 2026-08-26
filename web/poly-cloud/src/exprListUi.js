/**
 * Named parameter expression list UI (left rail).
 */

import {
  listExpressions,
  getSelectedId,
  selectExpr,
  insertExprAfter,
  removeExpr,
  mergeExprIntoPrevious,
  updateExpr,
  resolveExprRole,
} from "./expressions.js";
import { classifyExpr } from "./fit.js";

function readFieldLatex(mf) {
  return typeof mf.getValue === "function" ? String(mf.getValue("latex") || "") : String(mf.value || "");
}

/** True when the caret is collapsed at the start of the field. */
function isCursorAtStart(mf) {
  const sel = mf.selection;
  if (sel?.ranges?.length) {
    const [a, b] = sel.ranges[0];
    return a === 0 && b === 0;
  }
  if (typeof mf.position === "number") return mf.position === 0;
  if (typeof mf.selectionStart === "number") {
    return mf.selectionStart === 0 && mf.selectionEnd === 0;
  }
  return false;
}

function focusFieldAt(root, id, offset) {
  const mf = root.querySelector(`.expr-row[data-id="${id}"] math-field`);
  if (!mf) return;
  mf.focus?.();
  if (typeof mf.position === "number") {
    mf.position = offset;
    return;
  }
  if (typeof mf.setSelectionRange === "function") {
    mf.setSelectionRange(offset, offset);
  }
}

/**
 * @param {{
 *   root: HTMLElement,
 *   onExprChange: () => void,
 *   onStructuralChange: () => void,
 *   onColorChange?: () => void,
 * }} opts
 */
export function mountExprList(opts) {
  const { root, onExprChange, onStructuralChange, onColorChange } = opts;
  /** @type {Map<string, HTMLElement>} */
  const rowEls = new Map();

  function roleLabel(role, kind) {
    const r = resolveExprRole(role, kind);
    return r === "constraint" ? "manifold" : "density";
  }

  function classifySafe(latex) {
    try {
      if (!String(latex || "").trim()) return { kind: "bare", label: "empty" };
      return classifyExpr(latex);
    } catch {
      return { kind: "bare", label: "invalid" };
    }
  }

  function render() {
    const items = listExpressions();
    const selected = getSelectedId();
    root.replaceChildren();
    rowEls.clear();

    for (const item of items) {
      const meta = classifySafe(item.latex);
      const row = document.createElement("div");
      row.className = "expr-row" + (item.id === selected ? " selected" : "");
      row.dataset.id = item.id;

      const swatch = document.createElement("input");
      swatch.type = "color";
      swatch.className = "expr-color";
      swatch.value = item.color.startsWith("#") ? item.color : "#2d70b3";
      swatch.title = "Color";
      swatch.addEventListener("input", () => {
        updateExpr(item.id, { color: swatch.value });
        if (onColorChange) onColorChange();
        else onExprChange();
      });

      const mid = document.createElement("div");
      mid.className = "expr-mid";

      const mf = document.createElement("math-field");
      mf.setAttribute("math-virtual-keyboard-policy", "sandboxed");
      mf.setAttribute("smart-fence", "");
      mf.setAttribute("smart-superscript", "");
      mf.className = "expr-field";
      if (typeof mf.setValue === "function") mf.setValue(item.latex || "", { silenceNotifications: true });
      else mf.value = item.latex || "";

      mf.addEventListener("focus", () => selectExpr(item.id));
      mf.addEventListener("input", () => {
        const latex = readFieldLatex(mf);
        updateExpr(item.id, { latex });
        badge.textContent = roleLabel(item.role, classifySafe(latex).kind);
        onExprChange();
      });
      mf.addEventListener("keydown", (ev) => {
        if (ev.key === "Backspace" && isCursorAtStart(mf)) {
          const idx = listExpressions().findIndex((e) => e.id === item.id);
          if (idx > 0) {
            ev.preventDefault();
            ev.stopPropagation();
            updateExpr(item.id, { latex: readFieldLatex(mf) });
            const merged = mergeExprIntoPrevious(item.id);
            if (merged) {
              onStructuralChange();
              render();
              queueMicrotask(() => focusFieldAt(root, merged.id, merged.caretOffset));
            }
            return;
          }
        }
        if (ev.key === "Enter" && !ev.shiftKey) {
          ev.preventDefault();
          ev.stopPropagation();
          insertExprAfter(item.id, { latex: "" });
          onStructuralChange();
          render();
          queueMicrotask(() => {
            const next = root.querySelector(`.expr-row[data-id="${getSelectedId()}"] math-field`);
            next?.focus?.();
          });
        }
      });

      const badge = document.createElement("button");
      badge.type = "button";
      badge.className = "expr-badge";
      badge.textContent = roleLabel(item.role, meta.kind);
      badge.title = "Cycle: auto → density → manifold";
      badge.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const order = ["auto", "density", "constraint"];
        const i = order.indexOf(item.role);
        const next = order[(i + 1) % order.length];
        updateExpr(item.id, { role: next });
        badge.textContent = roleLabel(next, classifySafe(item.latex).kind);
        onExprChange();
      });

      mid.append(mf, badge);

      const del = document.createElement("button");
      del.type = "button";
      del.className = "expr-del secondary";
      del.textContent = "×";
      del.title = "Delete";
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();
        removeExpr(item.id);
        onStructuralChange();
        render();
      });

      row.addEventListener("click", (ev) => {
        if (ev.target === del || ev.target === swatch) return;
        selectExpr(item.id);
        render();
      });

      row.append(swatch, mid, del);
      root.appendChild(row);
      rowEls.set(item.id, row);
    }
  }

  return { render };
}
