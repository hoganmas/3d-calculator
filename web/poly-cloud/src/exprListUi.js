/**
 * Named parameter expression list UI (left rail).
 * Behaves like a multi-line text blob: Enter splits, Backspace merges,
 * Up/Down move the caret between rows.
 */

import {
  listExpressions,
  getSelectedId,
  selectExpr,
  removeExpr,
  mergeExprIntoPrevious,
  splitExprAt,
  updateExpr,
} from "./expressions.js";

function readFieldLatex(mf) {
  return typeof mf.getValue === "function" ? String(mf.getValue("latex") || "") : String(mf.value || "");
}

function getCaretPos(mf) {
  const sel = mf.selection;
  if (sel?.ranges?.length) return sel.ranges[0][0] | 0;
  if (typeof mf.position === "number") return mf.position | 0;
  if (typeof mf.selectionStart === "number") return mf.selectionStart | 0;
  return 0;
}

function getLastOffset(mf) {
  if (typeof mf.lastOffset === "number") return mf.lastOffset | 0;
  const latex = readFieldLatex(mf);
  // Fallback: treat as end when lastOffset unavailable.
  return Math.max(getCaretPos(mf), latex.length);
}

/** True when the caret is collapsed at the start of the field. */
function isCursorAtStart(mf) {
  const sel = mf.selection;
  if (sel?.ranges?.length) {
    const [a, b] = sel.ranges[0];
    return a === 0 && b === 0;
  }
  return getCaretPos(mf) === 0;
}

function focusFieldAt(root, id, offset) {
  const mf = root.querySelector(`.expr-row[data-id="${id}"] math-field`);
  if (!mf) return;
  selectExpr(id);
  root.querySelectorAll(".expr-row").forEach((r) => {
    r.classList.toggle("selected", r.dataset.id === id);
  });
  mf.focus?.();
  const end = getLastOffset(mf);
  const pos = Math.max(0, Math.min(offset | 0, end));
  if (typeof mf.position === "number") {
    mf.position = pos;
    return;
  }
  if (mf.selection) {
    try {
      mf.selection = { ranges: [[pos, pos]] };
    } catch (_) {
      /* ignore */
    }
  }
}

/** Latex left/right of the caret using MathLive range getValue when available. */
function latexAroundCaret(mf) {
  const pos = getCaretPos(mf);
  const end = getLastOffset(mf);
  try {
    if (typeof mf.getValue === "function") {
      // MathLive overload: getValue(start, end, format)
      if (mf.getValue.length >= 2 || end > 0) {
        const left = String(mf.getValue(0, pos, "latex") ?? "");
        const right = String(mf.getValue(pos, end, "latex") ?? "");
        // Sanity: if range API ignored args and returned full latex twice, fall through.
        if (!(left === right && left === readFieldLatex(mf) && pos > 0 && pos < end)) {
          return { left, right, pos };
        }
      }
    }
  } catch (_) {
    /* fall through */
  }
  // Marker fallback for builds without range getValue.
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
  } catch (_) {
    /* ignore */
  }
  // Last resort: latex-string index (imperfect for complex latex).
  return { left: full.slice(0, pos), right: full.slice(pos), pos };
}

function configureMathField(mf) {
  mf.setAttribute("math-virtual-keyboard-policy", "manual");
  mf.setAttribute("virtual-keyboard-mode", "off");
  mf.setAttribute("smart-fence", "");
  mf.setAttribute("smart-superscript", "");
  try {
    mf.mathVirtualKeyboardPolicy = "manual";
  } catch (_) {
    /* ignore */
  }
  try {
    mf.menuItems = [];
  } catch (_) {
    /* ignore */
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

  function render() {
    const items = listExpressions();
    const selected = getSelectedId();
    root.replaceChildren();

    for (const item of items) {
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
      configureMathField(mf);
      mf.className = "expr-field";
      if (typeof mf.setValue === "function") mf.setValue(item.latex || "", { silenceNotifications: true });
      else mf.value = item.latex || "";

      mf.addEventListener("focus", () => {
        selectExpr(item.id);
        root.querySelectorAll(".expr-row").forEach((r) => {
          r.classList.toggle("selected", r.dataset.id === item.id);
        });
      });
      mf.addEventListener("input", () => {
        updateExpr(item.id, { latex: readFieldLatex(mf) });
        onExprChange();
      });
      mf.addEventListener(
        "keydown",
        (ev) => {
          if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
            const list = listExpressions();
            const idx = list.findIndex((e) => e.id === item.id);
            const target = ev.key === "ArrowUp" ? list[idx - 1] : list[idx + 1];
            if (target) {
              ev.preventDefault();
              ev.stopPropagation();
              const pos = getCaretPos(mf);
              updateExpr(item.id, { latex: readFieldLatex(mf) });
              focusFieldAt(root, target.id, pos);
              return;
            }
          }

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
            const { left, right } = latexAroundCaret(mf);
            updateExpr(item.id, { latex: left });
            const split = splitExprAt(item.id, left, right);
            onStructuralChange();
            render();
            if (split) {
              queueMicrotask(() => focusFieldAt(root, split.id, 0));
            }
          }
        },
        true,
      );

      mid.append(mf);

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
        if (ev.target === mf || mf.contains?.(ev.target)) return;
        selectExpr(item.id);
        root.querySelectorAll(".expr-row").forEach((r) => {
          r.classList.toggle("selected", r.dataset.id === item.id);
        });
        mf.focus?.();
      });

      row.append(swatch, mid, del);
      root.appendChild(row);
    }
  }

  return { render };
}
