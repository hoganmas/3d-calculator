/**
 * Expression list UI (left rail).
 * Parameter rows (`a = …`) get a slider + play controls under the math field.
 * Enter splits, Backspace merges, Up/Down move between rows.
 */

import {
  listExpressions,
  getSelectedId,
  selectExpr,
  removeExpr,
  mergeExprIntoPrevious,
  splitExprAt,
  updateExpr,
  updateExprSilent,
  getExprWarning,
  commitAutoParams,
} from "./expressions.js";
import { classifyExpr } from "./fit.js";
import {
  getParam,
  listParamNames,
  setParamValue,
  updateParam,
  toggleParamAnimate,
  stopParamAnimation,
} from "./params.js";

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
  return Math.max(getCaretPos(mf), latex.length);
}

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

function latexAroundCaret(mf) {
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
  } catch (_) {
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
  } catch (_) {
    /* ignore */
  }
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

function fmtNum(v) {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1000 || a < 0.01)) return v.toPrecision(3);
  return String(Math.round(v * 1000) / 1000);
}

function classifyKind(latex) {
  try {
    return classifyExpr(latex);
  } catch {
    return null;
  }
}

const ICON_EYE = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>`;
const ICON_EYE_OFF = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 3l18 18"/><path d="M10.6 10.6a3 3 0 004.2 4.2"/><path d="M9.9 5.1A11 11 0 0112 5c6.5 0 10 7 10 7a18 18 0 01-4.2 4.8"/><path d="M6.1 6.1C4 7.8 2.5 10 2 12c0 0 3.5 7 10 7a11 11 0 005-.9"/></svg>`;

/**
 * @param {{
 *   root: HTMLElement,
 *   onExprChange: () => void,
 *   onStructuralChange: () => void,
 *   onColorChange?: () => void,
 *   onParamChange?: () => void,
 * }} opts
 */
export function mountExprList(opts) {
  const { root, onExprChange, onStructuralChange, onColorChange, onParamChange } = opts;
  const paramNotify = onParamChange || onExprChange;

  /** >0 while list DOM is rebuilt / focus restored — ignore blur commits. */
  let suppressAutoCommit = 0;

  function beginSuppressAutoCommit() {
    suppressAutoCommit++;
  }

  function endSuppressAutoCommit() {
    queueMicrotask(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          suppressAutoCommit = Math.max(0, suppressAutoCommit - 1);
        });
      });
    });
  }

  function focusedExprIdInList() {
    const snap = captureFocus();
    return snap?.id ?? null;
  }

  /**
   * Commit ephemeral auto-params only after a real leave (other row or outside list).
   * Re-render blur+restore must not commit, or prune never sees autoParam.
   */
  function scheduleCommitIfLeftExpr(fromExprId) {
    queueMicrotask(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (suppressAutoCommit) return;
          const focusedId = focusedExprIdInList();
          if (focusedId === fromExprId) return;
          commitAutoParams();
        });
      });
    });
  }

  function syncParamSlider(row, name) {
    const p = getParam(name);
    if (!p || !row) return;
    const block = row.querySelector(`[data-param-block="${CSS.escape(name)}"]`) || row;
    const slider = block.querySelector(".expr-param-slider");
    const minEl = block.querySelector(".expr-param-min");
    const maxEl = block.querySelector(".expr-param-max");
    const play = block.querySelector(".expr-param-play");
    const mf = row.querySelector("math-field");
    if (slider instanceof HTMLInputElement) {
      slider.min = String(p.min);
      slider.max = String(p.max);
      slider.step = String(p.step);
      slider.disabled = !!p.driven;
      slider.title = p.driven ? `value ${fmtNum(p.value)} (driven)` : `${name} = ${fmtNum(p.value)}`;
      if (document.activeElement !== slider) slider.value = String(p.value);
      const span = p.max - p.min;
      const zeroPct = span > 1e-12 ? ((0 - p.min) / span) * 100 : 50;
      const mark = block.querySelector(".expr-param-zero");
      if (mark instanceof HTMLElement) {
        const show = p.min < 0 && p.max > 0;
        mark.hidden = !show;
        if (show) mark.style.left = `${Math.min(100, Math.max(0, zeroPct))}%`;
      }
    }
    if (minEl instanceof HTMLInputElement && document.activeElement !== minEl) {
      minEl.value = fmtNum(p.min);
    }
    if (maxEl instanceof HTMLInputElement && document.activeElement !== maxEl) {
      maxEl.value = fmtNum(p.max);
    }
    if (play instanceof HTMLButtonElement) {
      play.disabled = !!p.driven;
      play.classList.toggle("on", p.animating);
      play.textContent = p.animating ? "⏸" : "▶";
      play.title = p.driven
        ? "Driven by equation (use t for time)"
        : p.animating
          ? "Pause animation"
          : "Animate between min and max";
    }
    if (mf && document.activeElement !== mf && !p.driven) {
      const cur = readFieldLatex(mf);
      if (cur !== p.latex) {
        if (typeof mf.setValue === "function") mf.setValue(p.latex, { silenceNotifications: true });
        else mf.value = p.latex;
      }
    }
    row.classList.toggle("has-error", !!p.error);
    mf?.classList.toggle("invalid", !!p.error);
  }

  function syncAllParamSliders() {
    for (const name of listParamNames()) {
      const p = getParam(name);
      if (!p?.exprId) continue;
      const row = root.querySelector(`.expr-row[data-id="${CSS.escape(p.exprId)}"]`);
      if (row) syncParamSlider(row, name);
    }
  }

  /**
   * Compact slider: min — track — max ▶
   * @param {HTMLElement} mid
   * @param {HTMLElement} row
   * @param {any} item
   * @param {string} paramName
   * @param {HTMLElement | null} mf
   */
  function appendParamControls(mid, row, item, paramName, mf) {
    const p = getParam(paramName);
    const min = p?.min ?? item.sliderMin;
    const max = p?.max ?? item.sliderMax;

    const block = document.createElement("div");
    block.className = "expr-param-block";
    block.dataset.paramBlock = paramName;

    const rail = document.createElement("div");
    rail.className = "expr-param-rail";

    const minEl = document.createElement("input");
    minEl.type = "text";
    minEl.inputMode = "decimal";
    minEl.className = "expr-param-min";
    minEl.value = fmtNum(min);
    minEl.title = "Minimum";
    minEl.addEventListener("change", () => {
      const n = Number(minEl.value);
      if (!Number.isFinite(n)) {
        syncParamSlider(row, paramName);
        return;
      }
      updateParam(paramName, { min: n });
      updateExprSilent(item.id, { sliderMin: n });
      syncParamSlider(row, paramName);
      paramNotify();
    });

    const trackWrap = document.createElement("div");
    trackWrap.className = "expr-param-track";

    const zero = document.createElement("span");
    zero.className = "expr-param-zero";
    zero.setAttribute("aria-hidden", "true");

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "expr-param-slider";
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(p?.step ?? 0.01);
    slider.value = String(p?.value ?? min);
    slider.disabled = !!p?.driven;
    slider.addEventListener("input", () => {
      const next = setParamValue(paramName, Number(slider.value), {
        stopAnim: true,
        rewriteLatex: true,
      });
      if (next) {
        if (mf) {
          updateExprSilent(item.id, { latex: next.latex, sliderAnimating: false });
          if (document.activeElement !== mf) {
            if (typeof mf.setValue === "function") mf.setValue(next.latex, { silenceNotifications: true });
            else mf.value = next.latex;
          }
        } else {
          updateExprSilent(item.id, { sliderAnimating: false });
        }
        syncParamSlider(row, paramName);
        paramNotify();
      }
    });

    trackWrap.append(zero, slider);

    const maxEl = document.createElement("input");
    maxEl.type = "text";
    maxEl.inputMode = "decimal";
    maxEl.className = "expr-param-max";
    maxEl.value = fmtNum(max);
    maxEl.title = "Maximum";
    maxEl.addEventListener("change", () => {
      const n = Number(maxEl.value);
      if (!Number.isFinite(n)) {
        syncParamSlider(row, paramName);
        return;
      }
      updateParam(paramName, { max: n });
      updateExprSilent(item.id, { sliderMax: n });
      syncParamSlider(row, paramName);
      paramNotify();
    });

    const play = document.createElement("button");
    play.type = "button";
    play.className = "expr-param-play";
    play.textContent = "▶";
    play.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const next = toggleParamAnimate(paramName);
      if (next) {
        updateExprSilent(item.id, { sliderAnimating: next.animating, sliderPhase: next.phase });
        syncParamSlider(row, paramName);
        paramNotify();
      }
    });

    rail.append(minEl, trackWrap, maxEl, play);
    block.appendChild(rail);
    mid.appendChild(block);
    queueMicrotask(() => syncParamSlider(row, paramName));
  }

  function captureFocus() {
    const active = document.activeElement;
    if (!active) return null;

    for (const row of root.querySelectorAll(".expr-row")) {
      if (!(row instanceof HTMLElement) || !row.dataset.id) continue;
      const mf = row.querySelector("math-field");
      if (!mf) continue;
      let hit = active === mf || (typeof mf.contains === "function" && mf.contains(active));
      if (!hit) {
        try {
          hit = !!(mf.shadowRoot && mf.shadowRoot.contains(active));
        } catch (_) {
          hit = false;
        }
      }
      if (!hit) {
        // Composed path (focus deep in MathLive internals).
        try {
          const path = typeof active.composedPath === "function" ? active.composedPath() : [];
          hit = path.includes(mf);
        } catch (_) {
          /* ignore */
        }
      }
      if (hit) return { id: row.dataset.id, pos: getCaretPos(mf) };
    }
    return null;
  }

  function restoreFocus(snap) {
    if (!snap?.id) return;
    const apply = () => focusFieldAt(root, snap.id, snap.pos);
    queueMicrotask(apply);
    requestAnimationFrame(() => requestAnimationFrame(apply));
  }

  /** Params needed under a row: owning `a=…` declaration only. */
  function neededParamsForItem(item) {
    /** @type {{ name: string }[]} */
    const out = [];
    const classified = String(item.latex || "").trim() ? classifyKind(item.latex) : null;
    if (classified?.kind === "parameter" && classified.paramName) {
      const p = getParam(classified.paramName);
      if (p && p.exprId === item.id) {
        out.push({ name: classified.paramName });
      }
    }
    return out;
  }

  /**
   * Update slider chrome in place without recreating math-fields (keeps caret).
   * Returns false when a full render is required.
   */
  function syncParamChrome() {
    const items = listExpressions();
    const rows = [...root.querySelectorAll(".expr-row")];
    if (rows.length !== items.length) return false;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const row = rows[i];
      if (!(row instanceof HTMLElement) || row.dataset.id !== item.id) return false;
      // Unified DOM: always color | mid | vis | del
      if (row.children.length !== 4) return false;
      const mid = row.querySelector(".expr-mid");
      const mf = row.querySelector("math-field");
      if (!(mid instanceof HTMLElement)) return false;

      const needed = neededParamsForItem(item);
      const neededSet = new Set(needed.map((n) => n.name));
      mid.querySelectorAll("[data-param-block]").forEach((block) => {
        const name = block.getAttribute("data-param-block");
        if (name && !neededSet.has(name)) block.remove();
      });
      for (const { name } of needed) {
        if (!mid.querySelector(`[data-param-block="${CSS.escape(name)}"]`)) {
          appendParamControls(mid, row, item, name, mf);
        }
      }

      const isParamDef = needed.length > 0;
      const warn = getExprWarning(item.id);
      row.classList.toggle("is-param", isParamDef || !!warn);
      row.classList.toggle("is-param-def", isParamDef || !!warn);
      row.classList.toggle("is-hidden", !item.enabled);
      row.classList.toggle("selected", item.id === getSelectedId());
      row.classList.toggle("has-error", !!warn);
      if (isParamDef) row.dataset.param = needed[0].name;
      else delete row.dataset.param;

      const swatch = row.querySelector(".expr-color");
      if (swatch instanceof HTMLInputElement) {
        swatch.disabled = isParamDef || !!warn;
        swatch.title = isParamDef || warn ? "Parameters are not drawn" : "Color";
      }
      if (mf instanceof HTMLElement) {
        mf.classList.toggle("invalid", !!warn);
        if (warn) mf.title = warn;
        else mf.removeAttribute("title");
      }

      for (const { name } of needed) syncParamSlider(row, name);
    }
    return true;
  }

  function render() {
    beginSuppressAutoCommit();
    try {
      const focusSnap = captureFocus();
      const items = listExpressions();
      const selected = getSelectedId();
      root.replaceChildren();

      for (const item of items) {
      const classified = String(item.latex || "").trim() ? classifyKind(item.latex) : null;
      const warn = getExprWarning(item.id);
      const isParamKind = classified?.kind === "parameter";
      const paramName = isParamKind ? classified.paramName : null;
      const ownsParam =
        !!paramName &&
        (() => {
          const p = getParam(paramName);
          return !!(p && p.exprId === item.id);
        })();

      const row = document.createElement("div");
      row.className =
        "expr-row" +
        (item.id === selected ? " selected" : "") +
        (item.enabled ? "" : " is-hidden") +
        (ownsParam || warn ? " is-param" : "") +
        (ownsParam || warn ? " is-param-def" : "") +
        (warn ? " has-error" : "");
      row.dataset.id = item.id;
      if (ownsParam && paramName) row.dataset.param = paramName;

      const swatch = document.createElement("input");
      swatch.type = "color";
      swatch.className = "expr-color";
      swatch.value = item.color.startsWith("#") ? item.color : "#2d70b3";
      swatch.title = ownsParam || warn ? "Parameters are not drawn" : "Color";
      swatch.disabled = !!(ownsParam || warn);
      swatch.addEventListener("input", () => {
        updateExpr(item.id, { color: swatch.value });
        if (onColorChange) onColorChange();
        else onExprChange();
      });

      const mid = document.createElement("div");
      mid.className = "expr-mid";

      const fieldRow = document.createElement("div");
      fieldRow.className = "expr-field-row";

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
        const classified = classifyKind(readFieldLatex(mf));
        if (classified?.kind === "parameter" && classified.paramName) {
          // Clicking the variable line counts as leaving the field expression.
          if (!suppressAutoCommit) commitAutoParams();
          const next = stopParamAnimation(classified.paramName);
          if (next) {
            updateExprSilent(item.id, { sliderAnimating: false });
            syncParamSlider(row, classified.paramName);
          }
        }
      });
      mf.addEventListener("blur", () => {
        scheduleCommitIfLeftExpr(item.id);
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
              updateExpr(item.id, { latex: readFieldLatex(mf) });
              focusFieldAt(root, target.id, getCaretPos(mf));
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

      fieldRow.appendChild(mf);
      mid.appendChild(fieldRow);
      if (warn) {
        mf.classList.add("invalid");
        mf.title = warn;
      }

      if (ownsParam && paramName) {
        appendParamControls(mid, row, item, paramName, mf);
      }

      const vis = document.createElement("button");
      vis.type = "button";
      vis.className = "expr-vis secondary" + (item.enabled ? "" : " is-off");
      vis.title = item.enabled ? "Hide" : "Show";
      vis.setAttribute("aria-label", item.enabled ? "Hide expression" : "Show expression");
      vis.setAttribute("aria-pressed", item.enabled ? "true" : "false");
      vis.innerHTML = item.enabled ? ICON_EYE : ICON_EYE_OFF;
      vis.addEventListener("click", (ev) => {
        ev.stopPropagation();
        updateExpr(item.id, { enabled: !item.enabled });
        onStructuralChange();
        render();
      });

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
        if (ev.target === del || ev.target === swatch || ev.target === vis) return;
        if (vis.contains(ev.target)) return;
        if (ev.target === mf || mf.contains?.(ev.target)) return;
        if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLButtonElement) return;
        selectExpr(item.id);
        root.querySelectorAll(".expr-row").forEach((r) => {
          r.classList.toggle("selected", r.dataset.id === item.id);
        });
        mf.focus?.();
      });

      // Always the same 4 children so CSS grid never crushes mid during kind flips.
      row.append(swatch, mid, vis, del);
      root.appendChild(row);
      }

      restoreFocus(focusSnap);
    } finally {
      endSuppressAutoCommit();
    }
  }

  return { render, syncAllParamSliders, syncParamChrome };
}
