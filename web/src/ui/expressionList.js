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
  moveExpr,
  DEFAULT_EXPR_COLOR,
  resolveExprGradient,
  cssGradientFromColors,
  normalizeGradColors,
  MAX_GRAD_STOPS,
  MIN_GRAD_STOPS,
} from "../model/expressions.js";
import { classifyExpr } from "../math/fit.js";
import { mountLiquidThumb, syncLiquidThumb } from "./liquidSlider.js";
import {
  getParam,
  listParamNames,
  setParamValue,
  updateParam,
  toggleParamAnimate,
  stopParamAnimation,
  phaseForValue,
  normalizeAnimMode,
} from "../model/params.js";

const ANIM_OPTS_ICON = `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 3.2a1.1 1.1 0 1 0 0 2.2 1.1 1.1 0 0 0 0-2.2Zm0 3.7a1.1 1.1 0 1 0 0 2.2 1.1 1.1 0 0 0 0-2.2Zm0 3.7a1.1 1.1 0 1 0 0 2.2 1.1 1.1 0 0 0 0-2.2Z"/></svg>`;

const ANIM_SPEED_MIN = 0.05;
const ANIM_SPEED_MAX = 2;
const ANIM_SPEED_STEP = 0.05;
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
  const mf = root.querySelector(`.expr-row[data-id="${CSS.escape(id)}"] math-field`);
  if (!mf) return false;
  selectExpr(id);
  root.querySelectorAll(".expr-row").forEach((r) => {
    r.classList.toggle("selected", r.dataset.id === id);
  });
  try {
    mf.focus?.({ preventScroll: true });
  } catch (_) {
    mf.focus?.();
  }
  const end = getLastOffset(mf);
  const pos = Math.max(0, Math.min(offset | 0, end));
  if (typeof mf.position === "number") {
    mf.position = pos;
    return true;
  }
  if (mf.selection) {
    try {
      mf.selection = { ranges: [[pos, pos]] };
    } catch (_) {
      /* ignore */
    }
  }
  return true;
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
  // Keep MathLive's default cos→\cos etc. shortcuts; give them time to fire
  // even if the user types slowly (0 = library default 1s flush).
  try {
    mf.inlineShortcutTimeout = 2000;
  } catch (_) {
    /* ignore */
  }
}

/** True when MathLive's latex suggestion UI is showing (owns ↑/↓). */
function isSuggestionUiActive(mf) {
  const panel = document.getElementById("mathlive-suggestion-popover");
  if (panel?.classList.contains("is-visible")) return true;
  try {
    if (mf?.mode === "latex") return true;
  } catch (_) {
    /* ignore */
  }
  return false;
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
   * After leave, recompile so deferred auto-param rows can materialize.
   */
  function scheduleCommitIfLeftExpr(fromExprId) {
    queueMicrotask(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (suppressAutoCommit) return;
          const focusedId = focusedExprIdInList();
          if (focusedId === fromExprId) return;
          commitAutoParams();
          onExprChange();
        });
      });
    });
  }

  function applyGradientChrome(row, item) {
    const grad = resolveExprGradient(item);
    const css = cssGradientFromColors(grad.colors);
    row.style.setProperty("--expr-grad", css);
    row.style.setProperty("--expr-c0", grad.color);
    row.style.setProperty("--expr-c1", grad.color2);
    const swatch = row.querySelector(".expr-color");
    if (swatch instanceof HTMLElement) {
      swatch.style.setProperty("--expr-grad", css);
      swatch.style.setProperty("--expr-c0", grad.color);
      swatch.style.setProperty("--expr-c1", grad.color2);
      if (!swatch.disabled) {
        swatch.title = `Edit gradient (${grad.colors.length} colors)`;
      }
    }
  }

  /** @type {HTMLElement | null} */
  let openGradPopover = null;
  /** @type {HTMLElement | null} */
  let openAnimPopover = null;

  function closeGradPopover() {
    if (openGradPopover) {
      openGradPopover.remove();
      openGradPopover = null;
    }
    document.removeEventListener("pointerdown", onGradOutside, true);
  }

  function closeAnimPopover() {
    if (openAnimPopover) {
      openAnimPopover.remove();
      openAnimPopover = null;
    }
    document.removeEventListener("pointerdown", onAnimOutside, true);
  }

  function closeAllPopovers() {
    closeGradPopover();
    closeAnimPopover();
  }

  function onGradOutside(ev) {
    if (!(ev.target instanceof Node)) return;
    if (openGradPopover?.contains(ev.target)) return;
    if (ev.target instanceof Element && ev.target.closest(".expr-color")) return;
    closeGradPopover();
  }

  function onAnimOutside(ev) {
    if (!(ev.target instanceof Node)) return;
    if (openAnimPopover?.contains(ev.target)) return;
    if (ev.target instanceof Element && ev.target.closest(".expr-param-anim-opts")) return;
    closeAnimPopover();
  }

  /**
   * @param {HTMLElement} anchor
   * @param {any} item
   * @param {HTMLElement} row
   */
  function openGradientEditor(anchor, item, row) {
    closeAllPopovers();
    const grad = resolveExprGradient(item);
    let draft = grad.colors.slice();

    const pop = document.createElement("div");
    pop.className = "grad-popover";
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-label", "Edit gradient");

    const head = document.createElement("div");
    head.className = "grad-popover-head";
    head.textContent = "Gradient colors";

    const preview = document.createElement("div");
    preview.className = "grad-popover-preview";

    const list = document.createElement("div");
    list.className = "grad-popover-stops";

    const actions = document.createElement("div");
    actions.className = "grad-popover-actions";

    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.className = "secondary";
    addBtn.textContent = "Add color";

    function commit(next) {
      draft = normalizeGradColors(next);
      updateExpr(item.id, { colors: draft });
      applyGradientChrome(row, { ...item, colors: draft });
      if (onColorChange) onColorChange();
      else onExprChange();
      renderStops();
    }

    function renderStops() {
      preview.style.background = cssGradientFromColors(draft);
      list.replaceChildren();
      draft.forEach((hex, i) => {
        const stop = document.createElement("div");
        stop.className = "grad-stop";

        const pick = document.createElement("input");
        pick.type = "color";
        pick.className = "grad-stop-color";
        pick.value = hex.startsWith("#") ? hex : DEFAULT_EXPR_COLOR;
        pick.title = `Stop ${i + 1}`;
        pick.addEventListener("input", () => {
          const next = draft.slice();
          next[i] = pick.value;
          commit(next);
        });
        pick.addEventListener("click", (ev) => ev.stopPropagation());

        const label = document.createElement("span");
        label.className = "grad-stop-label";
        label.textContent = i === 0 ? "Start" : i === draft.length - 1 ? "End" : `Stop ${i + 1}`;

        const rm = document.createElement("button");
        rm.type = "button";
        rm.className = "grad-stop-remove secondary";
        rm.textContent = "×";
        rm.title = "Remove stop";
        rm.disabled = draft.length <= MIN_GRAD_STOPS;
        rm.addEventListener("click", (ev) => {
          ev.stopPropagation();
          if (draft.length <= MIN_GRAD_STOPS) return;
          const next = draft.slice();
          next.splice(i, 1);
          commit(next);
        });

        stop.append(pick, label, rm);
        list.append(stop);
      });
      addBtn.disabled = draft.length >= MAX_GRAD_STOPS;
      addBtn.title =
        draft.length >= MAX_GRAD_STOPS
          ? `Max ${MAX_GRAD_STOPS} colors`
          : "Add a gradient stop";
    }

    addBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (draft.length >= MAX_GRAD_STOPS) return;
      const last = draft[draft.length - 1] || DEFAULT_EXPR_COLOR;
      commit([...draft, last]);
    });

    actions.append(addBtn);
    pop.append(head, preview, list, actions);

    const rect = anchor.getBoundingClientRect();
    pop.style.left = `${Math.min(window.innerWidth - 220, Math.max(8, rect.left))}px`;
    pop.style.top = `${Math.min(window.innerHeight - 280, rect.bottom + 6)}px`;
    document.body.append(pop);
    openGradPopover = pop;
    renderStops();
    document.addEventListener("pointerdown", onGradOutside, true);
  }

  /**
   * @param {number} speed
   */
  function fmtAnimSpeed(speed) {
    const s = Math.round(speed * 100) / 100;
    return Number.isInteger(s) ? String(s) : s.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  /**
   * @param {HTMLElement} anchor
   * @param {any} item
   * @param {HTMLElement} row
   * @param {string} paramName
   */
  function openAnimOptions(anchor, item, row, paramName) {
    closeAllPopovers();
    const p = getParam(paramName);
    if (!p || p.driven) return;

    const pop = document.createElement("div");
    pop.className = "anim-popover";
    pop.dataset.param = paramName;
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-label", "Animation options");

    const head = document.createElement("div");
    head.className = "anim-popover-head";
    head.textContent = "Animation";

    const modes = document.createElement("div");
    modes.className = "anim-popover-modes";
    modes.setAttribute("role", "group");
    modes.setAttribute("aria-label", "Curve");

    /** @type {{ mode: "pingpong" | "loop", label: string }[]} */
    const modeDefs = [
      { mode: "pingpong", label: "Back & forth" },
      { mode: "loop", label: "Loop" },
    ];
    /** @type {HTMLButtonElement[]} */
    const modeBtns = [];
    for (const def of modeDefs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = def.label;
      btn.dataset.mode = def.mode;
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        applyAnimPatch({ animMode: def.mode });
      });
      modeBtns.push(btn);
      modes.append(btn);
    }

    const speedBlock = document.createElement("div");
    speedBlock.className = "anim-popover-speed";

    const speedRow = document.createElement("div");
    speedRow.className = "anim-popover-speed-row";
    const speedLabel = document.createElement("span");
    speedLabel.textContent = "Speed";
    const speedVal = document.createElement("span");
    speedVal.className = "anim-popover-speed-val";

    const speedTrack = document.createElement("div");
    speedTrack.className = "anim-popover-speed-track expr-param-track";

    const speed = document.createElement("input");
    speed.type = "range";
    speed.className = "expr-param-slider";
    speed.min = String(ANIM_SPEED_MIN);
    speed.max = String(ANIM_SPEED_MAX);
    speed.step = String(ANIM_SPEED_STEP);
    speed.title = "Cycles per second";
    speed.addEventListener("input", () => {
      const n = Number(speed.value);
      if (!Number.isFinite(n) || n <= 0) return;
      applyAnimPatch({ speed: n });
    });
    speed.addEventListener("click", (ev) => ev.stopPropagation());

    speedTrack.append(speed);
    mountLiquidThumb(speedTrack, speed);

    speedRow.append(speedLabel, speedVal);
    speedBlock.append(speedRow, speedTrack);
    pop.append(head, modes, speedBlock);

    /**
     * @param {{ animMode?: "pingpong" | "loop", speed?: number }} patch
     */
    function applyAnimPatch(patch) {
      const cur = getParam(paramName);
      if (!cur) return;
      const timeSec = performance.now() / 1000;
      const nextMode = patch.animMode != null ? normalizeAnimMode(patch.animMode) : cur.animMode;
      const nextSpeed =
        Number.isFinite(patch.speed) && patch.speed > 0 ? patch.speed : cur.speed;
      const next = updateParam(paramName, {
        animMode: nextMode,
        speed: nextSpeed,
        phase: phaseForValue(
          { ...cur, animMode: nextMode, speed: nextSpeed },
          timeSec,
        ),
      });
      if (!next) return;
      updateExprSilent(item.id, {
        sliderAnimMode: next.animMode,
        sliderSpeed: next.speed,
        sliderPhase: next.phase,
      });
      syncParamSlider(row, paramName);
      syncPopoverFromParam(next);
      paramNotify();
    }

    /**
     * @param {{ value: number, min: number, max: number, speed: number, animMode: string }} st
     */
    function syncPopoverFromParam(st) {
      const mode = normalizeAnimMode(st.animMode);
      for (const btn of modeBtns) {
        btn.classList.toggle("on", btn.dataset.mode === mode);
      }
      if (document.activeElement !== speed) speed.value = String(st.speed);
      speedVal.textContent = `${fmtAnimSpeed(st.speed)}×`;
      syncLiquidThumb(speed);
    }

    syncPopoverFromParam(p);

    const rect = anchor.getBoundingClientRect();
    pop.style.left = `${Math.min(window.innerWidth - 240, Math.max(8, rect.right - 220))}px`;
    pop.style.top = `${Math.min(window.innerHeight - 200, rect.bottom + 6)}px`;
    document.body.append(pop);
    openAnimPopover = pop;
    document.addEventListener("pointerdown", onAnimOutside, true);
  }

  function syncParamSlider(row, name) {
    const p = getParam(name);
    if (!p || !row) return;
    const block = row.querySelector(`[data-param-block="${CSS.escape(name)}"]`) || row;
    const slider = block.querySelector(".expr-param-slider");
    const minEl = block.querySelector(".expr-param-min");
    const maxEl = block.querySelector(".expr-param-max");
    const play = block.querySelector(".expr-param-play");
    const opts = block.querySelector(".expr-param-anim-opts");
    const mf = row.querySelector("math-field");
    if (slider instanceof HTMLInputElement) {
      slider.min = String(p.min);
      slider.max = String(p.max);
      // Continuous thumb while playing — discrete step only for manual scrubbing.
      slider.step = p.animating && !p.driven ? "any" : String(p.step);
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
      syncLiquidThumb(slider);
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
        ? "Driven by equation"
        : p.animating
          ? "Pause animation"
          : "Animate between min and max";
    }
    if (opts instanceof HTMLButtonElement) {
      opts.disabled = !!p.driven;
      const mode = normalizeAnimMode(p.animMode);
      opts.title = p.driven
        ? "Driven by equation"
        : `Animation options (${mode === "loop" ? "loop" : "back & forth"}, ${fmtAnimSpeed(p.speed)}×)`;
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
    mountLiquidThumb(trackWrap, slider);

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

    const opts = document.createElement("button");
    opts.type = "button";
    opts.className = "expr-param-anim-opts";
    opts.innerHTML = ANIM_OPTS_ICON;
    opts.setAttribute("aria-label", "Animation options");
    opts.title = "Animation options";
    opts.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (openAnimPopover) {
        const same = openAnimPopover.dataset.param === paramName;
        closeAnimPopover();
        if (same) return;
      }
      openAnimOptions(opts, item, row, paramName);
    });

    rail.append(minEl, trackWrap, maxEl, play, opts);
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

  /** Invalidates in-flight restoreFocus from an older render. */
  let focusEpoch = 0;
  /**
   * Survives compile/auto-param rebuilds that bump focusEpoch after Enter/merge.
   * Cleared once a restore for this target successfully runs.
   * @type {{ id: string, pos: number } | null}
   */
  let pendingFocus = null;

  /**
   * @param {{ id: string, pos?: number } | null | undefined} snap
   * @param {number} [epoch]
   */
  function restoreFocus(snap, epoch = focusEpoch) {
    if (!snap?.id) return;
    const id = snap.id;
    const pos = snap.pos | 0;
    const apply = (clearPending) => {
      if (epoch !== focusEpoch) return;
      focusFieldAt(root, id, pos);
      if (clearPending && pendingFocus?.id === id) pendingFocus = null;
    };
    // Microtask: ASAP after DOM swap. rAF: MathLive custom elements often need a frame.
    queueMicrotask(() => apply(false));
    requestAnimationFrame(() => requestAnimationFrame(() => apply(true)));
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
      // Unified DOM: always drag | color | mid | vis | del
      if (row.children.length !== 5) return false;
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
      row.classList.toggle("is-param-def", isParamDef || !!warn);
      row.classList.toggle("is-hidden", !item.enabled);
      row.classList.toggle("selected", item.id === getSelectedId());
      row.classList.toggle("has-error", !!warn);
      if (isParamDef) row.dataset.param = needed[0].name;
      else delete row.dataset.param;

      const swatch = row.querySelector(".expr-color");
      if (swatch instanceof HTMLButtonElement) {
        const disabled = isParamDef || !!warn;
        swatch.disabled = disabled;
        swatch.title = disabled ? "Parameters are not drawn" : "Edit gradient colors";
        applyGradientChrome(row, item);
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

  /** @type {{ id: string, pointerId: number, startY: number, moved: boolean, offsetX: number, offsetY: number, width: number } | null} */
  let dragState = null;
  /** @type {string | null | undefined} */
  let liveBeforeId = undefined;
  /** @type {HTMLElement | null} */
  let dragPlaceholder = null;
  /** @type {HTMLElement | null} floating row (the real node) */
  let dragFloat = null;

  function rowElements() {
    return [...root.querySelectorAll(".expr-row")].filter(
      (r) => r instanceof HTMLElement && r.dataset.id && !r.classList.contains("is-drag-floating"),
    );
  }

  function clearDragVisuals() {
    root.classList.remove("is-reordering");
    if (dragFloat instanceof HTMLElement) {
      dragFloat.classList.remove("is-drag-floating");
      dragFloat.style.position = "";
      dragFloat.style.left = "";
      dragFloat.style.top = "";
      dragFloat.style.width = "";
      dragFloat.style.height = "";
      dragFloat.style.zIndex = "";
      dragFloat.style.margin = "";
      dragFloat.style.transform = "";
      dragFloat.style.pointerEvents = "";
      dragFloat.style.boxShadow = "";
      dragFloat = null;
    }
    if (dragPlaceholder) {
      dragPlaceholder.remove();
      dragPlaceholder = null;
    }
    root.querySelectorAll(".expr-row").forEach((r) => {
      if (r instanceof HTMLElement) {
        r.style.transition = "";
        r.style.transform = "";
      }
    });
  }

  function moveDragFloat(clientX, clientY) {
    if (!dragFloat || !dragState) return;
    dragFloat.style.transform = `translate(${clientX - dragState.offsetX}px, ${clientY - dragState.offsetY}px) scale(1.02)`;
  }

  /**
   * Lift the real row into a fixed float; leave a placeholder hole in the list.
   * @param {HTMLElement} row
   * @param {number} clientX
   * @param {number} clientY
   */
  function beginFloatDrag(row, clientX, clientY) {
    const rect = row.getBoundingClientRect();
    const ph = document.createElement("div");
    ph.className = "expr-row-placeholder";
    ph.style.height = `${rect.height}px`;
    ph.setAttribute("aria-hidden", "true");
    row.parentNode?.insertBefore(ph, row);
    dragPlaceholder = ph;

    // Keep capture target alive: float the real row (not a clone).
    dragFloat = row;
    row.classList.add("is-drag-floating");
    row.style.position = "fixed";
    row.style.left = "0";
    row.style.top = "0";
    row.style.width = `${rect.width}px`;
    row.style.height = `${rect.height}px`;
    row.style.zIndex = "10000";
    row.style.margin = "0";
    row.style.pointerEvents = "none";
    row.style.boxShadow = "var(--glass-shadow), var(--glass-fresnel)";
    document.body.appendChild(row);
    moveDragFloat(clientX, clientY);

    const next = ph.nextElementSibling;
    liveBeforeId =
      next instanceof HTMLElement && next.classList.contains("expr-row")
        ? next.dataset.id ?? null
        : null;
  }

  /**
   * @param {number} clientY
   * @param {string | null} excludeId
   * @returns {string | null}
   */
  function resolveBeforeId(clientY, excludeId) {
    const others = rowElements().filter((r) => r.dataset.id !== excludeId);
    if (!others.length) return null;

    const first = others[0];
    const last = others[others.length - 1];
    const firstRect = first.getBoundingClientRect();
    const lastRect = last.getBoundingClientRect();

    if (clientY < firstRect.top + firstRect.height * 0.5) {
      return first.dataset.id ?? null;
    }
    if (clientY >= lastRect.top + lastRect.height * 0.5) {
      return null;
    }

    for (const row of others) {
      const rect = row.getBoundingClientRect();
      if (clientY < rect.top + rect.height * 0.5) {
        return row.dataset.id ?? null;
      }
    }
    return null;
  }

  /**
   * @param {Map<string, DOMRect>} prevRects
   */
  function flipRows(prevRects) {
    for (const row of rowElements()) {
      const id = row.dataset.id;
      if (!id) continue;
      const prev = prevRects.get(id);
      if (!prev) continue;
      const next = row.getBoundingClientRect();
      const dy = prev.top - next.top;
      if (Math.abs(dy) < 0.5) continue;
      row.style.transition = "none";
      row.style.transform = `translateY(${dy}px)`;
      void row.offsetHeight;
      row.style.transition = "transform 200ms cubic-bezier(0.2, 0.8, 0.2, 1)";
      row.style.transform = "";
      const onEnd = (ev) => {
        if (ev.target !== row || ev.propertyName !== "transform") return;
        row.style.transition = "";
        row.removeEventListener("transitionend", onEnd);
      };
      row.addEventListener("transitionend", onEnd);
    }
  }

  /**
   * Move the placeholder hole (not the captured row) to match the pointer.
   * @param {number} clientY
   */
  function liveReorderToPointer(clientY) {
    if (!dragState || !dragPlaceholder) return;
    const beforeId = resolveBeforeId(clientY, dragState.id);
    if (beforeId === liveBeforeId) return;

    const ph = dragPlaceholder;
    const currentNext = ph.nextElementSibling;
    const currentBeforeId =
      currentNext instanceof HTMLElement && currentNext.classList.contains("expr-row")
        ? currentNext.dataset.id ?? null
        : null;
    if (currentBeforeId === beforeId) {
      liveBeforeId = beforeId;
      return;
    }

    const prevRects = new Map();
    for (const r of rowElements()) {
      if (r.dataset.id) prevRects.set(r.dataset.id, r.getBoundingClientRect());
    }

    if (beforeId) {
      const target = root.querySelector(`.expr-row[data-id="${CSS.escape(beforeId)}"]`);
      if (!(target instanceof HTMLElement)) return;
      root.insertBefore(ph, target);
    } else {
      const actions = root.querySelector(".expr-list-actions");
      if (actions) root.insertBefore(ph, actions);
      else root.appendChild(ph);
    }

    liveBeforeId = beforeId;
    flipRows(prevRects);
  }

  function unbindWindowDrag() {
    window.removeEventListener("pointermove", onWindowPointerMove);
    window.removeEventListener("pointerup", onWindowPointerUp);
    window.removeEventListener("pointercancel", onWindowPointerUp);
  }

  function bindWindowDrag() {
    window.addEventListener("pointermove", onWindowPointerMove);
    window.addEventListener("pointerup", onWindowPointerUp);
    window.addEventListener("pointercancel", onWindowPointerUp);
  }

  function onWindowPointerMove(ev) {
    if (!dragState || ev.pointerId !== dragState.pointerId) return;
    onPointerMove(ev);
  }

  function onWindowPointerUp(ev) {
    if (!dragState || ev.pointerId !== dragState.pointerId) return;
    onPointerUp(ev);
  }

  function finishPointerDrag() {
    const state = dragState;
    if (!state) return;
    dragState = null;
    unbindWindowDrag();
    if (!state.moved) {
      clearDragVisuals();
      return;
    }

    // Commit slot from the placeholder's live position.
    const ph = dragPlaceholder;
    let beforeId = null;
    if (ph) {
      const next = ph.nextElementSibling;
      beforeId =
        next instanceof HTMLElement && next.classList.contains("expr-row")
          ? next.dataset.id ?? null
          : null;
    } else if (typeof liveBeforeId !== "undefined") {
      beforeId = liveBeforeId;
    }
    liveBeforeId = undefined;

    const floatRow = dragFloat;
    if (floatRow && ph?.parentNode) {
      ph.parentNode.insertBefore(floatRow, ph);
    } else if (floatRow?.parentNode === document.body) {
      root.appendChild(floatRow);
    }
    clearDragVisuals();

    moveExpr(state.id, beforeId);
    onStructuralChange();
    render();
  }

  function onPointerMove(ev) {
    if (!dragState || ev.pointerId !== dragState.pointerId) return;
    if (!dragState.moved) {
      if (Math.abs(ev.clientY - dragState.startY) < 3) return;
      dragState.moved = true;
      commitAutoParams();
      root.classList.add("is-reordering");
      const row = root.querySelector(`.expr-row[data-id="${CSS.escape(dragState.id)}"]`);
      if (row instanceof HTMLElement) {
        beginFloatDrag(row, ev.clientX, ev.clientY);
      }
      bindWindowDrag();
    } else {
      moveDragFloat(ev.clientX, ev.clientY);
    }
    ev.preventDefault();
    liveReorderToPointer(ev.clientY);
  }

  function onPointerUp(ev) {
    if (!dragState || ev.pointerId !== dragState.pointerId) return;
    if (dragState.moved) liveReorderToPointer(ev.clientY);
    try {
      ev.currentTarget?.releasePointerCapture?.(ev.pointerId);
    } catch (_) {
      /* ignore */
    }
    finishPointerDrag();
  }

  /**
   * @param {HTMLElement} row
   * @param {any} item
   */
  function createDragHandle(row, item) {
    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "expr-drag";
    handle.title = "Drag to reorder";
    handle.setAttribute("aria-label", "Drag to reorder");
    handle.tabIndex = -1;
    handle.textContent = "⠿";

    handle.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0 || dragState) return;
      ev.preventDefault();
      ev.stopPropagation();
      const rect = row.getBoundingClientRect();
      dragState = {
        id: item.id,
        pointerId: ev.pointerId,
        startY: ev.clientY,
        moved: false,
        offsetX: ev.clientX - rect.left,
        offsetY: ev.clientY - rect.top,
        width: rect.width,
      };
      liveBeforeId = undefined;
      bindWindowDrag();
      try {
        handle.setPointerCapture(ev.pointerId);
      } catch (_) {
        /* ignore */
      }
    });
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerUp);
    handle.addEventListener("pointercancel", onPointerUp);

    return handle;
  }

  /**
   * @param {{ id: string, pos?: number } | null} [focusOverride]
   *        When set (e.g. after Enter split), restore this instead of the
   *        pre-rebuild caret — which is still on the row being left.
   */
  function render(focusOverride = null) {
    closeAllPopovers();
    beginSuppressAutoCommit();
    if (focusOverride?.id != null) {
      pendingFocus = { id: String(focusOverride.id), pos: focusOverride.pos | 0 };
    }
    const epoch = ++focusEpoch;
    try {
      // Prefer sticky Enter/merge target over captureFocus — a follow-up compile
      // re-render often runs before the new math-field has taken focus.
      const focusSnap = pendingFocus ?? captureFocus();
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
        (ownsParam || warn ? " is-param-def" : "") +
        (warn ? " has-error" : "");
      row.dataset.id = item.id;
      if (ownsParam && paramName) row.dataset.param = paramName;

      const swatch = document.createElement("button");
      swatch.type = "button";
      swatch.className = "expr-color";
      swatch.title = ownsParam || warn ? "Parameters are not drawn" : "Edit gradient colors";
      swatch.disabled = !!(ownsParam || warn);
      applyGradientChrome(row, item);
      swatch.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (swatch.disabled) return;
        openGradientEditor(swatch, item, row);
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
          // ↑/↓: defer only while latex suggestions own the keys.
          if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
            if (isSuggestionUiActive(mf)) return;
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
                render({ id: merged.id, pos: merged.caretOffset });
              }
            }
          }
        },
        true,
      );
      // Enter in bubble so MathLive (capture on its sink) runs first.
      // Split only when MathLive did not consume the key (e.g. latex complete).
      mf.addEventListener("keydown", (ev) => {
        if (ev.key !== "Enter" || ev.shiftKey) return;
        if (ev.defaultPrevented) return;
        ev.preventDefault();
        ev.stopPropagation();
        const { left, right } = latexAroundCaret(mf);
        updateExpr(item.id, { latex: left });
        const split = splitExprAt(item.id, left, right);
        onStructuralChange();
        // Pass explicit focus — render's capture would still see this field, and its
        // delayed restoreFocus would yank the caret back after we move to the new row.
        render(split ? { id: split.id, pos: 0 } : null);
      });

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
        if (ev.target instanceof Element && ev.target.closest(".expr-drag")) return;
        if (vis.contains(ev.target)) return;
        if (ev.target === mf || mf.contains?.(ev.target)) return;
        if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLButtonElement) return;
        selectExpr(item.id);
        root.querySelectorAll(".expr-row").forEach((r) => {
          r.classList.toggle("selected", r.dataset.id === item.id);
        });
        mf.focus?.();
      });

      const drag = createDragHandle(row, item);
      // Always the same 5 children so CSS grid never crushes mid during kind flips.
      row.append(drag, swatch, mid, vis, del);
      root.appendChild(row);
      }

      restoreFocus(focusSnap, epoch);
    } finally {
      endSuppressAutoCommit();
    }
  }

  return { render, syncAllParamSliders, syncParamChrome };
}
