<script lang="ts">
  import { onMount } from "svelte";
  import type { ExprItem } from "../../types/models.js";
import {
  resolveExprGradient,
  cssGradientFromColors,
  getExprWarning,
  updateExpr,
  removeExpr,
  mergeExprIntoPrevious,
  splitExprAt,
  listExpressions,
  commitAutoParams,
} from "../../model/expressions.js";
  import { updateExprSilent } from "../../model/expressions.js";
  import { stopParamAnimation, getParam } from "../../model/params.js";
  import {
    configureMathField,
    readFieldLatex,
    setFieldLatex,
    getCaretPos,
    setCaretPos,
    getLastOffset,
    isCursorAtStart,
    latexAroundCaret,
    isSuggestionUiActive,
    classifyKind,
    neededParamForItem,
    isParameterRow,
    isMathFieldFocused,
    ICON_EYE,
    ICON_EYE_OFF,
    forceReflow,
  } from "./helpers.ts";
  import { openGradientEditor } from "./popovers.ts";
  import ParamRail from "./ParamRail.svelte";
  import ParamAnimControls from "./ParamAnimControls.svelte";
  import { convertLatexToMarkup } from "mathlive";
  import {
  collectPendingParamsForExpr,
  createParamRows,
  formatPendingParamLabelPlain,
  formatPendingParamNamesLatex,
  formatPendingParamOverflow,
  pendingParamErrorMessage,
} from "../../app/pendingParams.js";
  import { scheduleDoubleRaf } from "./autoCommit.ts";

  interface Props {
    item: ExprItem;
    selected: boolean;
    paramTick: number;
    suppressAutoCommit: () => boolean;
    onExprChange: () => void;
    onStructuralChange: () => void;
    onColorChange?: () => void;
    onVisibilityChange?: () => void;
    onParamChange: () => void;
    onSelect: (id: string) => void;
    onFocusNav: (targetId: string, caret: number) => void;
    onSplit: (focus: { id: string; pos: number } | null) => void;
    onMerge: (focus: { id: string; pos: number }) => void;
    onDragStart: (row: HTMLElement, ev: PointerEvent) => void;
    onScheduleCommit: (fromId: string) => void;
    /** Mobile carousel: no Enter-to-split or Backspace-to-merge. */
    disableSplitMerge?: boolean;
  }

  let {
    item,
    selected,
    paramTick,
    suppressAutoCommit,
    onExprChange,
    onStructuralChange,
    onColorChange,
    onVisibilityChange,
    onParamChange,
    onSelect,
    onFocusNav,
    onSplit,
    onMerge,
    onDragStart,
    onScheduleCommit,
    disableSplitMerge = false,
  }: Props = $props();

  let rowEl: HTMLDivElement | undefined = $state();
  let mfEl: MathfieldElement | undefined = $state();
  let paramRail: ParamRail | undefined = $state();

  const warn = $derived(getExprWarning(item.id));
  const paramName = $derived.by(() => {
    void paramTick;
    return neededParamForItem(item, paramTick);
  });
  const paramErr = $derived.by(() => {
    void paramTick;
    return paramName ? getParam(paramName)?.error ?? null : null;
  });
  const pendingParams = $derived.by(() => {
    void paramTick;
    return collectPendingParamsForExpr(item);
  });
  const pendingParamLabelPlain = $derived(formatPendingParamLabelPlain(pendingParams));
  const pendingParamNamesLatex = $derived(formatPendingParamNamesLatex(pendingParams));
  const pendingParamOverflow = $derived(formatPendingParamOverflow(pendingParams));
  const pendingParamNamesMarkup = $derived(
    pendingParamNamesLatex
      ? convertLatexToMarkup(pendingParamNamesLatex, { defaultMode: "textstyle" })
      : "",
  );
  const pendingErr = $derived(pendingParamErrorMessage(pendingParams));
  const rowError = $derived(warn ?? paramErr ?? pendingErr);

  // Confirmed live in Safari: this row is `display: grid; overflow:
  // hidden`, and WebKit's auto grid-track height keeps its cached (taller)
  // size after .expr-pending-params is removed — getComputedStyle still
  // reports the stale height, so this is a real relayout bug, not a
  // paint/compositing one (seen on both the desktop sidebar list and the
  // mobile carousel, so the row itself is the actual source).
  $effect(() => {
    if (pendingParams.length > 0 || !rowEl) return;
    forceReflow(rowEl);
  });
  const isParamDef = $derived(isParameterRow(item.latex || ""));
  /** Bumped when the gradient editor commits; model mutates in place so `item` identity is stale. */
  let gradEpoch = $state(0);
  const grad = $derived.by(() => {
    void gradEpoch;
    const latest = listExpressions().find((e) => e.id === item.id) ?? item;
    return resolveExprGradient(latest);
  });
  const gradCss = $derived(cssGradientFromColors(grad.colors));
  const swatchDisabled = $derived(isParamDef);

  onMount(() => {
    if (!mfEl) return;
    configureMathField(mfEl, isParamDef ? "Parameter definition" : "Math expression");
    setFieldLatex(mfEl, item.latex || "");
  });

  $effect(() => {
    if (!mfEl || isMathFieldFocused(mfEl)) return;
    const cur = readFieldLatex(mfEl);
    if (cur !== item.latex) setFieldLatex(mfEl, item.latex || "");
  });

  function mathFieldRef() {
    return mfEl ?? null;
  }

  function onSwatchClick(ev: MouseEvent) {
    ev.stopPropagation();
    if (swatchDisabled || !rowEl) return;
    const btn = ev.currentTarget as HTMLButtonElement;
    const live = listExpressions().find((e) => e.id === item.id) ?? item;
    openGradientEditor(btn, live, () => {
      gradEpoch += 1;
      if (onColorChange) onColorChange();
      else onExprChange();
    });
  }

  function stopParamAnimForRowEdit() {
    if (!mfEl) return;
    const classified = classifyKind(readFieldLatex(mfEl));
    if (classified?.kind !== "parameter" || !classified.paramName) return;
    if (!suppressAutoCommit()) commitAutoParams();
    const next = stopParamAnimation(classified.paramName);
    if (next) {
      updateExprSilent(item.id, { sliderAnimating: false });
    }
  }

  function onMfFocus() {
    onSelect(item.id);
    // Carousel navigation focuses fields without editing; keep animation running until input.
    if (disableSplitMerge) return;
    stopParamAnimForRowEdit();
  }

  function onMfBlur() {
    onScheduleCommit(item.id);
  }

  let ignoreFieldInput = false;

  function onMfInput() {
    if (!mfEl || ignoreFieldInput) return;
    if (disableSplitMerge) stopParamAnimForRowEdit();
    updateExpr(item.id, { latex: readFieldLatex(mfEl) });
    onExprChange();
  }

  function syncFieldLatex() {
    if (!mfEl) return;
    updateExpr(item.id, { latex: readFieldLatex(mfEl) });
  }

  function createPendingParams() {
    if (!pendingParams.length) return false;
    syncFieldLatex();
    if (!createParamRows(pendingParams)) return false;
    onStructuralChange();
    onExprChange();
    return true;
  }

  function onPendingParamsClick(ev: MouseEvent) {
    ev.stopPropagation();
    createPendingParams();
  }

  function onMfKeydownCapture(ev: KeyboardEvent) {
    if (!mfEl) return;
    if (ev.key === "Enter" && !ev.shiftKey) {
      if (isSuggestionUiActive(mfEl)) return;
      if (ev.defaultPrevented) return;
      ev.preventDefault();
      ev.stopPropagation();
      const latex = readFieldLatex(mfEl);
      if (isParameterRow(latex)) {
        updateExpr(item.id, { latex });
        onExprChange();
        mfEl.blur?.();
        return;
      }
      if (disableSplitMerge) {
        updateExpr(item.id, { latex });
        onExprChange();
        mfEl.blur?.();
        return;
      }
      const { left, right } = latexAroundCaret(mfEl);
      ignoreFieldInput = true;
      try {
        setFieldLatex(mfEl, left);
        const split = splitExprAt(item.id, left, right);
        onSplit(split ? { id: split.id, pos: 0 } : null);
      } finally {
        scheduleDoubleRaf(() => {
          ignoreFieldInput = false;
        });
      }
      return;
    }
    if (ev.key === "Tab" && !ev.shiftKey) {
      if (isSuggestionUiActive(mfEl)) return;
      if (pendingParams.length > 0) {
        ev.preventDefault();
        ev.stopPropagation();
        createPendingParams();
        return;
      }
    }
    if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
      if (isSuggestionUiActive(mfEl)) return;
      const list = listExpressions();
      const idx = list.findIndex((e) => e.id === item.id);
      const target = ev.key === "ArrowUp" ? list[idx - 1] : list[idx + 1];
      if (target) {
        ev.preventDefault();
        ev.stopPropagation();
        updateExpr(item.id, { latex: readFieldLatex(mfEl) });
        onFocusNav(target.id, getCaretPos(mfEl));
      }
      return;
    }
    if (ev.key === "Backspace" && isCursorAtStart(mfEl)) {
      if (disableSplitMerge) return;
      const idx = listExpressions().findIndex((e) => e.id === item.id);
      if (idx > 0) {
        ev.preventDefault();
        ev.stopPropagation();
        const curLatex = readFieldLatex(mfEl);
        ignoreFieldInput = true;
        try {
          updateExprSilent(item.id, { latex: curLatex });
          const merged = mergeExprIntoPrevious(item.id);
          if (merged) {
            onMerge({ id: merged.id, pos: merged.caretOffset });
          }
        } finally {
          scheduleDoubleRaf(() => {
            ignoreFieldInput = false;
          });
        }
      }
    }
  }

  function onVisClick(ev: MouseEvent) {
    ev.stopPropagation();
    updateExpr(item.id, { enabled: !item.enabled });
    // Every layer's fit is unaffected by which are shown — skip the
    // structural-change path (fresh progressive refit ladder) in favor of a
    // single non-progressive repack that reuses cached layer data.
    if (onVisibilityChange) onVisibilityChange();
    else onStructuralChange();
  }

  function onDelClick(ev: MouseEvent) {
    ev.stopPropagation();
    removeExpr(item.id);
    onStructuralChange();
  }

  function onRowClick(ev: MouseEvent) {
    const t = ev.target;
    if (t instanceof Element && t.closest(".expr-drag, .expr-color, .expr-vis, .expr-del, .expr-param-block, .expr-param-side, .expr-pending-params")) {
      return;
    }
    if (t === mfEl || (mfEl && mfEl.contains(t as Node))) return;
    if (t instanceof HTMLInputElement || t instanceof HTMLButtonElement) return;
    onSelect(item.id);
    if (mfEl) focusAt(getLastOffset(mfEl));
  }

  function onDragPointerdown(ev: PointerEvent) {
    if (rowEl) onDragStart(rowEl, ev);
  }

  export function focusAt(pos: number) {
    if (!mfEl) return;
    try {
      mfEl.focus?.({ preventScroll: true });
    } catch {
      mfEl.focus?.();
    }
    setCaretPos(mfEl, pos);
  }

  export function getRowElement() {
    return rowEl ?? null;
  }

  export function getMathField() {
    return mfEl ?? null;
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  bind:this={rowEl}
  class="expr-row"
  class:selected
  class:is-hidden={!item.enabled}
  class:is-param-def={isParamDef}
  class:has-error={!!rowError}
  class:has-pending-params={pendingParams.length > 0}
  data-id={item.id}
  data-param={paramName ?? undefined}
  style:--expr-grad={gradCss}
  style:--expr-c0={grad.color}
  style:--expr-c1={grad.color2}
  onclick={onRowClick}
>
  <button
    type="button"
    class="expr-drag"
    title="Drag to reorder"
    aria-label="Drag to reorder"
    tabindex={-1}
    onpointerdown={onDragPointerdown}
  >
    ⠿
  </button>

  <button
    type="button"
    class="expr-color"
    disabled={swatchDisabled}
    title={swatchDisabled ? "Parameters are not drawn" : `Edit gradient (${grad.colors.length} colors)`}
    style:--expr-grad={gradCss}
    style:--expr-c0={grad.color}
    style:--expr-c1={grad.color2}
    onclick={onSwatchClick}
  ></button>

  <div class="expr-mid">
    <div class="expr-field-row">
      <math-field
        bind:this={mfEl}
        class="expr-field"
        class:invalid={!!rowError}
        aria-label={isParamDef ? "Parameter definition" : "Math expression"}
        title={rowError ?? undefined}
        onfocus={onMfFocus}
        onblur={onMfBlur}
        oninput={onMfInput}
        onkeydowncapture={onMfKeydownCapture}
      ></math-field>
    </div>
    {#if pendingParams.length}
      <button
        type="button"
        class="expr-pending-params"
        title={pendingErr ?? undefined}
        aria-label={`Create parameter rows for ${pendingParamLabelPlain}. Press Tab.`}
        onclick={onPendingParamsClick}
      >
        <span class="expr-pending-params-copy">
          Create {pendingParams.length === 1 ? "parameter" : "parameters"}
          <strong class="expr-pending-params-names">
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html pendingParamNamesMarkup}{#if pendingParamOverflow}<span class="expr-pending-params-overflow"> +{pendingParamOverflow}</span>{/if}
          </strong>
        </span>
        <kbd class="expr-pending-params-tab">Tab</kbd>
      </button>
    {/if}
    {#if paramName}
      <ParamRail
        bind:this={paramRail}
        {item}
        {paramName}
        {paramTick}
        {onParamChange}
        getMathField={mathFieldRef}
      />
    {/if}
  </div>

  {#if paramName}
    <ParamAnimControls
      {item}
      {paramName}
      {paramTick}
      {onParamChange}
      onAnimSync={() => paramRail?.syncFromParam()}
    />
  {:else}
    <button
      type="button"
      class="expr-vis secondary"
      class:is-off={!item.enabled}
      title={item.enabled ? "Hide" : "Show"}
      aria-label={item.enabled ? "Hide expression" : "Show expression"}
      aria-pressed={item.enabled ? "true" : "false"}
      onclick={onVisClick}
    >
      {@html item.enabled ? ICON_EYE : ICON_EYE_OFF}
    </button>
  {/if}

  <button type="button" class="expr-del secondary" title="Delete" onclick={onDelClick}>×</button>
</div>
