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
  import { stopParamAnimation } from "../../model/params.js";
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
    isMathFieldFocused,
    ICON_EYE,
    ICON_EYE_OFF,
  } from "./helpers.ts";
  import { openGradientEditor } from "./popovers.ts";
  import ParamRail from "./ParamRail.svelte";

  interface Props {
    item: ExprItem;
    selected: boolean;
    paramTick: number;
    suppressAutoCommit: () => boolean;
    onExprChange: () => void;
    onStructuralChange: () => void;
    onColorChange?: () => void;
    onParamChange: () => void;
    onSelect: (id: string) => void;
    onFocusNav: (targetId: string, caret: number) => void;
    onSplit: (focus: { id: string; pos: number } | null) => void;
    onMerge: (focus: { id: string; pos: number }) => void;
    onDragStart: (row: HTMLElement, ev: PointerEvent) => void;
    onScheduleCommit: (fromId: string) => void;
  }

  let {
    item,
    selected,
    paramTick,
    suppressAutoCommit,
    onExprChange,
    onStructuralChange,
    onColorChange,
    onParamChange,
    onSelect,
    onFocusNav,
    onSplit,
    onMerge,
    onDragStart,
    onScheduleCommit,
  }: Props = $props();

  let rowEl: HTMLDivElement | undefined = $state();
  let mfEl: MathfieldElement | undefined = $state();

  const warn = $derived(getExprWarning(item.id));
  const paramName = $derived(neededParamForItem(item));
  const isParamDef = $derived(!!paramName || !!warn);
  const grad = $derived(resolveExprGradient(item));
  const gradCss = $derived(cssGradientFromColors(grad.colors));
  const swatchDisabled = $derived(isParamDef);

  onMount(() => {
    if (!mfEl) return;
    configureMathField(mfEl);
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
    openGradientEditor(btn, item, () => {
      if (onColorChange) onColorChange();
      else onExprChange();
    });
  }

  function onMfFocus() {
    onSelect(item.id);
    const classified = classifyKind(readFieldLatex(mfEl!));
    if (classified?.kind === "parameter" && classified.paramName) {
      if (!suppressAutoCommit()) commitAutoParams();
      const next = stopParamAnimation(classified.paramName);
      if (next) {
        updateExprSilent(item.id, { sliderAnimating: false });
      }
    }
  }

  function onMfBlur() {
    onScheduleCommit(item.id);
  }

  function onMfInput() {
    if (!mfEl) return;
    updateExpr(item.id, { latex: readFieldLatex(mfEl) });
    onExprChange();
  }

  function onMfKeydownCapture(ev: KeyboardEvent) {
    if (!mfEl) return;
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
      const idx = listExpressions().findIndex((e) => e.id === item.id);
      if (idx > 0) {
        ev.preventDefault();
        ev.stopPropagation();
        updateExpr(item.id, { latex: readFieldLatex(mfEl) });
        const merged = mergeExprIntoPrevious(item.id);
        if (merged) {
          onMerge({ id: merged.id, pos: merged.caretOffset });
        }
      }
    }
  }

  function onMfKeydownEnter(ev: KeyboardEvent) {
    if (!mfEl) return;
    if (ev.key !== "Enter" || ev.shiftKey) return;
    if (ev.defaultPrevented) return;
    ev.preventDefault();
    ev.stopPropagation();
    const { left, right } = latexAroundCaret(mfEl);
    updateExpr(item.id, { latex: left });
    const split = splitExprAt(item.id, left, right);
    onSplit(split ? { id: split.id, pos: 0 } : null);
  }

  function onVisClick(ev: MouseEvent) {
    ev.stopPropagation();
    updateExpr(item.id, { enabled: !item.enabled });
    onStructuralChange();
  }

  function onDelClick(ev: MouseEvent) {
    ev.stopPropagation();
    removeExpr(item.id);
    onStructuralChange();
  }

  function onRowClick(ev: MouseEvent) {
    const t = ev.target;
    if (t instanceof Element && t.closest(".expr-drag, .expr-color, .expr-vis, .expr-del, .expr-param-block")) {
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
  class:has-error={!!warn}
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
        class:invalid={!!warn}
        title={warn ?? undefined}
        onfocus={onMfFocus}
        onblur={onMfBlur}
        oninput={onMfInput}
        onkeydown={onMfKeydownEnter}
        onkeydowncapture={onMfKeydownCapture}
      ></math-field>
    </div>
    {#if paramName}
      <ParamRail
        {item}
        {paramName}
        {paramTick}
        {onParamChange}
        getMathField={mathFieldRef}
      />
    {/if}
  </div>

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

  <button type="button" class="expr-del secondary" title="Delete" onclick={onDelClick}>×</button>
</div>
