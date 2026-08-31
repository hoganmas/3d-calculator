<script lang="ts">
  import { onMount } from "svelte";
  import { convertLatexToMarkup } from "mathlive";
  import type { ExprItem } from "../../types/models.js";
  import {
    listExpressions,
    getSelectedId,
    selectExpr,
    updateExpr,
    removeExpr,
    splitExprAt,
    commitAutoParams,
    resolveExprGradient,
    cssGradientFromColors,
    getExprWarning,
  } from "../../model/expressions.js";
  import { getParam } from "../../model/params.js";
  import ParamRail from "./ParamRail.svelte";
  import { openGradientEditor } from "./popovers.ts";
  import {
    neededParamForItem,
    isParameterRow,
    ICON_EYE,
    ICON_EYE_OFF,
  } from "./helpers.ts";
  import {
    collectPendingParamsForExpr,
    createParamRows,
    formatPendingParamNamesLatex,
    formatPendingParamOverflow,
    formatPendingParamLabelPlain,
    pendingParamErrorMessage,
  } from "../../app/pendingParams.js";
  import { isMobileExprUi } from "./mobileExprUi.ts";
  import { isPanelCollapsed } from "../../app/panelLayout.js";

  interface Props {
    paramTick?: number;
    onExprChange: () => void;
    onStructuralChange: () => void;
    onColorChange?: () => void;
    onParamChange: () => void;
    onSelectionSync?: () => void;
    onReturnToViewport?: () => void;
  }

  let {
    paramTick = 0,
    onExprChange,
    onStructuralChange,
    onColorChange,
    onParamChange,
    onSelectionSync,
    onReturnToViewport,
  }: Props = $props();

  let items: ExprItem[] = $state([]);
  let index = $state(0);
  let expanded = $state(false);
  let draftLatex = $state("");
  let sourceEl: HTMLTextAreaElement | undefined = $state();
  let swipeStartX = 0;
  let swipeActive = false;
  let mobileUi = $state(isMobileExprUi());

  const currentItem = $derived(items[index] ?? null);

  const previewMarkup = $derived.by(() => {
    const latex = draftLatex.trim() || "\\text{\\;}\\text{empty expression}";
    try {
      return convertLatexToMarkup(latex, { defaultMode: "math" });
    } catch {
      return latex.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
  });

  const paramName = $derived.by(() => {
    void paramTick;
    return currentItem ? neededParamForItem(currentItem, paramTick) : null;
  });

  const pendingParams = $derived.by(() => {
    void paramTick;
    return currentItem ? collectPendingParamsForExpr(currentItem) : [];
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
  const paramErr = $derived.by(() => {
    void paramTick;
    return paramName ? getParam(paramName)?.error ?? null : null;
  });
  const rowError = $derived(
    (currentItem ? getExprWarning(currentItem.id) : null) ?? paramErr ?? pendingErr,
  );
  const isParamDef = $derived(isParameterRow(draftLatex));
  const grad = $derived(currentItem ? resolveExprGradient(currentItem) : null);
  const gradCss = $derived(grad ? cssGradientFromColors(grad.colors) : "transparent");
  const swatchDisabled = $derived(isParamDef);

  function syncIndexFromSelection() {
    const sid = getSelectedId();
    if (!sid) return;
    const i = items.findIndex((e) => e.id === sid);
    if (i >= 0) index = i;
  }

  function syncDraftFromItem(force = false) {
    if (!currentItem) return;
    if (!force && expanded && sourceEl && document.activeElement === sourceEl) return;
    draftLatex = currentItem.latex ?? "";
  }

  export function syncFromList() {
    items = listExpressions();
    if (index >= items.length) index = Math.max(0, items.length - 1);
    syncIndexFromSelection();
    syncDraftFromItem();
  }

  function commitDraft() {
    if (!currentItem) return;
    const trimmed = draftLatex;
    if (trimmed !== currentItem.latex) {
      updateExpr(currentItem.id, { latex: trimmed });
      onExprChange();
    }
  }

  function go(delta: number) {
    if (!items.length) return;
    commitDraft();
    const next = Math.max(0, Math.min(items.length - 1, index + delta));
    if (next === index) return;
    index = next;
    selectExpr(items[index]!.id);
    onSelectionSync?.();
    syncDraftFromItem(true);
  }

  function onSwipePointerDown(ev: PointerEvent) {
    if (expanded) return;
    swipeStartX = ev.clientX;
    swipeActive = true;
    (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
  }

  function onSwipePointerUp(ev: PointerEvent) {
    if (!swipeActive || expanded) return;
    swipeActive = false;
    const dx = ev.clientX - swipeStartX;
    if (Math.abs(dx) >= 48) go(dx < 0 ? 1 : -1);
  }

  function openExpanded() {
    if (!currentItem) return;
    onReturnToViewport?.();
    syncDraftFromItem(true);
    expanded = true;
    document.documentElement.classList.add("mobile-expr-expanded");
    queueMicrotask(() => sourceEl?.focus({ preventScroll: true }));
  }

  function closeExpanded() {
    commitDraft();
    expanded = false;
    document.documentElement.classList.remove("mobile-expr-expanded");
    sourceEl?.blur();
    onReturnToViewport?.();
  }

  function onSourceInput() {
    if (!currentItem) return;
    updateExpr(currentItem.id, { latex: draftLatex });
    onExprChange();
  }

  function onCreatePendingParams() {
    if (!pendingParams.length) return;
    commitDraft();
    if (!createParamRows(pendingParams)) return;
    onStructuralChange();
    onExprChange();
  }

  function toggleVisibility(ev?: MouseEvent) {
    ev?.stopPropagation();
    if (!currentItem) return;
    updateExpr(currentItem.id, { enabled: !currentItem.enabled });
    items = listExpressions();
    onStructuralChange();
  }

  function deleteCurrent() {
    if (!currentItem) return;
    commitAutoParams();
    const id = currentItem.id;
    removeExpr(id);
    onStructuralChange();
    items = listExpressions();
    index = Math.min(index, Math.max(0, items.length - 1));
    syncDraftFromItem(true);
    if (!items.length) closeExpanded();
  }

  function addExpression() {
    commitDraft();
    const list = listExpressions();
    items = list;
    const last = list[list.length - 1];
    if (last && !String(last.latex || "").trim()) {
      index = list.length - 1;
      selectExpr(last.id);
      syncDraftFromItem(true);
      openExpanded();
      return;
    }
    if (!currentItem) return;
    const split = splitExprAt(currentItem.id, currentItem.latex, "");
    if (!split) return;
    onStructuralChange();
    items = listExpressions();
    const i = items.findIndex((e) => e.id === split.id);
    index = i >= 0 ? i : items.length - 1;
    syncDraftFromItem(true);
    openExpanded();
  }

  function onSwatchClick(ev: MouseEvent) {
    ev.stopPropagation();
    if (!currentItem || swatchDisabled) return;
    openGradientEditor(ev.currentTarget as HTMLElement, currentItem, () => {
      if (onColorChange) onColorChange();
      else onExprChange();
    });
  }

  function onLayoutChange() {
    mobileUi = isMobileExprUi();
    if (!mobileUi && expanded) closeExpanded();
  }

  function onPanelCollapsed() {
    if (!isPanelCollapsed() && expanded) closeExpanded();
  }

  onMount(() => {
    items = listExpressions();
    syncIndexFromSelection();
    syncDraftFromItem(true);
    const mq = window.matchMedia("(max-width: 800px)");
    mq.addEventListener("change", onLayoutChange);
    window.addEventListener("laplaci:panel-collapsed", onPanelCollapsed);
    return () => {
      mq.removeEventListener("change", onLayoutChange);
      window.removeEventListener("laplaci:panel-collapsed", onPanelCollapsed);
    };
  });
</script>

{#if mobileUi}
  {#if expanded && currentItem}
    <div class="mobile-expr-sheet" role="dialog" aria-modal="true" aria-label="Edit expression">
      <header class="mobile-expr-sheet-head">
        <button type="button" class="icon-btn mobile-expr-done" aria-label="Done" onclick={closeExpanded}>
          <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.75">
            <path stroke-linecap="round" d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>
        <span class="mobile-expr-sheet-index">{index + 1} / {items.length}</span>
        <span class="mobile-expr-head-spacer" aria-hidden="true"></span>
      </header>

      <div
        class="expr-row mobile-expr-row selected"
        class:is-hidden={!currentItem.enabled}
        class:is-param-def={isParamDef}
        class:has-error={!!rowError}
        class:has-pending-params={pendingParams.length > 0}
        style:--expr-grad={gradCss}
        style:--expr-c0={grad?.color}
        style:--expr-c1={grad?.color2}
      >
        <button
          type="button"
          class="expr-color"
          disabled={swatchDisabled}
          title={swatchDisabled ? "Parameters are not drawn" : "Edit gradient"}
          aria-label="Edit gradient"
          onclick={onSwatchClick}
        ></button>

        <div class="expr-mid">
          <div class="mobile-expr-preview" class:has-error={!!rowError} aria-live="polite">
            <!-- eslint-disable-next-line svelte/no-at-html-tags -->
            {@html previewMarkup}
          </div>
          {#if rowError}
            <p class="mobile-expr-error">{rowError}</p>
          {/if}

          <label class="mobile-expr-source-label">
            <span>Source</span>
            <textarea
              bind:this={sourceEl}
              class="mobile-expr-source expr-field"
              class:invalid={!!rowError}
              bind:value={draftLatex}
              rows={4}
              spellcheck="false"
              autocapitalize="off"
              autocomplete="off"
              aria-label={isParamDef ? "Parameter definition" : "Expression source"}
              oninput={onSourceInput}
            ></textarea>
          </label>

          {#if pendingParams.length}
            <button
              type="button"
              class="expr-pending-params"
              title={pendingErr ?? undefined}
              aria-label={`Create parameter rows for ${pendingParamLabelPlain}`}
              onclick={onCreatePendingParams}
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
              item={currentItem}
              {paramName}
              {paramTick}
              {onParamChange}
              getMathField={() => null}
            />
          {/if}
        </div>

        <button
          type="button"
          class="expr-vis secondary"
          class:is-off={!currentItem.enabled}
          title={currentItem.enabled ? "Hide" : "Show"}
          aria-label={currentItem.enabled ? "Hide expression" : "Show expression"}
          aria-pressed={currentItem.enabled ? "true" : "false"}
          onclick={toggleVisibility}
        >
          {@html currentItem.enabled ? ICON_EYE : ICON_EYE_OFF}
        </button>

        <button type="button" class="expr-del secondary" title="Delete" aria-label="Delete" onclick={deleteCurrent}>
          ×
        </button>
      </div>
    </div>
  {/if}

  <footer
    class="mobile-expr-footer"
    aria-label="Expression carousel"
    onpointerdown={onSwipePointerDown}
    onpointerup={onSwipePointerUp}
    onpointercancel={() => {
      swipeActive = false;
    }}
  >
    <button
      type="button"
      class="mobile-expr-nav"
      aria-label="Previous expression"
      disabled={index <= 0}
      onclick={() => go(-1)}
    >
      ‹
    </button>

    {#if currentItem}
      <button
        type="button"
        class="expr-color mobile-expr-footer-swatch"
        disabled={swatchDisabled}
        aria-label="Edit gradient"
        style:--expr-grad={gradCss}
        style:--expr-c0={grad?.color}
        style:--expr-c1={grad?.color2}
        onclick={onSwatchClick}
      ></button>
    {/if}

    <button type="button" class="mobile-expr-chip" onclick={openExpanded} disabled={!currentItem}>
      <span class="mobile-expr-chip-preview" aria-hidden="true">
        <!-- eslint-disable-next-line svelte/no-at-html-tags -->
        {@html previewMarkup}
      </span>
      <span class="mobile-expr-chip-meta">
        <span class="mobile-expr-chip-hint">Tap to edit</span>
        <span class="mobile-expr-counter">{items.length ? index + 1 : 0} / {items.length}</span>
      </span>
    </button>

    {#if currentItem}
      <button
        type="button"
        class="expr-vis secondary mobile-expr-footer-vis"
        class:is-off={!currentItem.enabled}
        aria-label={currentItem.enabled ? "Hide expression" : "Show expression"}
        onclick={toggleVisibility}
      >
        {@html currentItem.enabled ? ICON_EYE : ICON_EYE_OFF}
      </button>
    {/if}

    <button
      type="button"
      class="mobile-expr-nav"
      aria-label="Next expression"
      disabled={index >= items.length - 1}
      onclick={() => go(1)}
    >
      ›
    </button>

    <button type="button" class="mobile-expr-add secondary" aria-label="Add expression" onclick={addExpression}>
      +
    </button>
  </footer>
{/if}
