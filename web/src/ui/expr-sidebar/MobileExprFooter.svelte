<script lang="ts">
  import { onMount } from "svelte";
  import type { ExprItem } from "../../types/models.js";
  import {
    listExpressions,
    getSelectedId,
    selectExpr,
    updateExpr,
    commitAutoParams,
  } from "../../model/expressions.js";
  import ExprRow from "./ExprRow.svelte";
  import { readFieldLatex, neededParamForItem, forceReflow } from "./helpers.ts";
  import { getParam } from "../../model/params.js";
  import {
    createSuppressAutoCommitCounter,
    scheduleCommitIfLeftExpr as scheduleAutoCommitIfLeftExpr,
  } from "./autoCommit.ts";
  import { isMobileExprUi } from "./mobileExprUi.ts";

  interface Props {
    paramTick?: number;
    onExprChange: () => void;
    onStructuralChange: () => void;
    onColorChange?: () => void;
    onVisibilityChange?: () => void;
    onParamChange: () => void;
    onSelectionSync?: () => void;
  }

  let {
    paramTick: externalParamTick = 0,
    onExprChange,
    onStructuralChange,
    onColorChange,
    onVisibilityChange,
    onParamChange,
    onSelectionSync,
  }: Props = $props();

  let items: ExprItem[] = $state([]);
  let index = $state(0);
  let localParamTick = $state(0);
  let mobileUi = $state(isMobileExprUi());
  let footerEl: HTMLElement | undefined = $state();
  let viewportEl: HTMLElement | undefined = $state();
  let viewportWidth = $state(0);

  let rowRefs = $state<Record<string, ExprRow | undefined>>({});

  const suppressCtrl = createSuppressAutoCommitCounter();
  let dragX = $state(0);
  let dragging = $state(false);
  let swipeStartX = 0;
  let swipeStartY = 0;
  let swipePointerId: number | null = null;
  let swipeLocked: "none" | "pending" | "horizontal" | "vertical" = "none";

  const paramTick = $derived(externalParamTick + localParamTick);
  const trackTransform = $derived(
    viewportWidth > 0
      ? `translateX(${-index * viewportWidth + dragX}px)`
      : `translateX(${dragX}px)`,
  );

  function isSuppressingAutoCommit() {
    return suppressCtrl.isActive();
  }

  function syncIndexFromSelection() {
    const sid = getSelectedId();
    if (!sid) return;
    const i = items.findIndex((e) => e.id === sid);
    if (i >= 0) index = i;
  }

  function syncViewportWidth() {
    if (!viewportEl) return;
    const w = Math.round(viewportEl.clientWidth);
    viewportWidth = w;
    viewportEl.style.setProperty("--mobile-expr-slide-w", `${w}px`);
  }

  function syncViewportHeight() {
    if (!viewportEl) return;
    const slide = viewportEl.querySelector('.mobile-expr-slide:not([aria-hidden="true"])');
    if (!(slide instanceof HTMLElement)) return;
    viewportEl.style.height = `${Math.ceil(slide.getBoundingClientRect().height)}px`;
    forceReflow(viewportEl);
    syncFooterHeight();
  }

  function syncFooterHeight() {
    if (!footerEl) return;
    const h = Math.ceil(footerEl.getBoundingClientRect().height);
    document.documentElement.style.setProperty("--mobile-expr-footer-h", `${h}px`);
  }

  export function syncFromList() {
    items = listExpressions();
    if (index >= items.length) index = Math.max(0, items.length - 1);
    syncIndexFromSelection();
    queueMicrotask(() => {
      syncViewportWidth();
      syncViewportHeight();
    });
  }

  export function syncAllParamSliders() {
    items = listExpressions();
    localParamTick++;
  }

  export function syncParamChrome(): boolean {
    const latest = listExpressions();
    if (items.length !== latest.length) return false;
    for (let i = 0; i < latest.length; i++) {
      if (items[i].id !== latest[i].id) return false;
    }
    items = latest;
    syncIndexFromSelection();
    localParamTick++;
    return true;
  }

  function slideMounted(i: number) {
    if (Math.abs(i - index) <= 1) return true;
    const item = items[i];
    if (!item) return false;
    const name = neededParamForItem(item, paramTick);
    return !!(name && getParam(name)?.animating);
  }

  function currentRow() {
    const id = items[index]?.id;
    return id ? rowRefs[id] : undefined;
  }

  function commitCurrentField() {
    const row = currentRow();
    const item = items[index];
    const mf = row?.getMathField();
    if (!mf || !item) return;
    const latex = readFieldLatex(mf);
    // updateExpr() invalidates this layer's bake fingerprint whenever the
    // patch touches `latex`, even to the same value — skip it on a plain
    // navigation (every swipe calls this) so leaving a slide untouched
    // doesn't mark it dirty for the next fit pass.
    if (latex === item.latex) return;
    updateExpr(item.id, { latex });
  }

  function scheduleCommitIfLeftExpr(fromExprId: string) {
    scheduleAutoCommitIfLeftExpr(fromExprId, {
      isSuppressingAutoCommit: () => suppressCtrl.isActive(),
      getFocusedExprId: () => {
        for (const item of items) {
          const mf = rowRefs[item.id]?.getMathField();
          if (mf && document.activeElement === mf) return item.id;
        }
        return null;
      },
      commitAutoParams,
      onExprChange,
    });
  }

  function refreshAfterStructural(focus?: { id: string; pos: number } | null) {
    onStructuralChange();
    items = listExpressions();
    if (focus?.id) {
      const i = items.findIndex((e) => e.id === focus.id);
      if (i >= 0) index = i;
      selectExpr(focus.id);
      queueMicrotask(() => rowRefs[focus.id]?.focusAt(focus.pos));
    } else {
      index = Math.min(index, Math.max(0, items.length - 1));
      syncIndexFromSelection();
    }
    onSelectionSync?.();
    localParamTick++;
    queueMicrotask(syncViewportHeight);
  }

  /** Item count/order is unaffected by a visibility toggle — skip the
   * index/focus reconciliation refreshAfterStructural needs for add/remove. */
  function refreshAfterVisibility() {
    if (onVisibilityChange) onVisibilityChange();
    else onStructuralChange();
    items = listExpressions();
    onSelectionSync?.();
    localParamTick++;
  }

  function goTo(next: number) {
    if (next === index || next < 0 || next >= items.length || dragging) return;
    commitCurrentField();
    commitAutoParams();
    dragX = 0;
    index = next;
    selectExpr(items[next]!.id);
    onSelectionSync?.();
    localParamTick++;
    queueMicrotask(syncViewportHeight);
  }

  /** Block swipe only on explicit controls — not the math field (gesture lock handles that). */
  function swipeTargetBlocked(target: EventTarget | null) {
    if (!(target instanceof Element)) return false;
    return !!target.closest(
      "button, input, textarea, select, .expr-param-block, .expr-param-side, .expr-pending-params, .mobile-expr-dot, .liquid-range, .liquid-thumb",
    );
  }

  function rubberBand(dx: number) {
    if (items.length <= 1) return dx * 0.25;
    if (index === 0 && dx > 0) return dx * 0.25;
    if (index >= items.length - 1 && dx < 0) return dx * 0.25;
    return dx;
  }

  // Set only while a swipe gesture holds the suppress counter up (see below) —
  // lets resetSwipe() release exactly the suppression this gesture acquired,
  // regardless of how/when it fires (pointerup, cancel, or a vertical bail).
  let suppressingForSwipe = false;

  function resetSwipe() {
    if (viewportEl && swipePointerId != null) {
      try {
        viewportEl.releasePointerCapture(swipePointerId);
      } catch {
        /* ignore */
      }
    }
    dragging = false;
    dragX = 0;
    swipeLocked = "none";
    swipePointerId = null;
    if (suppressingForSwipe) {
      suppressingForSwipe = false;
      suppressCtrl.end();
    }
  }

  function onSwipePointerDown(ev: PointerEvent) {
    if (ev.button !== 0 || swipeTargetBlocked(ev.target)) return;
    swipeStartX = ev.clientX;
    swipeStartY = ev.clientY;
    swipePointerId = ev.pointerId;
    swipeLocked = "pending";
    dragX = 0;
    (ev.currentTarget as HTMLElement).setPointerCapture(ev.pointerId);
  }

  function onSwipePointerMove(ev: PointerEvent) {
    if (swipePointerId !== ev.pointerId || swipeLocked === "none") return;

    const dx = ev.clientX - swipeStartX;
    const dy = ev.clientY - swipeStartY;

    if (swipeLocked === "pending") {
      if (Math.abs(dx) < 8 && Math.abs(dy) < 8) return;
      if (Math.abs(dx) > Math.abs(dy) * 1.15) {
        swipeLocked = "horizontal";
        dragging = true;
        const active = document.activeElement;
        if (active instanceof HTMLElement && active.closest(".mobile-expr-footer")) {
          // This blur is a swipe-gesture technicality (stop the focused field
          // from eating the drag), not the user leaving the field — without
          // suppression it still trips ExprRow's onMfBlur -> scheduleCommitIfLeftExpr,
          // which calls the real onExprChange (a full scheduleUploadFit rebuild)
          // a couple of rAFs later on every swipe that happened to blur a field.
          suppressCtrl.begin();
          suppressingForSwipe = true;
          active.blur();
        }
        ev.preventDefault();
      } else {
        swipeLocked = "vertical";
        resetSwipe();
        return;
      }
    }

    if (!dragging) return;
    ev.preventDefault();
    dragX = rubberBand(dx);
  }

  function onSwipePointerUp(ev: PointerEvent) {
    if (swipePointerId !== ev.pointerId) return;

    const wasDragging = dragging;
    const dx = dragX;
    resetSwipe();

    if (!wasDragging) return;

    const threshold = Math.min(52, viewportWidth * 0.16);
    let next = index;
    if (dx < -threshold && index < items.length - 1) next = index + 1;
    else if (dx > threshold && index > 0) next = index - 1;

    if (next !== index) {
      commitCurrentField();
      commitAutoParams();
      index = next;
      selectExpr(items[next]!.id);
      onSelectionSync?.();
      localParamTick++;
      queueMicrotask(syncViewportHeight);
    }
  }

  function onLayoutChange() {
    mobileUi = isMobileExprUi();
    queueMicrotask(() => {
      syncViewportWidth();
      syncViewportHeight();
    });
  }

  onMount(() => {
    items = listExpressions();
    syncIndexFromSelection();

    const mq = window.matchMedia("(max-width: 800px)");
    mq.addEventListener("change", onLayoutChange);

    return () => {
      mq.removeEventListener("change", onLayoutChange);
    };
  });

  $effect(() => {
    if (!viewportEl) return;
    // The viewport is overflow:hidden and positioned only via the track's
    // translateX transform — it must never actually scroll. But it's still a
    // valid scroll container, so focusing a control inside a slide (e.g. the
    // param slider) makes the browser auto-scroll it into view natively,
    // leaving scrollLeft stuck nonzero forever after (nothing else resets
    // it), which then desyncs every subsequent index*viewportWidth transform
    // from where content actually renders. Snap any such scroll back to 0.
    const onScroll = () => {
      if (viewportEl!.scrollLeft !== 0) viewportEl!.scrollLeft = 0;
    };
    viewportEl.addEventListener("scroll", onScroll, { passive: true });
    return () => viewportEl?.removeEventListener("scroll", onScroll);
  });

  $effect(() => {
    if (!footerEl || typeof ResizeObserver === "undefined") return;
    syncFooterHeight();
    const ro = new ResizeObserver(() => syncFooterHeight());
    ro.observe(footerEl);
    return () => ro.disconnect();
  });

  $effect(() => {
    if (!viewportEl || typeof ResizeObserver === "undefined") return;
    syncViewportWidth();
    const ro = new ResizeObserver(() => syncViewportWidth());
    ro.observe(viewportEl);
    return () => ro.disconnect();
  });

  $effect(() => {
    void index;
    void paramTick;
    void items.length;
    if (!viewportEl || typeof ResizeObserver === "undefined") return;
    const card = viewportEl.querySelector(
      '.mobile-expr-slide:not([aria-hidden="true"]) .mobile-expr-panel-card',
    );
    if (!(card instanceof HTMLElement)) {
      queueMicrotask(syncViewportHeight);
      return;
    }
    syncViewportHeight();
    const ro = new ResizeObserver(() => syncViewportHeight());
    ro.observe(card);
    return () => ro.disconnect();
  });
</script>

{#if mobileUi}
  <footer bind:this={footerEl} class="mobile-expr-footer" aria-label="Expression carousel">
    {#if items.length > 0}
      <div class="mobile-expr-dots" role="tablist" aria-label="Expression index">
        {#each items as item, i (item.id)}
          <button
            type="button"
            class="mobile-expr-dot"
            class:active={i === index}
            role="tab"
            aria-selected={i === index}
            aria-label={`Expression ${i + 1} of ${items.length}`}
            disabled={dragging}
            onclick={() => goTo(i)}
          ></button>
        {/each}
      </div>
    {/if}

    <div
      bind:this={viewportEl}
      class="mobile-expr-viewport"
      class:is-dragging={dragging}
      role="region"
      aria-label="Swipe between expressions"
      aria-roledescription="carousel"
      onpointerdown={onSwipePointerDown}
      onpointermove={onSwipePointerMove}
      onpointerup={onSwipePointerUp}
      onpointercancel={resetSwipe}
    >
      <div
        class="mobile-expr-track"
        class:no-transition={dragging}
        style:transform={trackTransform}
      >
        {#each items as item, i (item.id)}
          <div class="mobile-expr-slide" aria-hidden={i !== index}>
            <div class="mobile-expr-panel-card">
              <button
                type="button"
                class="mobile-expr-edge-nav secondary"
                aria-label="Previous expression"
                disabled={i <= 0 || dragging}
                onclick={() => goTo(i - 1)}
              >
                ‹
              </button>

              <div class="mobile-expr-row-wrap">
                {#if slideMounted(i)}
                  <ExprRow
                    bind:this={rowRefs[item.id]}
                    {item}
                    selected={i === index}
                    disableSplitMerge={true}
                    {paramTick}
                    suppressAutoCommit={isSuppressingAutoCommit}
                    onExprChange={() => {
                      localParamTick++;
                      onExprChange();
                      queueMicrotask(syncViewportHeight);
                    }}
                    onStructuralChange={() => refreshAfterStructural()}
                    {onColorChange}
                    onVisibilityChange={refreshAfterVisibility}
                    onParamChange={() => {
                      localParamTick++;
                      (onParamChange ?? onExprChange)();
                      queueMicrotask(syncViewportHeight);
                    }}
                    onSelect={(id) => {
                      selectExpr(id);
                      const idx = items.findIndex((e) => e.id === id);
                      if (idx >= 0) goTo(idx);
                    }}
                    onFocusNav={(targetId, caret) => {
                      const idx = items.findIndex((e) => e.id === targetId);
                      if (idx >= 0) goTo(idx);
                      queueMicrotask(() => rowRefs[targetId]?.focusAt(caret));
                      onSelectionSync?.();
                    }}
                    onSplit={(focus) => refreshAfterStructural(focus)}
                    onMerge={(focus) => refreshAfterStructural(focus)}
                    onDragStart={() => {}}
                    onScheduleCommit={scheduleCommitIfLeftExpr}
                  />
                {/if}
              </div>

              <button
                type="button"
                class="mobile-expr-edge-nav secondary"
                aria-label="Next expression"
                disabled={i >= items.length - 1 || dragging}
                onclick={() => goTo(i + 1)}
              >
                ›
              </button>
            </div>
          </div>
        {/each}
      </div>
    </div>
  </footer>
{/if}
