<script lang="ts">
  import { onDestroy, onMount, tick } from "svelte";
  import type { ExprItem } from "../../types/models.js";
  import {
    listExpressions,
    getSelectedId,
    selectExpr,
    commitAutoParams,
  } from "../../model/expressions.js";
  import { clearAllExprs } from "../../app/compile.js";
  import ExprRow from "./ExprRow.svelte";
  import { DragReorderController } from "./dragReorder.ts";
  import { closeAllPopovers } from "./popovers.ts";
  import { getCaretPos, isMathFieldFocused } from "./helpers.ts";
  import {
    createSuppressAutoCommitCounter,
    scheduleCommitIfLeftExpr as scheduleAutoCommitIfLeftExpr,
  } from "./autoCommit.ts";
  import { isMobileExprUi } from "./mobileExprUi.ts";
  import { isPanelCollapsed } from "../../app/panelLayout.js";

  interface Props {
    onExprChange: () => void;
    onStructuralChange: () => void;
    onColorChange?: () => void;
    onParamChange?: () => void;
  }

  let {
    onExprChange,
    onStructuralChange,
    onColorChange,
    onParamChange,
  }: Props = $props();

  let listRoot: HTMLDivElement | undefined = $state();
  let items: ExprItem[] = $state([]);
  let selectedId: string | null = $state(null);
  let paramTick = $state(0);
  const suppressCtrl = createSuppressAutoCommitCounter();
  let focusEpoch = 0;
  let pendingFocus: { id: string; pos: number } | null = null;

  let rowRefs = $state<Record<string, ExprRow | undefined>>({});
  /** Bumped after drag-reorder so Svelte remounts rows (imperative float breaks bindings). */
  let listEpoch = $state(0);
  let dragCtrl: DragReorderController | null = null;

  function beginSuppressAutoCommit() {
    suppressCtrl.begin();
  }

  function endSuppressAutoCommit() {
    suppressCtrl.end();
  }

  function isSuppressingAutoCommit() {
    return suppressCtrl.isActive();
  }

  function captureFocus(): { id: string; pos: number } | null {
    for (const item of items) {
      const row = rowRefs[item.id];
      const mf = row?.getMathField() ?? null;
      if (mf && isMathFieldFocused(mf)) return { id: item.id, pos: getCaretPos(mf) };
    }
    return null;
  }

  function restoreFocus(snap: { id: string; pos?: number } | null, epoch = focusEpoch) {
    if (!snap?.id) return;
    const id = snap.id;
    const pos = snap.pos ?? 0;
    const apply = (clearPending: boolean) => {
      if (epoch !== focusEpoch) return;
      const row = rowRefs[id];
      row?.focusAt(pos);
      selectExpr(id);
      selectedId = id;
      if (clearPending && pendingFocus?.id === id) pendingFocus = null;
    };
    queueMicrotask(() => apply(false));
    requestAnimationFrame(() => requestAnimationFrame(() => apply(true)));
  }

  function scheduleCommitIfLeftExpr(fromExprId: string) {
    scheduleAutoCommitIfLeftExpr(fromExprId, {
      isSuppressingAutoCommit: () => suppressCtrl.isActive(),
      getFocusedExprId: () => captureFocus()?.id ?? null,
      commitAutoParams,
      onExprChange,
    });
  }

  function handleSelect(id: string) {
    selectExpr(id);
    selectedId = id;
  }

  function handleFocusNav(targetId: string, caret: number) {
    handleSelect(targetId);
    rowRefs[targetId]?.focusAt(caret);
  }

  export function render(focusOverride: { id: string; pos?: number } | null = null) {
    closeAllPopovers();
    beginSuppressAutoCommit();
    if (focusOverride?.id != null) {
      pendingFocus = { id: String(focusOverride.id), pos: focusOverride.pos ?? 0 };
    }
    const epoch = ++focusEpoch;
    const focusSnap = pendingFocus ?? captureFocus();
    items = listExpressions();
    selectedId = getSelectedId();
    paramTick++;
    if (!(isMobileExprUi() && isPanelCollapsed())) {
      restoreFocus(focusSnap, epoch);
    }
    endSuppressAutoCommit();
  }

  export function syncAllParamSliders() {
    paramTick++;
  }

  export function syncParamChrome(): boolean {
    const latest = listExpressions();
    if (items.length !== latest.length) return false;
    for (let i = 0; i < latest.length; i++) {
      if (items[i].id !== latest[i].id) return false;
    }
    items = latest;
    selectedId = getSelectedId();
    paramTick++;
    return true;
  }

  function requestRender(focus?: { id: string; pos?: number } | null) {
    render(focus ?? null);
  }

  function handleStructuralChange() {
    onStructuralChange();
    requestRender();
  }

  function handleDragStart(row: HTMLElement, ev: PointerEvent) {
    const id = row.dataset.id;
    if (!id || !dragCtrl) return;
    dragCtrl.beginPointer(row, id, ev);
  }

  function handleClear() {
    commitAutoParams();
    clearAllExprs();
    items = listExpressions();
    selectedId = getSelectedId();
    onExprChange();
    onStructuralChange();
  }

  export function clearAll() {
    handleClear();
  }

  onMount(async () => {
    items = listExpressions();
    selectedId = getSelectedId();
    await tick();
    if (listRoot) {
      dragCtrl = new DragReorderController(listRoot, {
        onReorderStart: () => {
          commitAutoParams();
          listRoot?.classList.add("is-reordering");
        },
        onReorderEnd: () => {
          listRoot?.classList.remove("is-reordering");
          rowRefs = {};
          listEpoch += 1;
        },
        onStructuralChange: () => onStructuralChange(),
        onRender: () => render(),
        getRowElements: () =>
          items
            .map((it) => rowRefs[it.id]?.getRowElement())
            .filter((el): el is HTMLElement => el instanceof HTMLElement),
      });
    }
  });

  onDestroy(() => {
    dragCtrl?.destroy();
    dragCtrl = null;
    closeAllPopovers();
  });
</script>

<div class="expr-sidebar">
  <div class="expr-sidebar-toolbar">
    <button
      type="button"
      class="secondary expr-clear-btn"
      onclick={handleClear}
    >
      Clear
    </button>
  </div>
  <div class="expr-list" bind:this={listRoot} aria-label="Expressions">
  {#key listEpoch}
  {#each items as item (item.id)}
    <ExprRow
      bind:this={rowRefs[item.id]}
      {item}
      selected={item.id === selectedId}
      {paramTick}
      suppressAutoCommit={isSuppressingAutoCommit}
      onExprChange={() => {
        paramTick++;
        onExprChange();
      }}
      onStructuralChange={handleStructuralChange}
      {onColorChange}
      onParamChange={() => {
        paramTick++;
        (onParamChange ?? onExprChange)();
      }}
      onSelect={handleSelect}
      onFocusNav={handleFocusNav}
      onSplit={(focus) => {
        onStructuralChange();
        requestRender(focus);
      }}
      onMerge={(focus) => {
        onStructuralChange();
        requestRender(focus);
      }}
      onDragStart={handleDragStart}
      onScheduleCommit={scheduleCommitIfLeftExpr}
    />
  {/each}
  {/key}
  </div>
</div>
