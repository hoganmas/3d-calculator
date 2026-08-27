<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import type { ExprItem } from "../../types/models.js";
  import {
    listExpressions,
    getSelectedId,
    selectExpr,
    commitAutoParams,
  } from "../../model/expressions.js";
  import ExprRow from "./ExprRow.svelte";
  import { DragReorderController } from "./dragReorder.ts";
  import { closeAllPopovers } from "./popovers.ts";
  import { getCaretPos } from "./helpers.ts";

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
  let suppressAutoCommit = 0;
  let focusEpoch = 0;
  let pendingFocus: { id: string; pos: number } | null = null;

  const rowRefs: Record<string, ExprRow | undefined> = {};
  let dragCtrl: DragReorderController | null = null;

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

  function isSuppressingAutoCommit() {
    return suppressAutoCommit > 0;
  }

  function captureFocus(): { id: string; pos: number } | null {
    const active = document.activeElement;
    if (!active) return null;

    for (const item of items) {
      const row = rowRefs[item.id];
      const mf = row?.getMathField() ?? null;
      if (!mf) continue;

      let hit = active === mf || (typeof mf.contains === "function" && mf.contains(active));
      if (!hit) {
        try {
          hit = !!(mf.shadowRoot && mf.shadowRoot.contains(active));
        } catch {
          hit = false;
        }
      }
      if (!hit) {
        try {
          const path = typeof active.composedPath === "function" ? active.composedPath() : [];
          hit = path.includes(mf);
        } catch {
          /* ignore */
        }
      }
      if (hit) return { id: item.id, pos: getCaretPos(mf) };
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
    queueMicrotask(() => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (suppressAutoCommit) return;
          const snap = captureFocus();
          if (snap?.id === fromExprId) return;
          commitAutoParams();
          onExprChange();
        });
      });
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
    restoreFocus(focusSnap, epoch);
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

  onMount(() => {
    items = listExpressions();
    selectedId = getSelectedId();
    if (listRoot) {
      dragCtrl = new DragReorderController(listRoot, {
        onReorderStart: () => {
          commitAutoParams();
          listRoot?.classList.add("is-reordering");
        },
        onReorderEnd: () => {
          listRoot?.classList.remove("is-reordering");
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

<div class="expr-list" bind:this={listRoot} aria-label="Expressions">
  {#each items as item (item.id)}
    <ExprRow
      bind:this={rowRefs[item.id]}
      {item}
      selected={item.id === selectedId}
      {paramTick}
      suppressAutoCommit={isSuppressingAutoCommit}
      onExprChange={() => onExprChange()}
      onStructuralChange={handleStructuralChange}
      {onColorChange}
      onParamChange={() => (onParamChange ?? onExprChange)()}
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
</div>
