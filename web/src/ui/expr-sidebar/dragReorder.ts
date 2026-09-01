import { moveExpr, trailingEmptyExprId } from "../../model/expressions.js";

export interface DragState {
  id: string;
  pointerId: number;
  startY: number;
  moved: boolean;
  offsetX: number;
  offsetY: number;
  width: number;
}

export interface DragReorderCallbacks {
  onReorderStart: () => void;
  onReorderEnd: () => void;
  onStructuralChange: () => void;
  onRender: () => void;
  getRowElements: () => HTMLElement[];
}

/**
 * Imperative drag-reorder controller (floats real row DOM node like vanilla).
 */
export class DragReorderController {
  private dragState: DragState | null = null;
  private liveBeforeId: string | null | undefined = undefined;
  private dragPlaceholder: HTMLElement | null = null;
  private dragFloat: HTMLElement | null = null;
  private listRoot: HTMLElement;
  private cb: DragReorderCallbacks;

  constructor(listRoot: HTMLElement, cb: DragReorderCallbacks) {
    this.listRoot = listRoot;
    this.cb = cb;
  }

  get isReordering() {
    return !!this.dragState?.moved;
  }

  get placeholderHeight(): number {
    return this.dragPlaceholder?.offsetHeight ?? 0;
  }

  beginPointer(row: HTMLElement, itemId: string, ev: PointerEvent) {
    if (ev.button !== 0 || this.dragState) return;
    ev.preventDefault();
    ev.stopPropagation();
    const rect = row.getBoundingClientRect();
    this.dragState = {
      id: itemId,
      pointerId: ev.pointerId,
      startY: ev.clientY,
      moved: false,
      offsetX: ev.clientX - rect.left,
      offsetY: ev.clientY - rect.top,
      width: rect.width,
    };
    this.liveBeforeId = undefined;
    window.addEventListener("pointermove", this.onWindowMove);
    window.addEventListener("pointerup", this.onWindowUp);
    window.addEventListener("pointercancel", this.onWindowUp);
    try {
      (ev.currentTarget as HTMLElement)?.setPointerCapture?.(ev.pointerId);
    } catch {
      /* ignore */
    }
  }

  private onWindowMove = (ev: PointerEvent) => {
    if (!this.dragState || ev.pointerId !== this.dragState.pointerId) return;
    if (!this.dragState.moved) {
      if (Math.abs(ev.clientY - this.dragState.startY) < 3) return;
      this.dragState.moved = true;
      this.cb.onReorderStart();
      const row = this.listRoot.querySelector(
        `.expr-row[data-id="${CSS.escape(this.dragState.id)}"]`,
      );
      if (row instanceof HTMLElement) {
        this.beginFloatDrag(row, ev.clientX, ev.clientY);
      }
    } else {
      this.moveDragFloat(ev.clientX, ev.clientY);
    }
    ev.preventDefault();
    this.liveReorderToPointer(ev.clientY);
  };

  private onWindowUp = (ev: PointerEvent) => {
    if (!this.dragState || ev.pointerId !== this.dragState.pointerId) return;
    if (this.dragState.moved) this.liveReorderToPointer(ev.clientY);
    this.finishPointerDrag();
  };

  private rowElements(): HTMLElement[] {
    return this.cb.getRowElements().filter(
      (r) => r.dataset.id && !r.classList.contains("is-drag-floating"),
    );
  }

  private clearDragVisuals() {
    this.listRoot.classList.remove("is-reordering");
    if (this.dragFloat) {
      this.dragFloat.classList.remove("is-drag-floating");
      this.dragFloat.style.cssText = "";
      this.dragFloat = null;
    }
    if (this.dragPlaceholder) {
      this.dragPlaceholder.remove();
      this.dragPlaceholder = null;
    }
    this.listRoot.querySelectorAll(".expr-row").forEach((r) => {
      if (r instanceof HTMLElement) {
        r.style.transition = "";
        r.style.transform = "";
      }
    });
  }

  private moveDragFloat(clientX: number, clientY: number) {
    if (!this.dragFloat || !this.dragState) return;
    this.dragFloat.style.transform = `translate(${clientX - this.dragState.offsetX}px, ${clientY - this.dragState.offsetY}px) scale(1.02)`;
  }

  private beginFloatDrag(row: HTMLElement, clientX: number, clientY: number) {
    const rect = row.getBoundingClientRect();
    const ph = document.createElement("div");
    ph.className = "expr-row-placeholder";
    ph.style.height = `${rect.height}px`;
    ph.setAttribute("aria-hidden", "true");
    row.parentNode?.insertBefore(ph, row);
    this.dragPlaceholder = ph;

    this.dragFloat = row;
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
    this.moveDragFloat(clientX, clientY);

    const next = ph.nextElementSibling;
    this.liveBeforeId =
      next instanceof HTMLElement && next.classList.contains("expr-row")
        ? next.dataset.id ?? null
        : null;
  }

  private resolveBeforeId(clientY: number, excludeId: string): string | null {
    const others = this.rowElements().filter((r) => r.dataset.id !== excludeId);
    if (!others.length) return null;

    const first = others[0];
    const last = others[others.length - 1];
    const firstRect = first.getBoundingClientRect();
    const lastRect = last.getBoundingClientRect();

    if (clientY < firstRect.top + firstRect.height * 0.5) {
      return first.dataset.id ?? null;
    }
    if (clientY >= lastRect.top + lastRect.height * 0.5) {
      return trailingEmptyExprId();
    }
    for (const row of others) {
      const rect = row.getBoundingClientRect();
      if (clientY < rect.top + rect.height * 0.5) {
        return row.dataset.id ?? null;
      }
    }
    return null;
  }

  private flipRows(prevRects: Map<string, DOMRect>) {
    for (const row of this.rowElements()) {
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
      const onEnd = (ev: TransitionEvent) => {
        if (ev.target !== row || ev.propertyName !== "transform") return;
        row.style.transition = "";
        row.removeEventListener("transitionend", onEnd);
      };
      row.addEventListener("transitionend", onEnd);
    }
  }

  private liveReorderToPointer(clientY: number) {
    if (!this.dragState || !this.dragPlaceholder) return;
    const beforeId = this.resolveBeforeId(clientY, this.dragState.id);
    if (beforeId === this.liveBeforeId) return;

    const ph = this.dragPlaceholder;
    const currentNext = ph.nextElementSibling;
    const currentBeforeId =
      currentNext instanceof HTMLElement && currentNext.classList.contains("expr-row")
        ? (currentNext.dataset.id ?? null)
        : null;
    if (currentBeforeId === beforeId) {
      this.liveBeforeId = beforeId;
      return;
    }

    const prevRects = new Map<string, DOMRect>();
    for (const row of this.rowElements()) {
      if (row.dataset.id) prevRects.set(row.dataset.id, row.getBoundingClientRect());
    }

    if (beforeId) {
      const target = this.listRoot.querySelector(
        `.expr-row[data-id="${CSS.escape(beforeId)}"]`,
      );
      if (!(target instanceof HTMLElement)) return;
      this.listRoot.insertBefore(ph, target);
    } else {
      const trailingId = trailingEmptyExprId();
      const trailing = trailingId
        ? this.listRoot.querySelector(`.expr-row[data-id="${CSS.escape(trailingId)}"]`)
        : null;
      if (trailing instanceof HTMLElement) {
        this.listRoot.insertBefore(ph, trailing);
      } else {
        this.listRoot.appendChild(ph);
      }
    }

    this.liveBeforeId = beforeId;
    this.flipRows(prevRects);
  }

  private finishPointerDrag() {
    const state = this.dragState;
    if (!state) return;
    this.dragState = null;
    window.removeEventListener("pointermove", this.onWindowMove);
    window.removeEventListener("pointerup", this.onWindowUp);
    window.removeEventListener("pointercancel", this.onWindowUp);

    if (!state.moved) {
      this.clearDragVisuals();
      return;
    }

    let beforeId: string | null = null;
    if (this.dragPlaceholder) {
      const next = this.dragPlaceholder.nextElementSibling;
      beforeId =
        next instanceof HTMLElement && next.classList.contains("expr-row")
          ? next.dataset.id ?? null
          : null;
    } else if (this.liveBeforeId !== undefined) {
      beforeId = this.liveBeforeId;
    }
    beforeId = beforeId ?? trailingEmptyExprId();
    this.liveBeforeId = undefined;

    if (this.dragFloat && this.dragPlaceholder?.parentNode) {
      this.dragPlaceholder.parentNode.insertBefore(this.dragFloat, this.dragPlaceholder);
    } else if (this.dragFloat?.parentNode === document.body) {
      this.listRoot.appendChild(this.dragFloat);
    }
    this.clearDragVisuals();

    moveExpr(state.id, beforeId);
    this.cb.onStructuralChange();
    this.cb.onReorderEnd();
    this.cb.onRender();
  }

  destroy() {
    this.clearDragVisuals();
    this.dragState = null;
    window.removeEventListener("pointermove", this.onWindowMove);
    window.removeEventListener("pointerup", this.onWindowUp);
    window.removeEventListener("pointercancel", this.onWindowUp);
  }
}
