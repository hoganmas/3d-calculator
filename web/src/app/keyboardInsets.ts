/**
 * Keep the mobile layout above the OS keyboard by tracking visualViewport.
 * iOS keeps `100dvh` tall while the keyboard covers the bottom; we mirror the
 * visible viewport as `--vv-height` / `--vv-offset-top` on the root layout.
 */
import { isHorizontalPanelLayout } from "./panelLayout.js";

export const VIEWPORT_SYNC_EVENT = "laplaci:viewport-sync";

const KEYBOARD_OPEN_PX = 80;

/** Covered strip between layout viewport bottom and visible viewport bottom. */
export function keyboardBottomInsetFrom(
  innerHeight: number,
  vvHeight: number,
  vvOffsetTop: number,
): number {
  return Math.max(0, innerHeight - (vvHeight + vvOffsetTop));
}

export function keyboardTopInsetFrom(vvOffsetTop: number): number {
  return Math.max(0, vvOffsetTop);
}

function clearVvVars(root: HTMLElement): void {
  for (const key of ["--vv-height", "--vv-offset-top", "--vv-top", "--vv-bottom"]) {
    root.style.removeProperty(key);
  }
}

function findFocusedExprRow(): HTMLElement | null {
  const active = document.activeElement;
  if (!(active instanceof Element)) return null;
  const direct = active.closest?.(".expr-row");
  if (direct instanceof HTMLElement) return direct;
  const root = active.getRootNode();
  if (root instanceof ShadowRoot) {
    const host = root.host;
    if (host instanceof Element) {
      const fromHost = host.closest(".expr-row");
      if (fromHost instanceof HTMLElement) return fromHost;
    }
  }
  return null;
}

export function syncKeyboardInsets(): void {
  const root = document.documentElement;
  if (!isHorizontalPanelLayout()) {
    clearVvVars(root);
    delete root.dataset.keyboardOpen;
    return;
  }

  const vv = window.visualViewport;
  if (!vv) return;

  const bottom = keyboardBottomInsetFrom(window.innerHeight, vv.height, vv.offsetTop);
  const top = keyboardTopInsetFrom(vv.offsetTop);
  root.style.setProperty("--vv-height", `${vv.height}px`);
  root.style.setProperty("--vv-offset-top", `${top}px`);
  root.style.setProperty("--vv-top", `${top}px`);
  root.style.setProperty("--vv-bottom", `${bottom}px`);
  if (bottom >= KEYBOARD_OPEN_PX) root.dataset.keyboardOpen = "true";
  else delete root.dataset.keyboardOpen;

  window.dispatchEvent(new Event(VIEWPORT_SYNC_EVENT));
}

/** Scroll the focused expression row into the visible list / footer area. */
export function scrollFocusedExprIntoView(): void {
  if (!isHorizontalPanelLayout()) return;
  const row = findFocusedExprRow();
  if (!row) return;

  if (row.closest(".mobile-expr-footer")) {
    row.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    return;
  }

  const list = document.getElementById("exprList");
  if (!(list instanceof HTMLElement)) {
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    return;
  }
  const rowRect = row.getBoundingClientRect();
  const listRect = list.getBoundingClientRect();
  const pad = 12;
  if (rowRect.top < listRect.top + pad) {
    list.scrollTop -= listRect.top + pad - rowRect.top;
  } else if (rowRect.bottom > listRect.bottom - pad) {
    list.scrollTop += rowRect.bottom - (listRect.bottom - pad);
  }
}

export function initKeyboardInsets(): void {
  syncKeyboardInsets();

  const onChange = () => {
    syncKeyboardInsets();
    if (document.documentElement.dataset.keyboardOpen === "true") {
      requestAnimationFrame(() => scrollFocusedExprIntoView());
    }
  };

  window.addEventListener("resize", onChange);
  const vv = window.visualViewport;
  if (vv) {
    vv.addEventListener("resize", onChange);
    vv.addEventListener("scroll", onChange);
  }

  document.addEventListener(
    "focusin",
    () => {
      syncKeyboardInsets();
      requestAnimationFrame(() => {
        syncKeyboardInsets();
        scrollFocusedExprIntoView();
        window.setTimeout(() => {
          syncKeyboardInsets();
          scrollFocusedExprIntoView();
        }, 280);
      });
    },
    true,
  );
}
