/**
 * Keep the mobile panel above the OS keyboard by tracking visualViewport.
 * iOS keeps `100dvh` / layout viewport tall while the keyboard covers the bottom;
 * visualViewport.height shrinks — we mirror that as CSS insets.
 */
import { isHorizontalPanelLayout } from "./panelLayout.js";

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

function keyboardBottomInset(): number {
  const vv = window.visualViewport;
  if (!vv) return 0;
  return keyboardBottomInsetFrom(window.innerHeight, vv.height, vv.offsetTop);
}

function keyboardTopInset(): number {
  const vv = window.visualViewport;
  if (!vv) return 0;
  return keyboardTopInsetFrom(vv.offsetTop);
}

export function syncKeyboardInsets(): void {
  const root = document.documentElement;
  const bottom = keyboardBottomInset();
  const top = keyboardTopInset();
  root.style.setProperty("--vv-top", `${top}px`);
  root.style.setProperty("--vv-bottom", `${bottom}px`);
  if (bottom >= KEYBOARD_OPEN_PX) root.dataset.keyboardOpen = "true";
  else delete root.dataset.keyboardOpen;
}

/** Scroll the focused expression row into the visible list area. */
export function scrollFocusedExprIntoView(): void {
  if (!isHorizontalPanelLayout()) return;
  const active = document.activeElement;
  if (!(active instanceof Element)) return;
  const row =
    active.closest?.(".expr-row") ||
    (active.getRootNode() instanceof ShadowRoot
      ? (active.getRootNode() as ShadowRoot).host?.closest?.(".expr-row")
      : null);
  if (!(row instanceof HTMLElement)) return;
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
