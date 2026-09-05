/**
 * Right-click "info" context menu: a small glass popover showing a description
 * for a settings control, in place of inline hint text.
 */
import { mountFloatingPopover, type PopoverHandle } from "./popover.js";

let openMenu: PopoverHandle | null = null;

export function closeInfoMenu() {
  openMenu?.destroy();
  openMenu = null;
}

export function attachInfoContextMenu(target: HTMLElement | null | undefined, text: string) {
  if (!(target instanceof HTMLElement)) return;
  target.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    closeInfoMenu();

    const pop = document.createElement("div");
    pop.className = "info-menu";
    pop.setAttribute("role", "note");
    pop.textContent = text;

    openMenu = mountFloatingPopover(target, pop, {
      placement: "top-start",
      onOutside: () => closeInfoMenu(),
    });
  });
}
