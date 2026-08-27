/**
 * Floating UI positioning for fixed popovers (gradient editor, anim options).
 */
import {
  autoUpdate,
  computePosition,
  flip,
  offset,
  shift,
  type Placement,
} from "@floating-ui/dom";

export interface PopoverHandle {
  pop: HTMLElement;
  destroy: () => void;
}

/**
 * Mount `pop` in `document.body`, position against `anchor`, keep updated on scroll/resize.
 */
export function mountFloatingPopover(
  anchor: HTMLElement,
  pop: HTMLElement,
  options: {
    placement?: Placement;
    offsetPx?: number;
    onOutside?: (ev: PointerEvent) => void;
    outsideIgnore?: (target: EventTarget | null) => boolean;
  } = {},
): PopoverHandle {
  const placement = options.placement ?? "bottom-start";
  const gap = options.offsetPx ?? 6;

  pop.style.position = "fixed";
  pop.style.left = "0";
  pop.style.top = "0";
  document.body.append(pop);

  const update = async () => {
    const { x, y } = await computePosition(anchor, pop, {
      placement,
      middleware: [offset(gap), flip(), shift({ padding: 8 })],
    });
    pop.style.left = `${x}px`;
    pop.style.top = `${y}px`;
  };

  const stop = autoUpdate(anchor, pop, update);
  void update();

  const onOutside = (ev: PointerEvent) => {
    const t = ev.target;
    if (!(t instanceof Node)) return;
    if (pop.contains(t)) return;
    if (options.outsideIgnore?.(t)) return;
    options.onOutside?.(ev);
  };

  if (options.onOutside) {
    document.addEventListener("pointerdown", onOutside, true);
  }

  return {
    pop,
    destroy: () => {
      stop();
      if (options.onOutside) {
        document.removeEventListener("pointerdown", onOutside, true);
      }
      pop.remove();
    },
  };
}
