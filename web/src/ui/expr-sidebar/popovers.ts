import {
  DEFAULT_EXPR_COLOR,
  normalizeGradColors,
  resolveExprGradient,
  cssGradientFromColors,
  MAX_GRAD_STOPS,
  MIN_GRAD_STOPS,
  updateExpr,
  updateExprSilent,
} from "../../model/expressions.js";
import {
  getParam,
  updateParam,
  phaseForValue,
  normalizeAnimMode,
} from "../../model/params.js";
import { mountFloatingPopover, type PopoverHandle } from "../popover.js";
import type { ExprItem } from "../../types/models.js";
import {
  ANIM_SPEED_MAX,
  ANIM_SPEED_MIN,
  ANIM_SPEED_STEP,
  fmtAnimSpeed,
} from "./helpers.js";
import { mountLiquidThumb, syncLiquidThumb } from "../liquidSlider.js";

let openGrad: PopoverHandle | null = null;
let openAnim: PopoverHandle | null = null;

export function closeAllPopovers() {
  openGrad?.destroy();
  openAnim?.destroy();
  openGrad = null;
  openAnim = null;
}

export function openGradientEditor(
  anchor: HTMLElement,
  item: ExprItem,
  onColorChange: () => void,
) {
  closeAllPopovers();
  const grad = resolveExprGradient(item);
  let draft = grad.colors.slice();

  const pop = document.createElement("div");
  pop.className = "grad-popover";
  pop.setAttribute("role", "dialog");
  pop.setAttribute("aria-label", "Edit gradient");

  const head = document.createElement("div");
  head.className = "grad-popover-head";
  head.textContent = "Gradient colors";

  const preview = document.createElement("div");
  preview.className = "grad-popover-preview";

  const list = document.createElement("div");
  list.className = "grad-popover-stops";

  const actions = document.createElement("div");
  actions.className = "grad-popover-actions";

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "secondary";
  addBtn.textContent = "Add color";

  function commit(next: string[]) {
    draft = normalizeGradColors(next);
    updateExpr(item.id, { colors: draft });
    onColorChange();
    renderStops();
  }

  function renderStops() {
    preview.style.background = cssGradientFromColors(draft);
    list.replaceChildren();
    draft.forEach((hex, i) => {
      const stop = document.createElement("div");
      stop.className = "grad-stop";

      const pick = document.createElement("input");
      pick.type = "color";
      pick.className = "grad-stop-color";
      pick.value = hex.startsWith("#") ? hex : DEFAULT_EXPR_COLOR;
      pick.title = `Stop ${i + 1}`;
      pick.addEventListener("input", () => {
        const next = draft.slice();
        next[i] = pick.value;
        commit(next);
      });
      pick.addEventListener("click", (ev) => ev.stopPropagation());

      const label = document.createElement("span");
      label.className = "grad-stop-label";
      label.textContent = i === 0 ? "Start" : i === draft.length - 1 ? "End" : `Stop ${i + 1}`;

      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "grad-stop-remove secondary";
      rm.textContent = "×";
      rm.title = "Remove stop";
      rm.disabled = draft.length <= MIN_GRAD_STOPS;
      rm.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (draft.length <= MIN_GRAD_STOPS) return;
        const next = draft.slice();
        next.splice(i, 1);
        commit(next);
      });

      stop.append(pick, label, rm);
      list.append(stop);
    });
    addBtn.disabled = draft.length >= MAX_GRAD_STOPS;
    addBtn.title =
      draft.length >= MAX_GRAD_STOPS
        ? `Max ${MAX_GRAD_STOPS} colors`
        : "Add a gradient stop";
  }

  addBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (draft.length >= MAX_GRAD_STOPS) return;
    const last = draft[draft.length - 1] || DEFAULT_EXPR_COLOR;
    commit([...draft, last]);
  });

  actions.append(addBtn);
  pop.append(head, preview, list, actions);

  openGrad = mountFloatingPopover(anchor, pop, {
    placement: "bottom-start",
    onOutside: () => closeAllPopovers(),
    outsideIgnore: (t) => t instanceof Element && !!t.closest(".expr-color"),
  });
  renderStops();
}

export function openAnimOptions(
  anchor: HTMLElement,
  item: ExprItem,
  paramName: string,
  onParamChange: () => void,
  syncRow: () => void,
) {
  closeAllPopovers();
  const p = getParam(paramName);
  if (!p || p.driven) return;

  const pop = document.createElement("div");
  pop.className = "anim-popover";
  pop.dataset.param = paramName;
  pop.setAttribute("role", "dialog");
  pop.setAttribute("aria-label", "Animation options");

  const head = document.createElement("div");
  head.className = "anim-popover-head";
  head.textContent = "Animation";

  const modes = document.createElement("div");
  modes.className = "anim-popover-modes";
  modes.setAttribute("role", "group");
  modes.setAttribute("aria-label", "Curve");

  const modeDefs = [
    { mode: "pingpong" as const, label: "Back & forth" },
    { mode: "loop" as const, label: "Loop" },
  ];
  const modeBtns: HTMLButtonElement[] = [];

  const speedBlock = document.createElement("div");
  speedBlock.className = "anim-popover-speed";
  const speedRow = document.createElement("div");
  speedRow.className = "anim-popover-speed-row";
  const speedLabel = document.createElement("span");
  speedLabel.textContent = "Speed";
  const speedVal = document.createElement("span");
  speedVal.className = "anim-popover-speed-val";
  const speedTrack = document.createElement("div");
  speedTrack.className = "anim-popover-speed-track expr-param-track";
  const speed = document.createElement("input");
  speed.type = "range";
  speed.className = "expr-param-slider";
  speed.min = String(ANIM_SPEED_MIN);
  speed.max = String(ANIM_SPEED_MAX);
  speed.step = String(ANIM_SPEED_STEP);
  speed.title = "Cycles per second";

  function applyAnimPatch(patch: { animMode?: "pingpong" | "loop"; speed?: number }) {
    const cur = getParam(paramName);
    if (!cur) return;
    const timeSec = performance.now() / 1000;
    const nextMode = patch.animMode != null ? normalizeAnimMode(patch.animMode) : cur.animMode;
    const nextSpeed =
      Number.isFinite(patch.speed) && patch.speed! > 0 ? patch.speed! : cur.speed;
    const next = updateParam(paramName, {
      animMode: nextMode,
      speed: nextSpeed,
      phase: phaseForValue({ ...cur, animMode: nextMode, speed: nextSpeed }, timeSec),
    });
    if (!next) return;
    updateExprSilent(item.id, {
      sliderAnimMode: next.animMode,
      sliderSpeed: next.speed,
      sliderPhase: next.phase,
    });
    syncRow();
    syncPopoverFromParam(next);
    onParamChange();
  }

  function syncPopoverFromParam(st: NonNullable<ReturnType<typeof getParam>>) {
    const mode = normalizeAnimMode(st.animMode);
    for (const btn of modeBtns) {
      btn.classList.toggle("on", btn.dataset.mode === mode);
    }
    if (document.activeElement !== speed) speed.value = String(st.speed);
    speedVal.textContent = `${fmtAnimSpeed(st.speed)}×`;
    syncLiquidThumb(speed);
  }

  for (const def of modeDefs) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = def.label;
    btn.dataset.mode = def.mode;
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      applyAnimPatch({ animMode: def.mode });
    });
    modeBtns.push(btn);
    modes.append(btn);
  }

  speed.addEventListener("input", () => {
    const n = Number(speed.value);
    if (!Number.isFinite(n) || n <= 0) return;
    applyAnimPatch({ speed: n });
  });
  speed.addEventListener("click", (ev) => ev.stopPropagation());
  speedTrack.append(speed);
  mountLiquidThumb(speedTrack, speed);
  speedRow.append(speedLabel, speedVal);
  speedBlock.append(speedRow, speedTrack);
  pop.append(head, modes, speedBlock);

  syncPopoverFromParam(p);

  openAnim = mountFloatingPopover(anchor, pop, {
    placement: "bottom-end",
    onOutside: () => closeAllPopovers(),
    outsideIgnore: (t) => t instanceof Element && !!t.closest(".expr-param-anim-opts"),
  });
}

export function isAnimPopoverOpen(paramName: string): boolean {
  return openAnim?.pop.dataset.param === paramName;
}

export function closeAnimPopover() {
  if (openAnim) {
    openAnim.destroy();
    openAnim = null;
  }
}
