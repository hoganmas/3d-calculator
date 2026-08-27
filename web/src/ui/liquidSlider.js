/**
 * Liquid-glass range thumb: overlays a morphing blob on a native <input type="range">.
 * While dragging, the blob goops toward the pointer via skew (no rotation).
 */

const THUMB_PX = 18;

/**
 * @param {HTMLInputElement} input
 * @returns {number} 0…1
 */
function valueT(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const val = Number(input.value);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return 0;
  return Math.min(1, Math.max(0, (val - min) / (max - min)));
}

/**
 * @param {HTMLElement} thumb
 * @param {number} t
 * @param {{
 *   skewX?: number,
 *   skewY?: number,
 *   dx?: number,
 *   dy?: number,
 *   dragging?: boolean,
 * }} [morph]
 */
function applyThumb(thumb, t, morph = {}) {
  thumb.style.setProperty("--t", String(t));
  thumb.style.setProperty("--skew-x", `${morph.skewX ?? 0}deg`);
  thumb.style.setProperty("--skew-y", `${morph.skewY ?? 0}deg`);
  thumb.style.setProperty("--dx", `${morph.dx ?? 0}px`);
  thumb.style.setProperty("--dy", `${morph.dy ?? 0}px`);
  thumb.classList.toggle("is-dragging", !!morph.dragging);
}

/**
 * Sync overlay position from the range value (e.g. after programmatic updates).
 * @param {HTMLInputElement | null | undefined} input
 */
export function syncLiquidThumb(input) {
  if (!(input instanceof HTMLInputElement)) return;
  const track = input.closest(".liquid-track");
  const thumb = track?.querySelector(".liquid-thumb");
  if (!(thumb instanceof HTMLElement)) return;
  if (thumb.classList.contains("is-dragging")) {
    thumb.style.setProperty("--t", String(valueT(input)));
    return;
  }
  applyThumb(thumb, valueT(input));
}

/**
 * Ensure a liquid thumb exists in `track` and follows `input`.
 * @param {HTMLElement} track
 * @param {HTMLInputElement} input
 */
export function mountLiquidThumb(track, input) {
  if (!(track instanceof HTMLElement) || !(input instanceof HTMLInputElement)) return;
  track.classList.add("liquid-track");
  input.classList.add("liquid-range");
  track.style.setProperty("--thumb-size", `${THUMB_PX}px`);

  let thumb = track.querySelector(".liquid-thumb");
  if (!(thumb instanceof HTMLElement)) {
    thumb = document.createElement("span");
    thumb.className = "liquid-thumb";
    thumb.setAttribute("aria-hidden", "true");
    thumb.innerHTML =
      '<span class="liquid-thumb-halo"></span><span class="liquid-thumb-core"></span><span class="liquid-thumb-sheen"></span>';
    track.appendChild(thumb);
  }

  if (track.dataset.liquidBound === "1") {
    syncLiquidThumb(input);
    return;
  }
  track.dataset.liquidBound = "1";

  /** @type {{ x: number, y: number } | null} */
  let pointer = null;
  let dragging = false;
  /** @type {{ skewX: number, skewY: number, dx: number, dy: number }} */
  let morph = { skewX: 0, skewY: 0, dx: 0, dy: 0 };
  /** @type {number} */
  let settleRaf = 0;
  /** @type {number} */
  let dragRaf = 0;

  const stopSettle = () => {
    if (settleRaf) cancelAnimationFrame(settleRaf);
    settleRaf = 0;
  };

  const stopDragRaf = () => {
    if (dragRaf) cancelAnimationFrame(dragRaf);
    dragRaf = 0;
  };

  /**
   * Skew + nudge the blob toward the pointer from the track-laid-out thumb center.
   * @returns {typeof morph}
   */
  const morphTowardPointer = () => {
    const t = valueT(input);
    thumb.style.setProperty("--t", String(t));

    if (!pointer) {
      return { skewX: 0, skewY: 0, dx: 0, dy: 0 };
    }

    const trackRect = track.getBoundingClientRect();
    const size = THUMB_PX;
    const cx = trackRect.left + t * (trackRect.width - size) + size / 2;
    const cy = trackRect.top + trackRect.height / 2;
    const ox = pointer.x - cx;
    const oy = pointer.y - cy;
    const dist = Math.hypot(ox, oy);

    if (dist < 0.5) {
      return { skewX: 0, skewY: 0, dx: 0, dy: 0 };
    }

    const nx = ox / dist;
    const ny = oy / dist;
    const reach = Math.min(6.5, dist * 0.24);
    // Skew leans toward the pointer (deg). Clamp so it stays readable.
    const skewAmt = Math.min(26, dist * 0.42);

    return {
      // Vertical offset → horizontal shear; horizontal offset → vertical shear.
      skewX: ny * skewAmt,
      skewY: nx * skewAmt,
      dx: nx * reach,
      dy: ny * reach,
    };
  };

  const paint = (next, isDragging) => {
    morph = next;
    applyThumb(thumb, valueT(input), { ...next, dragging: isDragging });
  };

  const tickDrag = () => {
    dragRaf = 0;
    if (!dragging) return;
    const target = morphTowardPointer();
    paint(
      {
        skewX: morph.skewX * 0.55 + target.skewX * 0.45,
        skewY: morph.skewY * 0.55 + target.skewY * 0.45,
        dx: morph.dx * 0.55 + target.dx * 0.45,
        dy: morph.dy * 0.55 + target.dy * 0.45,
      },
      true,
    );
    dragRaf = requestAnimationFrame(tickDrag);
  };

  const settle = () => {
    stopSettle();
    stopDragRaf();
    const start = performance.now();
    const from = { ...morph };
    const tick = (now) => {
      const u = Math.min(1, (now - start) / 340);
      const e = 1 - (1 - u) ** 3;
      const bounce = Math.sin(u * Math.PI) * (1 - u);
      paint(
        {
          skewX: from.skewX * (1 - e) - from.skewX * bounce * 0.22,
          skewY: from.skewY * (1 - e) - from.skewY * bounce * 0.22,
          dx: from.dx * (1 - e) - from.dx * bounce * 0.28,
          dy: from.dy * (1 - e) - from.dy * bounce * 0.28,
        },
        false,
      );
      if (u < 1) settleRaf = requestAnimationFrame(tick);
      else {
        morph = { skewX: 0, skewY: 0, dx: 0, dy: 0 };
        settleRaf = 0;
        applyThumb(thumb, valueT(input));
      }
    };
    settleRaf = requestAnimationFrame(tick);
  };

  const onPointerMove = (ev) => {
    pointer = { x: ev.clientX, y: ev.clientY };
  };

  const onInput = () => {
    if (dragging) thumb.style.setProperty("--t", String(valueT(input)));
    else syncLiquidThumb(input);
  };

  const onPointerDown = (ev) => {
    stopSettle();
    dragging = true;
    pointer = { x: ev.clientX, y: ev.clientY };
    thumb.classList.add("is-dragging");
    try {
      input.setPointerCapture(ev.pointerId);
    } catch (_) {
      /* ignore */
    }
    paint(morphTowardPointer(), true);
    stopDragRaf();
    dragRaf = requestAnimationFrame(tickDrag);
  };

  const onPointerUp = (ev) => {
    dragging = false;
    pointer = null;
    stopDragRaf();
    try {
      if (ev && typeof ev.pointerId === "number") input.releasePointerCapture(ev.pointerId);
    } catch (_) {
      /* ignore */
    }
    settle();
  };

  input.addEventListener("input", onInput);
  input.addEventListener("pointerdown", onPointerDown);
  input.addEventListener("pointermove", onPointerMove);
  input.addEventListener("pointerup", onPointerUp);
  input.addEventListener("pointercancel", onPointerUp);
  input.addEventListener("lostpointercapture", () => {
    if (!dragging) return;
    dragging = false;
    pointer = null;
    stopDragRaf();
    settle();
  });

  applyThumb(thumb, valueT(input));
}
