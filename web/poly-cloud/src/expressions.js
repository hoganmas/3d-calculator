/**
 * Expression list: density / constraint fields and optional named parameters.
 * Parameter rows (`a = …`) share values across all field expressions.
 */

let nextId = 1;

/** Max / min stops in a layer gradient. */
export const MAX_GRAD_STOPS = 6;
export const MIN_GRAD_STOPS = 2;

/** Glassmorphic gradient pairs for densities / isosurfaces. */
export const EXPR_GRADIENTS = [
  { color: "#ff4500", color2: "#ffec00" }, /* flame → gold */
  { color: "#ff1493", color2: "#7b2fff" }, /* deep pink → violet */
  { color: "#0066ff", color2: "#00ffaa" }, /* royal blue → spring green */
  { color: "#ff6b4a", color2: "#ffb0d8" }, /* coral → blush */
  { color: "#00c8e0", color2: "#c080ff" }, /* aqua → lilac glass */
];

/** Primary swatch colors (gradient start). */
export const EXPR_COLORS = EXPR_GRADIENTS.map((g) => g.color);

export const DEFAULT_EXPR_COLOR = EXPR_COLORS[0];
export const DEFAULT_EXPR_COLOR2 = EXPR_GRADIENTS[0].color2;

function gradientForIndex(i) {
  return EXPR_GRADIENTS[i % EXPR_GRADIENTS.length];
}

/** Normalize a list of hex stops (clamp length, ensure ≥2). */
export function normalizeGradColors(input, fallbackPrimary = DEFAULT_EXPR_COLOR) {
  let cols = Array.isArray(input)
    ? input.map((c) => String(c || "").trim()).filter(Boolean)
    : [];
  cols = cols.map((c) => (c.startsWith("#") ? c : `#${c}`));
  if (cols.length === 0) {
    cols = [fallbackPrimary, color2ForPrimary(fallbackPrimary)];
  } else if (cols.length === 1) {
    cols = [cols[0], color2ForPrimary(cols[0])];
  }
  if (cols.length > MAX_GRAD_STOPS) cols = cols.slice(0, MAX_GRAD_STOPS);
  return cols;
}

/**
 * @param {{ color?: string, color2?: string, colors?: string[] }} item
 * @returns {{ colors: string[], color: string, color2: string }}
 */
export function resolveExprGradient(item) {
  const colors = normalizeGradColors(
    item.colors?.length
      ? item.colors
      : [
          item.color ?? DEFAULT_EXPR_COLOR,
          item.color2 ?? color2ForPrimary(item.color ?? DEFAULT_EXPR_COLOR),
        ],
    item.color ?? DEFAULT_EXPR_COLOR,
  );
  return {
    colors,
    color: colors[0],
    color2: colors[colors.length - 1],
  };
}

/** Secondary endpoint when the user edits the primary swatch only. */
export function color2ForPrimary(primary) {
  const hit = EXPR_GRADIENTS.find(
    (g) => g.color.toLowerCase() === String(primary || "").toLowerCase(),
  );
  return hit ? hit.color2 : EXPR_GRADIENTS[0].color2;
}

/** CSS linear-gradient from stop list. */
export function cssGradientFromColors(colors, angle = "145deg") {
  const cols = normalizeGradColors(colors);
  if (cols.length === 1) return cols[0];
  const stops = cols
    .map((c, i) => `${c} ${((i / (cols.length - 1)) * 100).toFixed(1)}%`)
    .join(", ");
  return `linear-gradient(${angle}, ${stops})`;
}

function colorForIndex(i) {
  return gradientForIndex(i).color;
}

/** Next gradient pair after the last expression. */
export function nextExprGradient() {
  if (!items.length) return { ...EXPR_GRADIENTS[0], colors: [EXPR_GRADIENTS[0].color, EXPR_GRADIENTS[0].color2] };
  const last = resolveExprGradient(items[items.length - 1]);
  const idx = EXPR_GRADIENTS.findIndex((g) => g.color === last.color);
  const g = gradientForIndex(idx >= 0 ? idx + 1 : items.length);
  return { ...g, colors: [g.color, g.color2] };
}

/** Next palette color after the last expression (or index 0 if empty). */
export function nextExprColor() {
  return nextExprGradient().color;
}

/**
 * @typedef {"auto" | "density" | "constraint"} ExprRole
 * @typedef {{
 *   id: string,
 *   latex: string,
 *   color: string,
 *   color2: string,
 *   colors: string[],
 *   role: ExprRole,
 *   enabled: boolean,
 *   sliderMin: number,
 *   sliderMax: number,
 *   sliderSpeed: number,
 *   sliderAnimating: boolean,
 *   sliderPhase: number,
 *   autoParam: boolean,
 * }} ExprItem
 */

/** @type {ExprItem[]} */
let items = [];

/** @type {string | null} */
let selectedId = null;

/** @type {(() => void) | null} */
let onChange = null;

/** Per-row compile warnings (e.g. duplicate variable). @type {Map<string, string>} */
let exprWarnings = new Map();

export function setExpressionsOnChange(fn) {
  onChange = fn;
}

/** @param {[string, string][] | Map<string, string> | Iterable<[string, string]>} entries */
export function replaceExprWarnings(entries) {
  exprWarnings = new Map(entries);
}

/** @param {string} id */
export function getExprWarning(id) {
  return exprWarnings.get(id) ?? null;
}

function emit() {
  if (onChange) onChange();
}

/**
 * @param {Partial<ExprItem> & { animate?: boolean, min?: number, max?: number, speed?: number, value?: number }} [init]
 * @returns {ExprItem}
 */
export function createExprItem(init = {}) {
  const id = init.id ?? `e${nextId++}`;
  let sliderMin = Number.isFinite(init.sliderMin)
    ? init.sliderMin
    : Number.isFinite(init.min)
      ? init.min
      : -10;
  let sliderMax = Number.isFinite(init.sliderMax)
    ? init.sliderMax
    : Number.isFinite(init.max)
      ? init.max
      : 10;
  if (sliderMax < sliderMin) [sliderMin, sliderMax] = [sliderMax, sliderMin];
  const grad = resolveExprGradient(
    init.colors?.length || init.color
      ? { color: init.color, color2: init.color2, colors: init.colors }
      : nextExprGradient(),
  );
  return {
    id,
    latex: init.latex ?? "",
    color: grad.color,
    color2: grad.color2,
    colors: grad.colors,
    role: init.role ?? "auto",
    enabled: init.enabled !== false,
    sliderMin,
    sliderMax,
    sliderSpeed:
      Number.isFinite(init.sliderSpeed) && init.sliderSpeed > 0
        ? init.sliderSpeed
        : Number.isFinite(init.speed) && init.speed > 0
          ? init.speed
          : 0.35,
    sliderAnimating: !!(init.sliderAnimating ?? init.animate),
    sliderPhase: Number.isFinite(init.sliderPhase) ? init.sliderPhase : Math.random(),
    autoParam: !!init.autoParam,
  };
}

/** @returns {ExprItem[]} */
export function listExpressions() {
  return items.slice();
}

/** @returns {ExprItem | null} */
export function getSelectedExpr() {
  return items.find((e) => e.id === selectedId) ?? null;
}

export function getSelectedId() {
  return selectedId;
}

/** @param {string | null} id */
export function selectExpr(id) {
  selectedId = id;
  emit();
}

/**
 * Replace the whole list (e.g. preset / reset).
 * @param {Partial<ExprItem>[]} list
 */
export function setExpressions(list) {
  items = list.map((init, i) =>
    createExprItem({
      ...init,
      color: init.color ?? colorForIndex(i),
    }),
  );
  selectedId = items[0]?.id ?? null;
  emit();
}

/**
 * Insert a row at `index` without selecting it or notifying listeners.
 * Used when auto-creating parameter declarations during compile.
 * @param {number} index
 * @param {Partial<ExprItem>} [init]
 */
export function insertExprAt(index, init = {}) {
  const at = Math.max(0, Math.min(items.length, index | 0));
  const row = createExprItem({
    ...init,
    color: init.color ?? colorForIndex(at),
  });
  items.splice(at, 0, row);
  return row;
}

/**
 * Insert a new expression after `afterId` (or at end). Selects the new row.
 * @param {string | null} [afterId]
 * @param {Partial<ExprItem>} [init]
 */
export function insertExprAfter(afterId = selectedId, init = {}) {
  const idx = afterId ? items.findIndex((e) => e.id === afterId) : items.length - 1;
  const at = idx >= 0 ? idx + 1 : items.length;
  const row = createExprItem({
    ...init,
    color: init.color ?? colorForIndex(at),
  });
  items.splice(at, 0, row);
  selectedId = row.id;
  emit();
  return row;
}

/** @param {string} id */
export function removeExpr(id) {
  const idx = items.findIndex((e) => e.id === id);
  if (idx < 0) return;
  items.splice(idx, 1);
  if (selectedId === id) {
    selectedId = items[Math.min(idx, items.length - 1)]?.id ?? null;
  }
  if (items.length === 0) {
    const row = createExprItem({ latex: "" });
    items.push(row);
    selectedId = row.id;
  }
  emit();
}

/** Remove without notifying listeners (compile-time auto-param prune). */
export function removeExprSilent(id) {
  const idx = items.findIndex((e) => e.id === id);
  if (idx < 0) return false;
  items.splice(idx, 1);
  if (selectedId === id) {
    selectedId = items[Math.min(idx, items.length - 1)]?.id ?? null;
  }
  if (items.length === 0) {
    const row = createExprItem({ latex: "" });
    items.push(row);
    selectedId = row.id;
  }
  return true;
}

/** Mark auto-created param rows as permanent (user left the editing expression). */
export function commitAutoParams() {
  for (const item of items) {
    if (item.autoParam) item.autoParam = false;
  }
}

/**
 * Move expression `id` so it sits before `beforeId` (or at end if `beforeId` is null).
 * @param {string} id
 * @param {string | null} beforeId
 * @returns {boolean} true if order changed
 */
export function moveExpr(id, beforeId) {
  const from = items.findIndex((e) => e.id === id);
  if (from < 0) return false;
  if (beforeId === id) return false;
  let to = beforeId == null ? items.length : items.findIndex((e) => e.id === beforeId);
  if (to < 0) to = items.length;
  // Index after removing `from`.
  const dest = from < to ? to - 1 : to;
  if (dest === from) return false;
  const [row] = items.splice(from, 1);
  items.splice(dest, 0, row);
  emit();
  return true;
}

/**
 * Merge row `id` into the one above (concat LaTeX). Keeps the upper row's color/role.
 * @returns {{ id: string, caretOffset: number } | null}
 */
export function mergeExprIntoPrevious(id) {
  const idx = items.findIndex((e) => e.id === id);
  if (idx <= 0) return null;
  const prev = items[idx - 1];
  const cur = items[idx];
  const caretOffset = prev.latex.length;
  prev.latex = prev.latex + cur.latex;
  items.splice(idx, 1);
  selectedId = prev.id;
  emit();
  return { id: prev.id, caretOffset };
}

/**
 * Split row `id` at a latex boundary into left (same row) + right (new row below).
 * @returns {{ id: string, caretOffset: number } | null}
 */
export function splitExprAt(id, leftLatex, rightLatex) {
  const idx = items.findIndex((e) => e.id === id);
  if (idx < 0) return null;
  items[idx].latex = leftLatex ?? "";
  const row = createExprItem({
    latex: rightLatex ?? "",
    color: colorForIndex(idx + 1),
  });
  items.splice(idx + 1, 0, row);
  selectedId = row.id;
  emit();
  return { id: row.id, caretOffset: 0 };
}

/**
 * @param {string} id
 * @param {Partial<ExprItem>} patch
 */
export function updateExpr(id, patch) {
  const row = items.find((e) => e.id === id);
  if (!row) return null;
  Object.assign(row, patch);
  if (patch.colors || patch.color != null || patch.color2 != null) {
    const g = resolveExprGradient({
      colors: patch.colors ?? row.colors,
      color: patch.color ?? row.color,
      color2: patch.color2 ?? row.color2,
    });
    row.colors = g.colors;
    row.color = g.color;
    row.color2 = g.color2;
  }
  emit();
  return row;
}

/** Like updateExpr but does not notify listeners (animation / silent latex sync). */
export function updateExprSilent(id, patch) {
  const row = items.find((e) => e.id === id);
  if (!row) return null;
  Object.assign(row, patch);
  if (patch.colors || patch.color != null || patch.color2 != null) {
    const g = resolveExprGradient({
      colors: patch.colors ?? row.colors,
      color: patch.color ?? row.color,
      color2: patch.color2 ?? row.color2,
    });
    row.colors = g.colors;
    row.color = g.color;
    row.color2 = g.color2;
  }
  return row;
}

/** Hex → [r,g,b] in 0..1 */
export function hexToRgb01(hex) {
  const s = String(hex || "").replace("#", "");
  const n = s.length === 3
    ? s.split("").map((c) => parseInt(c + c, 16))
    : [parseInt(s.slice(0, 2), 16), parseInt(s.slice(2, 4), 16), parseInt(s.slice(4, 6), 16)];
  if (n.some((v) => !Number.isFinite(v))) return [0.55, 0.75, 1.0];
  return n.map((v) => v / 255);
}

/**
 * Effective role from expression kind.
 * @param {ExprRole} _role
 * @param {"parameter" | "constraint" | "definition" | "bare"} kind
 * @returns {"parameter" | "density" | "constraint"}
 */
export function resolveExprRole(_role, kind) {
  if (kind === "parameter") return "parameter";
  return kind === "constraint" ? "constraint" : "density";
}
