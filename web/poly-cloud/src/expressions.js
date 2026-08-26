/**
 * Named parameter expression list: equation, color, density vs constraint role.
 * Free parameters are shared across all expressions (union of symbols).
 */

let nextId = 1;

/** Named parameter default palette. */
export const EXPR_COLORS = [
  "#c74440",
  "#2d70b3",
  "#388c46",
  "#6042a6",
  "#fa7e19",
  "#000000",
  "#00a2b3",
];

/**
 * @typedef {"auto" | "density" | "constraint"} ExprRole
 * @typedef {{
 *   id: string,
 *   latex: string,
 *   color: string,
 *   role: ExprRole,
 *   enabled: boolean,
 * }} ExprItem
 */

/** @type {ExprItem[]} */
let items = [];

/** @type {string | null} */
let selectedId = null;

/** @type {(() => void) | null} */
let onChange = null;

export function setExpressionsOnChange(fn) {
  onChange = fn;
}

function emit() {
  if (onChange) onChange();
}

function colorForIndex(i) {
  return EXPR_COLORS[i % EXPR_COLORS.length];
}

/**
 * @param {Partial<ExprItem>} [init]
 * @returns {ExprItem}
 */
export function createExprItem(init = {}) {
  const id = init.id ?? `e${nextId++}`;
  return {
    id,
    latex: init.latex ?? "",
    color: init.color ?? colorForIndex(items.length),
    role: init.role ?? "auto",
    enabled: init.enabled !== false,
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
 * Insert a new expression after `afterId` (or at end). Selects the new row.
 * @param {string | null} [afterId]
 * @param {Partial<ExprItem>} [init]
 */
export function insertExprAfter(afterId = selectedId, init = {}) {
  const row = createExprItem(init);
  const idx = afterId ? items.findIndex((e) => e.id === afterId) : items.length - 1;
  const at = idx >= 0 ? idx + 1 : items.length;
  items.splice(at, 0, row);
  // Re-tint if color not explicit
  if (!init.color) row.color = colorForIndex(at);
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

/**
 * @param {string} id
 * @param {Partial<ExprItem>} patch
 */
export function updateExpr(id, patch) {
  const row = items.find((e) => e.id === id);
  if (!row) return null;
  Object.assign(row, patch);
  emit();
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
 * Resolve effective role from auto-classify + override.
 * @param {ExprRole} role
 * @param {"constraint" | "definition" | "bare"} kind
 * @returns {"density" | "constraint"}
 */
export function resolveExprRole(role, kind) {
  if (role === "density" || role === "constraint") return role;
  return kind === "constraint" ? "constraint" : "density";
}
