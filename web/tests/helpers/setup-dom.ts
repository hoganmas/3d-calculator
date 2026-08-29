/**
 * Minimal DOM stub so app modules (dom.ts, compile.ts) load in Node tests.
 * Import this module before any app-layer imports.
 */

if (typeof globalThis.HTMLElement === "undefined") {
  globalThis.HTMLElement = class HTMLElement {} as typeof HTMLElement;
}

function makeElement(id: string, tag = "div"): HTMLElement {
  const node = {
    id,
    tagName: tag.toUpperCase(),
    value: "",
    textContent: "",
    hidden: false,
    clientWidth: 800,
    clientHeight: 600,
    appendChild() {},
    addEventListener() {},
    removeEventListener() {},
    contains() {
      return false;
    },
    closest() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
    dataset: {},
    classList: {
      add() {},
      remove() {},
      toggle() {},
      contains() {
        return false;
      },
    },
  };
  if (id === "deg") node.value = "20";
  if (id === "scale") node.value = "2.5";
  if (id === "steps") node.value = "16";
  if (id === "boxSize") node.value = "5";
  if (id === "marchDownscale") node.value = "2";
  if (id === "preset") node.value = "sincos";
  return node as unknown as HTMLElement;
}

const elementCache = new Map<string, HTMLElement>();

function getElement(id: string): HTMLElement {
  if (!elementCache.has(id)) elementCache.set(id, makeElement(id));
  return elementCache.get(id)!;
}

if (typeof globalThis.document === "undefined") {
  globalThis.document = {
    getElementById(id: string) {
      return getElement(id);
    },
    createElement(tag: string) {
      return makeElement(`_${tag}`, tag);
    },
    activeElement: null,
  } as unknown as Document;
}

if (typeof globalThis.window === "undefined") {
  globalThis.window = globalThis as unknown as Window & typeof globalThis;
}

/** Mock a focused MathLive field for shouldDeferAutoParamRows tests. */
export function setMockFocusedMathField(opts: { paramDefRow?: boolean } = {}) {
  const row = {
    classList: {
      contains(cls: string) {
        return opts.paramDefRow === true && cls === "is-param-def";
      },
    },
  };
  const mf = Object.create(HTMLElement.prototype) as HTMLElement & {
    tagName: string;
    contains: () => boolean;
    closest: (sel: string) => unknown;
  };
  mf.tagName = "MATH-FIELD";
  mf.contains = () => false;
  mf.closest = (sel: string) => (sel === ".expr-row" ? row : null);
  (document as Document & { activeElement: Element | null }).activeElement = mf;
}

export function clearMockFocusedMathField() {
  (document as Document & { activeElement: Element | null }).activeElement = null;
}
