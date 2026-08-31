/** Minimal MathLive custom element typings. */
declare namespace JSX {
  interface IntrinsicElements {
    "math-field": import("mathlive").MathfieldElementAttributes & {
      ref?: (el: MathfieldElement | null) => void;
    };
  }
}

declare global {
  interface MathfieldElement extends HTMLElement {
    value?: string;
    position?: number;
    selection?: { ranges: [number, number][] };
    lastOffset?: number;
    selectionStart?: number;
    mode?: string;
    mathVirtualKeyboardPolicy?: string;
    menuItems?: unknown[];
    inlineShortcutTimeout?: number;
    inlineShortcuts?: Record<string, string | { value: string; after?: string }>;
    overrideDefaultInlineShortcuts?: boolean;
    mathModeSpace?: string;
    smartMode?: boolean;
    macros?: Record<string, string | object>;
    contains(node: Node | null): boolean;
    getValue(
      start?: number | "latex",
      end?: number,
      format?: string,
    ): string;
    setValue(value: string, options?: { silenceNotifications?: boolean }): void;
    executeCommand?(command: string | [string, ...unknown[]]): void;
    focus(options?: { preventScroll?: boolean }): void;
    shadowRoot: ShadowRoot | null;
  }

  interface HTMLElementTagNameMap {
    "math-field": MathfieldElement;
  }
}

export {};
