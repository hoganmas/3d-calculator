/** Minimal MathLive custom element typings. */
declare namespace JSX {
  interface IntrinsicElements {
    "math-field": import("mathlive").MathfieldElementAttributes & {
      ref?: (el: import("mathlive").MathfieldElement | null) => void;
    };
  }
}

interface MathfieldElement extends HTMLElement {
  value?: string;
  position?: number;
  selection?: { ranges: [number, number][] };
  lastOffset?: number;
  selectionStart?: number;
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

declare global {
  interface HTMLElementTagNameMap {
    "math-field": MathfieldElement;
  }
}

export {};
