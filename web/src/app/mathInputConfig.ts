/** Dev UI wiring for curated MathLive inline shortcuts. */

import {
  isCuratedShortcutsEnabled,
  resetCuratedShortcutsCache,
  setCuratedShortcutsState,
} from "./curatedShortcutsState.js";
import { reconfigureAllMathfields } from "../ui/expr-sidebar/mathfieldConfig.js";

export { isCuratedShortcutsEnabled } from "./curatedShortcutsState.js";

export function setCuratedShortcutsEnabled(on: boolean): void {
  setCuratedShortcutsState(on);
  reconfigureAllMathfields();
}

export function syncCuratedShortcutsCheckbox(checkbox: HTMLInputElement | null | undefined): void {
  if (!checkbox) return;
  checkbox.checked = isCuratedShortcutsEnabled();
}

export function wireMathInputConfigDom(checkbox: HTMLInputElement | null | undefined): void {
  if (!checkbox || import.meta.env.PROD) return;
  syncCuratedShortcutsCheckbox(checkbox);
  checkbox.addEventListener("change", () => {
    setCuratedShortcutsEnabled(checkbox.checked);
  });
}

export function initMathInputConfig(): void {
  resetCuratedShortcutsCache();
  isCuratedShortcutsEnabled();
}
