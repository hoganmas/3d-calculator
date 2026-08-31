import "../helpers/setup-dom.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";
import {
  buildCuratedInlineShortcuts,
  CURATED_SHORTCUT_SUPPRESS,
  LAPLACI_MATH_MODE_SPACE,
} from "../../src/ui/expr-sidebar/mathfieldConfig.js";
import {
  isCuratedShortcutsEnabled,
  resetCuratedShortcutsCache,
  setCuratedShortcutsState,
} from "../../src/app/curatedShortcutsState.js";

export async function run() {
  return runSuite("ui / mathfield-config", [
    {
      name: "buildCuratedInlineShortcuts constrains xx and removes grad/del",
      fn: () => {
        const merged = buildCuratedInlineShortcuts({
          xx: "\\times",
          grad: "\\nabla",
          del: "\\partial",
          pi: "\\pi",
        });
        assert(typeof merged.xx === "object" && merged.xx !== null, "xx is contextual override");
        assert(
          (merged.xx as { after?: string }).after?.includes("digit"),
          "xx requires non-letter context",
        );
        assert(!(merged.xx as { after?: string }).after?.includes("letter"), "xx not after letter");
        assert(merged.pi === "\\pi", "pi preserved");
        assert(!("grad" in merged), "grad suppressed");
        assert(!("del" in merged), "del suppressed");
        for (const key of CURATED_SHORTCUT_SUPPRESS) {
          assert(!(key in merged), `${key} suppressed`);
        }
      },
    },
    {
      name: "math mode space is thin space",
      fn: () => {
        assert(LAPLACI_MATH_MODE_SPACE === "\\,", "thin space command");
      },
    },
    {
      name: "curated shortcuts toggle persists in localStorage",
      fn: () => {
        resetCuratedShortcutsCache();
        setCuratedShortcutsState(false);
        assert(isCuratedShortcutsEnabled() === false, "disabled");
        assert(localStorage.getItem("laplacian-curated-shortcuts") === "0", "stored off");
        setCuratedShortcutsState(true);
        assert(isCuratedShortcutsEnabled() === true, "enabled");
        assert(localStorage.getItem("laplacian-curated-shortcuts") === "1", "stored on");
      },
    },
  ]);
}
