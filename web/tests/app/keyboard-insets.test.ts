import {
  keyboardBottomInsetFrom,
  keyboardTopInsetFrom,
} from "../../src/app/keyboardInsets.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

export async function run() {
  return runSuite("app / keyboard-insets", [
    {
      name: "keyboard bottom inset from visualViewport geometry",
      fn: () => {
        assert(keyboardBottomInsetFrom(852, 500, 0) === 352, "keyboard covers bottom");
        assert(keyboardBottomInsetFrom(852, 852, 0) === 0, "no keyboard");
        assert(keyboardBottomInsetFrom(852, 500, 100) === 252, "accounts for offsetTop");
        assert(keyboardTopInsetFrom(120) === 120, "top inset");
        assert(keyboardTopInsetFrom(0) === 0, "no top inset");
      },
    },
  ]);
}
