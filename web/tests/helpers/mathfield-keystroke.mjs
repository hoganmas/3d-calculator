/**
 * Playwright helpers for MathLive keystroke integration tests.
 */

const STORAGE_KEY = "laplacian-curated-shortcuts";

export async function setCuratedShortcuts(page, on) {
  await page.addInitScript(
    ([key, value]) => {
      if (value == null) localStorage.removeItem(key);
      else localStorage.setItem(key, value ? "1" : "0");
    },
    [STORAGE_KEY, on ? "1" : "0"],
  );
}

export function withCuratedQuery(base, on) {
  const url = new URL(base);
  url.searchParams.set("curatedShortcuts", on ? "1" : "0");
  return url.toString();
}

export async function focusEmptyExprField(page) {
  const field = page.locator(".expr-row:not(.is-param-def) math-field").first();
  await field.waitFor({ state: "visible", timeout: 30000 });
  await field.click();
  await page.evaluate(() => {
    const mf = document.querySelector(".expr-row:not(.is-param-def) math-field");
    if (!mf) throw new Error("no math-field");
    if (typeof mf.setValue === "function") {
      mf.setValue("", { silenceNotifications: true });
    } else {
      mf.value = "";
    }
    if (typeof mf.executeCommand === "function") {
      try {
        mf.executeCommand("selectAll");
        mf.executeCommand("deleteBackward");
      } catch {
        /* ignore */
      }
    }
    mf.focus?.();
  });
  await page.waitForTimeout(50);
  return field;
}

export async function typeKeys(page, keys, delay = 30) {
  await page.keyboard.type(keys, { delay });
}

export async function readFieldLatex(page) {
  return page.evaluate(() => {
    const mf = document.querySelector(".expr-row:not(.is-param-def) math-field");
    if (!mf) return "";
    if (typeof mf.getValue === "function") return String(mf.getValue("latex") || "");
    return String(mf.value || "");
  });
}

export function latexMatches(actual, expected) {
  if (expected instanceof RegExp) return expected.test(actual);
  return actual === expected;
}

export async function runKeystrokeCase(page, testCase) {
  await focusEmptyExprField(page);
  await typeKeys(page, testCase.keys, testCase.delay ?? 30);
  await page.waitForTimeout(80);
  const actual = await readFieldLatex(page);
  const pass = latexMatches(actual, testCase.expectLatex);
  return {
    pass,
    actual,
    expected: String(testCase.expectLatex),
    name: testCase.name,
  };
}
