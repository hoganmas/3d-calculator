/**
 * Playwright: MathLive keystroke → LaTeX expectations.
 * Requires dev server: npm run dev
 * Run: npm run test:integration:keystroke
 */
import { chromium } from "playwright";
import { BASE } from "./base-url.mjs";
import { KEYSTROKE_CASES } from "../ui/keystroke-cases.mjs";
import {
  runKeystrokeCase,
  setCuratedShortcuts,
  withCuratedQuery,
} from "../helpers/mathfield-keystroke.mjs";

function filterCases(cases, curated) {
  return cases.filter((c) => {
    if (curated && c.rawOnly) return false;
    if (!curated && c.curatedOnly) return false;
    return true;
  });
}

async function runSuite(page, curated) {
  const cases = filterCases(KEYSTROKE_CASES, curated);
  const failures = [];
  for (const testCase of cases) {
    const result = await runKeystrokeCase(page, testCase);
    if (!result.pass) failures.push({ ...result, curated });
  }
  return failures;
}

const browser = await chromium.launch({ headless: true });

try {
  for (const curated of [true, false]) {
    const page = await browser.newPage();
    await setCuratedShortcuts(page, curated);
    await page.goto(withCuratedQuery(BASE, curated), { waitUntil: "networkidle", timeout: 60000 });
    await page.waitForTimeout(3000);

    const failures = await runSuite(page, curated);
    await page.close();

    if (failures.length) {
      console.error(
        JSON.stringify(
          {
            ok: false,
            curated,
            failures,
          },
          null,
          2,
        ),
      );
      process.exit(1);
    }
    console.log(`keystroke suite passed (curated=${curated}, cases=${filterCases(KEYSTROKE_CASES, curated).length})`);
  }

  console.log(JSON.stringify({ ok: true }, null, 2));
} finally {
  await browser.close();
}
