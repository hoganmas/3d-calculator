/**
 * Run all Playwright integration tests (requires `npm run dev`).
 */
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const root = dirname(fileURLToPath(import.meta.url));
const files = readdirSync(root)
  .filter((f) => f.endsWith(".test.mjs") && f !== "run.mjs")
  .sort();

if (!files.length) {
  console.error("No integration tests found");
  process.exit(1);
}

let failed = 0;
for (const file of files) {
  console.log(`\n--- integration / ${file} ---`);
  const r = spawnSync(process.execPath, [join(root, file)], {
    stdio: "inherit",
    env: process.env,
  });
  if (r.status !== 0) failed++;
}

if (failed) {
  console.error(`\n${failed} integration test file(s) failed`);
  process.exit(1);
}
console.log("\nAll integration tests passed");
