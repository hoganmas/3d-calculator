/**
 * Discover and run all *.test.ts suites under tests/.
 * Usage: npm test [filter]
 *   npm test           — all suites
 *   npm test flow      — only paths containing "flow"
 */
import { readdirSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const filter = process.argv.slice(2).join("/").toLowerCase();

function collect(dir, out = []) {
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, ent.name);
    if (ent.isDirectory() && ent.name !== "helpers") collect(p, out);
    else if (ent.isFile() && ent.name.endsWith(".test.ts")) out.push(p);
  }
  return out;
}

const files = collect(root)
  .filter((f) => !filter || relative(root, f).toLowerCase().includes(filter))
  .sort();

if (!files.length) {
  console.error(filter ? `No tests match "${filter}"` : "No test files found");
  process.exit(1);
}

let totalFailed = 0;
for (const file of files) {
  const mod = await import(pathToFileURL(file).href);
  if (typeof mod.run !== "function") {
    console.error(`SKIP ${relative(root, file)}: missing run() export`);
    totalFailed++;
    continue;
  }
  totalFailed += await mod.run();
}

if (totalFailed) {
  console.error(`\n${totalFailed} failure(s)`);
  process.exit(1);
}
console.log("\nAll tests passed");
