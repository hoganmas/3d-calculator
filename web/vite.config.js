import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import fs from "node:fs";
import path from "node:path";

/** When a `.js` import has no file on disk, resolve the sibling `.ts` (TS migration). */
function resolveTsFromJs() {
  return {
    name: "resolve-ts-from-js",
    resolveId(source, importer) {
      if (!importer || !source.endsWith(".js") || source.includes("node_modules")) return null;
      const base = path.resolve(path.dirname(importer), source);
      if (fs.existsSync(base)) return null;
      const ts = base.slice(0, -3) + ".ts";
      if (fs.existsSync(ts)) return ts;
      return null;
    },
  };
}

export default defineConfig({
  base: "/",
  plugins: [svelte(), resolveTsFromJs()],
  optimizeDeps: {
    // Pre-bundle large ESM deps added after first dev start (avoids 504 Outdated Optimize Dep).
    include: ["mathlive", "@cortex-js/compute-engine", "three"],
  },
});
