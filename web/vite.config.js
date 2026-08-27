import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  base: "/3d-calculator/",
  plugins: [svelte()],
  optimizeDeps: {
    // Pre-bundle large ESM deps added after first dev start (avoids 504 Outdated Optimize Dep).
    include: ["mathlive", "@cortex-js/compute-engine", "three"],
  },
});
