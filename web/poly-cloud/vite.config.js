import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    // Pre-bundle large ESM deps added after first dev start (avoids 504 Outdated Optimize Dep).
    include: ["mathlive", "@cortex-js/compute-engine", "three"],
  },
});
