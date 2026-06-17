import { defineConfig } from "vite";
import preact from "@preact/preset-vite";
import tailwindcss from "@tailwindcss/vite";

// Builds the control-center SPA to ../static (committed, served by the Python
// stdlib http.server). `base: "./"` keeps asset URLs relative so they resolve
// when served from "/". Everything is bundled (no CDN) so it works offline.
export default defineConfig({
  plugins: [preact(), tailwindcss()],
  base: "./",
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
});
