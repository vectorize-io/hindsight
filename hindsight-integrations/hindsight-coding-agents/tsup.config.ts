import { defineConfig } from "tsup";

export default defineConfig({
  entry: { index: "src/index.ts", backfill: "src/backfill.ts" },
  format: ["esm"],
  target: "node18",
  clean: true,
  dts: { entry: "src/index.ts" },
  shims: false,
});
