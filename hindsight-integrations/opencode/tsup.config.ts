import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/index.v2.ts"],
  format: ["esm"],
  dts: true,
  outDir: "dist",
  clean: true,
  sourcemap: true,
  bundle: true,
});
