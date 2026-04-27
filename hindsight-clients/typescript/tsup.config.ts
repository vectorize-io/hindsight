import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm"],
  dts: true,
  outDir: "dist",
  clean: true,
  sourcemap: true,
  // Bundle all relative imports (src + generated) into the output
  bundle: true,
});
