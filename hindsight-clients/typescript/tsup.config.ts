import { defineConfig } from "tsup";
import pkg from "./package.json";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm"],
  dts: true,
  outDir: "dist",
  clean: true,
  sourcemap: true,
  // Bundle all relative imports (src + generated) into the output
  bundle: true,
  // Substitute __CLIENT_VERSION__ with package.json's version at build time so
  // the constant in src/index.ts can never drift from what npm publishes.
  define: {
    __CLIENT_VERSION__: JSON.stringify(pkg.version),
  },
});
