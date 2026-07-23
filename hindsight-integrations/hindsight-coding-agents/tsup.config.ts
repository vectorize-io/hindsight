import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    backfill: "src/backfill.ts",
    "claude-hook": "src/claude-hook.ts",
    "cursor-hook": "src/cursor-hook.ts",
    "codex-hook": "src/codex-hook.ts",
  },
  format: ["esm"],
  target: "node18",
  clean: true,
  dts: { entry: "src/index.ts" },
  shims: false,
});
