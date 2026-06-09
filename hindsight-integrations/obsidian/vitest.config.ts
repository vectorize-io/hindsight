import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    include: ["tests/**/*.spec.ts"],
    environment: "node",
    globals: false,
    alias: {
      // The real `obsidian` package has no runtime outside the app — swap in our mock.
      obsidian: path.resolve(__dirname, "tests/__mocks__/obsidian.ts"),
    },
  },
});
