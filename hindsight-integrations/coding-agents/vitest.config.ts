import { defineConfig } from "vitest/config";
import { tmpdir } from "node:os";
import { join } from "node:path";

export default defineConfig({
  test: {
    env: {
      // Unit tests exercise code paths that emit diagnostics; keep them out of the REAL
      // /tmp/hindsight-plugin.log so a developer's diag trail isn't polluted with test noise.
      HINDSIGHT_DIAG_FILE: join(tmpdir(), "hindsight-plugin-test.log"),
      HINDSIGHT_LOG_FILE: join(tmpdir(), "hindsight-plugin-test-leveled.log"),
      // Same reasoning for the CONFIG file: `loadConfig` otherwise resolves the developer's real
      // ~/.hindsight/coding-agent.json, so a machine that has a token (or a bank override, or a
      // different apiUrl) fails assertions that a clean machine passes. It must be set HERE rather
      // than with vi.stubEnv in a test: config.ts captures the path in a module-level constant, so
      // by the time a test body runs the value is already fixed. Points at a path with no file.
      HINDSIGHT_CONFIG: join(tmpdir(), "hindsight-coding-agent-test-absent.json"),
    },
  },
});
