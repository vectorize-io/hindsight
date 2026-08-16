import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { register } from "../src/instrumentation";

const originalPidFile = process.env.HINDSIGHT_EMBED_UI_PID_FILE;

afterEach(() => {
  if (originalPidFile === undefined) {
    delete process.env.HINDSIGHT_EMBED_UI_PID_FILE;
  } else {
    process.env.HINDSIGHT_EMBED_UI_PID_FILE = originalPidFile;
  }
});

describe("control-plane instrumentation", () => {
  it("records the PID of the actual UI server process", async () => {
    const dir = mkdtempSync(join(tmpdir(), "hindsight-ui-pid-"));
    const pidFile = join(dir, "nested", "ui.pid");
    process.env.HINDSIGHT_EMBED_UI_PID_FILE = pidFile;

    try {
      await register();
      expect(readFileSync(pidFile, "utf8")).toBe(String(process.pid));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
