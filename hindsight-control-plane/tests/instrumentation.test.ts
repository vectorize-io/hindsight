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
      const receipt = JSON.parse(readFileSync(pidFile, "utf8"));
      expect(receipt).toMatchObject({ version: 1, pid: process.pid });
      expect(receipt.birth_marker).toBeTypeOf("string");
      expect(receipt.birth_marker.length).toBeGreaterThan(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
