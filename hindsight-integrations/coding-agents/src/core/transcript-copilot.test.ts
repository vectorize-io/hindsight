import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readCopilotTranscript } from "./transcript-copilot";

let root = "";
afterEach(() => root && rmSync(root, { recursive: true, force: true }));

describe("readCopilotTranscript", () => {
  it("reads Copilot CLI user and assistant events and strips injected memory", () => {
    root = mkdtempSync(join(tmpdir(), "hs-copilot-transcript-"));
    const path = join(root, "events.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({ type: "session.start", data: {} }),
        JSON.stringify({
          type: "user.message",
          timestamp: "2026-01-01T00:00:00Z",
          data: { content: "ship it" },
        }),
        JSON.stringify({
          type: "assistant.message",
          data: { content: "done\n<hindsight_memory>old</hindsight_memory>" },
        }),
      ].join("\n")
    );
    expect(readCopilotTranscript(path)).toEqual([
      { role: "user", content: "ship it", timestamp: "2026-01-01T00:00:00Z" },
      { role: "assistant", content: "done" },
    ]);
  });
});
