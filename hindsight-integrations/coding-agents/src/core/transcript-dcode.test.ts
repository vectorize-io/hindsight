import { writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readDcodeTranscript } from "./transcript-dcode";

describe("readDcodeTranscript", () => {
  it("normalizes Dcode records and compacts tool messages", () => {
    const path = "/tmp/hindsight-dcode-transcript-test.jsonl";
    writeFileSync(
      path,
      [
        JSON.stringify({
          schema_version: 1,
          role: "user",
          content: "Fix the retry policy",
          timestamp: "2026-08-29T10:00:00Z",
        }),
        JSON.stringify({
          schema_version: 1,
          role: "assistant",
          content: [{ type: "text", text: "I will inspect the policy." }],
          timestamp: "2026-08-29T10:00:01Z",
        }),
        JSON.stringify({
          schema_version: 1,
          role: "tool",
          name: "read_file",
          content: [{ type: "text", text: "src/retry.ts" }],
        }),
        JSON.stringify({
          schema_version: 1,
          role: "assistant",
          content: "before <hindsight_memories>old context</hindsight_memories> after",
        }),
        JSON.stringify({ schema_version: 2, role: "user", content: "unknown" }),
        "not json",
      ].join("\n")
    );

    expect(readDcodeTranscript(path)).toEqual([
      { role: "user", content: "Fix the retry policy", timestamp: "2026-08-29T10:00:00Z" },
      {
        role: "assistant",
        content: "I will inspect the policy.",
        timestamp: "2026-08-29T10:00:01Z",
      },
      { role: "action", content: "read_file" },
      { role: "assistant", content: "before  after" },
    ]);
  });

  it("fails open for a missing transcript", () => {
    expect(readDcodeTranscript("/tmp/does-not-exist-dcode-transcript.jsonl")).toEqual([]);
  });
});
