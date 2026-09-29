import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readCursorTranscript } from "./transcript-cursor";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-cursor-transcript-"));
  file = join(root, "session.jsonl");
});

afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("readCursorTranscript", () => {
  it("keeps messages, compacts tool calls, and strips injected memory", () => {
    writeFileSync(
      file,
      [
        JSON.stringify({
          type: "user",
          timestamp: "2026-07-30T10:00:00Z",
          message: { content: [{ type: "text", text: "Fix the parser" }] },
        }),
        JSON.stringify({
          type: "assistant",
          timestamp: "2026-07-30T10:00:01Z",
          message: {
            role: "assistant",
            content: [
              {
                type: "text",
                text: "<hindsight_memories>old</hindsight_memories> I found the bug",
              },
              { type: "tool_use", name: "read", input: { path: "src/parser.ts" } },
            ],
          },
        }),
        JSON.stringify({
          type: "tool_call",
          timestamp: "2026-07-30T10:00:02Z",
          name: "write",
          args: { path: "src/parser.ts" },
        }),
        "not json",
      ].join("\n")
    );

    expect(readCursorTranscript(file)).toEqual([
      { role: "user", content: "Fix the parser", timestamp: "2026-07-30T10:00:00Z" },
      { role: "assistant", content: "I found the bug", timestamp: "2026-07-30T10:00:01Z" },
      { role: "action", content: "read src/parser.ts", timestamp: "2026-07-30T10:00:01Z" },
      { role: "action", content: "write src/parser.ts", timestamp: "2026-07-30T10:00:02Z" },
    ]);
  });

  it("returns no turns for missing files or lifecycle-only events", () => {
    expect(readCursorTranscript(join(root, "missing.jsonl"))).toEqual([]);
    writeFileSync(file, JSON.stringify({ type: "status", status: "idle" }));
    expect(readCursorTranscript(file)).toEqual([]);
  });
});
