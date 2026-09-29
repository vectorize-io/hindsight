import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { grokTranscriptPath, readGrokTranscript } from "./transcript-grok";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-grok-transcript-"));
  file = join(root, "chat_history.jsonl");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("readGrokTranscript", () => {
  it("keeps real user prompts, assistant text, and compact tool actions", () => {
    writeFileSync(
      file,
      [
        JSON.stringify({
          type: "user",
          content: [{ type: "text", text: "synthetic setup" }],
          synthetic_reason: "system_reminder",
        }),
        JSON.stringify({
          type: "user",
          content: [{ type: "text", text: "Fix the retry bug" }],
          prompt_index: 0,
        }),
        JSON.stringify({
          type: "assistant",
          content: "I will inspect it.",
          tool_calls: [{ name: "read_file", arguments: '{"path":"src/retry.ts"}' }],
        }),
      ].join("\n")
    );
    expect(readGrokTranscript(file)).toEqual([
      { role: "user", content: "Fix the retry bug" },
      { role: "assistant", content: "I will inspect it." },
      { role: "action", content: "read_file src/retry.ts" },
    ]);
  });

  it("resolves the normal URL-encoded session location", () => {
    const cwd = "/Users/me/project";
    const path = grokTranscriptPath(cwd, "session-1", root);
    expect(path).toBe(
      join(root, "sessions", "%2FUsers%2Fme%2Fproject", "session-1", "chat_history.jsonl")
    );
  });

  it("resolves Grok's long-path .cwd fallback", () => {
    const cwd = "/Users/me/a-very-long-project";
    const dir = join(root, "sessions", "project-hash");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, ".cwd"), cwd);
    expect(grokTranscriptPath(cwd, "session-1", root)).toBe(
      join(dir, "session-1", "chat_history.jsonl")
    );
  });
});
