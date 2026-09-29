import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readAntigravityTranscript } from "./transcript-antigravity";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-antigravity-transcript-"));
  file = join(root, "transcript.jsonl");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("readAntigravityTranscript", () => {
  it("reads supported user and assistant message shapes while ignoring metadata", () => {
    writeFileSync(
      file,
      [
        JSON.stringify({ type: "system", content: "metadata" }),
        JSON.stringify({
          role: "user",
          content: "Fix the retry bug",
          timestamp: "2026-01-01T00:00:00Z",
        }),
        JSON.stringify({ message: { role: "assistant", text: "I will inspect it." } }),
      ].join("\n")
    );
    expect(readAntigravityTranscript(file)).toEqual([
      { role: "user", content: "Fix the retry bug", timestamp: "2026-01-01T00:00:00Z" },
      { role: "assistant", content: "I will inspect it." },
    ]);
  });

  it("reads Antigravity CLI's native USER_INPUT records", () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "USER_INPUT",
        content: "<USER_REQUEST>\nCreate a bank for this repo.\n</USER_REQUEST>",
      })
    );
    expect(readAntigravityTranscript(file)).toEqual([
      { role: "user", content: "<USER_REQUEST>\nCreate a bank for this repo.\n</USER_REQUEST>" },
    ]);
  });

  it("strips injected memory and fails open for an unavailable transcript", () => {
    writeFileSync(
      file,
      JSON.stringify({
        role: "user",
        text: "<hindsight_memory>old</hindsight_memory>\nWhat changed?",
      })
    );
    expect(readAntigravityTranscript(file)).toEqual([{ role: "user", content: "What changed?" }]);
    expect(readAntigravityTranscript(join(root, "missing.jsonl"))).toEqual([]);
  });
});
