import { writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { dcodeAssistantText, readDcodeTranscript } from "./transcript-dcode";

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

describe("dcodeAssistantText", () => {
  // Captured verbatim from `dcode -n … -M gpt-5.5` (deepagents-code 0.1.65): a reasoning model's
  // Stop event, with the 3KB encrypted_content shortened. The reply is the single word BANANA.
  const REAL_REASONING_STOP_VALUE =
    "[{'id': 'rs_085f844b6351dc51006a95a4cdd83487d2833dd4da6bc0174d', 'summary': [], " +
    "'type': 'reasoning', 'content': [], 'index': 0, " +
    "'encrypted_content': 'gAAAAABqlaTOleQ05Dl0ENKkT1XOj3QR3qG0j_WYFuINfdnUV9YeaEuEEwvoPle7=='}, " +
    "{'type': 'text', 'text': 'BANANA', 'phase': 'final_answer', 'index': 1, " +
    "'id': 'msg_085f844b6351dc51006a95a4ce0fa887d2bf2a96d466965564'}]";

  it("recovers the reply text from a reasoning model's serialized content blocks", () => {
    expect(dcodeAssistantText(REAL_REASONING_STOP_VALUE)).toBe("BANANA");
  });

  it("never leaks the provider's encrypted reasoning payload into the retained text", () => {
    expect(dcodeAssistantText(REAL_REASONING_STOP_VALUE)).not.toContain("encrypted_content");
    expect(dcodeAssistantText(REAL_REASONING_STOP_VALUE)).not.toContain("gAAAAA");
  });

  it("passes plain string content through untouched", () => {
    expect(dcodeAssistantText("Done — I updated the retry policy.")).toBe(
      "Done — I updated the retry policy."
    );
    // Prose that merely starts with a bracket is not a block list and must survive verbatim.
    expect(dcodeAssistantText("[note] see src/retry.ts")).toBe("[note] see src/retry.ts");
  });

  it("joins multiple text blocks exactly as the transcript reader does", () => {
    expect(
      dcodeAssistantText("[{'type': 'text', 'text': 'one'}, {'type': 'text', 'text': 'two'}]")
    ).toBe("one\ntwo");
  });

  it("decodes the escapes repr() emits", () => {
    expect(dcodeAssistantText(`[{'type': 'text', 'text': 'line\\none\\ttab "q" \\'s\\''}]`)).toBe(
      "line\none\ttab \"q\" 's'"
    );
  });

  it("handles the value forms repr() produces for LangChain content", () => {
    expect(
      dcodeAssistantText(
        "[{'type': 'text', 'text': 'ok', 'cache': None, 'partial': False, 'n': 1, 'r': -0.5, " +
          "'meta': {'nested': ['a', True]}}]"
      )
    ).toBe("ok");
  });

  it("drops an unparseable block list rather than retaining the raw blob", () => {
    // A dropped turn is recovered by the next Stop, which rebuilds the document from the whole
    // transcript; a stored blob would be permanent.
    expect(dcodeAssistantText("[{'type': 'text', 'text': 'unterminated}]")).toBe("");
    expect(dcodeAssistantText("[{'type': <object at 0x10>}]")).toBe("");
  });

  it("drops a block list carrying no text blocks", () => {
    expect(dcodeAssistantText("[{'type': 'reasoning', 'encrypted_content': 'abc'}]")).toBe("");
  });
});
