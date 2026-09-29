import { describe, expect, it } from "vitest";
import { readOpencodeMessages, opencodeSessionId, type OcMessage } from "./transcript-opencode";

describe("readOpencodeMessages", () => {
  it("keeps user/assistant text + compact action turns; drops other roles/parts and tool outputs", () => {
    const messages: OcMessage[] = [
      // non-conversational role: dropped
      { info: { role: "system" }, parts: [{ type: "text", text: "you are opencode" }] },
      // real user prompt: kept
      {
        info: { role: "user", sessionID: "ses_1", time: { created: 1_700_000_000_000 } },
        parts: [{ type: "text", text: "add retry backoff to the uploader" }],
      },
      // assistant message: text + a completed tool call (output NOT retained) + a dropped reasoning part
      {
        info: { role: "assistant" },
        parts: [
          { type: "reasoning", text: "thinking…" },
          { type: "text", text: "I'll add exponential backoff." },
          {
            type: "tool",
            tool: "bash",
            state: { status: "completed", input: { command: "npm test" }, output: "12 passed" },
          },
        ],
      },
      // assistant message with an errored tool call: still just the compact action line (no error text)
      {
        info: { role: "assistant" },
        parts: [
          {
            type: "tool",
            tool: "read",
            state: { status: "error", input: { path: "nope.ts" }, error: "ENOENT" },
          },
        ],
      },
    ];

    expect(readOpencodeMessages(messages)).toEqual([
      {
        role: "user",
        content: "add retry backoff to the uploader",
        timestamp: new Date(1_700_000_000_000).toISOString(),
      },
      { role: "assistant", content: "I'll add exponential backoff." },
      { role: "action", content: "bash npm test" },
      { role: "action", content: "read nope.ts" },
    ]);
  });

  it("strips injected memory that leaks into a kept message", () => {
    const messages: OcMessage[] = [
      {
        info: { role: "user" },
        parts: [
          {
            type: "text",
            text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?",
          },
        ],
      },
    ];
    expect(readOpencodeMessages(messages)).toEqual([{ role: "user", content: "Why retry?" }]);
  });

  it("never retains tool output — even a huge one yields only the compact action line", () => {
    const messages: OcMessage[] = [
      {
        info: { role: "assistant" },
        parts: [
          {
            type: "tool",
            tool: "bash",
            state: {
              status: "completed",
              input: { command: "cat big.log" },
              output: "x".repeat(5000),
            },
          },
        ],
      },
    ];
    expect(readOpencodeMessages(messages)).toEqual([
      { role: "action", content: "bash cat big.log" },
    ]);
  });

  it("action turns inherit the message timestamp when present", () => {
    const messages: OcMessage[] = [
      {
        info: { role: "assistant", time: { created: 1_700_000_000_000 } },
        parts: [{ type: "tool", tool: "edit", state: { input: { file_path: "uploader.ts" } } }],
      },
    ];
    expect(readOpencodeMessages(messages)).toEqual([
      {
        role: "action",
        content: "edit uploader.ts",
        timestamp: new Date(1_700_000_000_000).toISOString(),
      },
    ]);
  });

  it("drops a message with no usable parts, and tolerates a missing parts array", () => {
    const messages: OcMessage[] = [
      { info: { role: "assistant" }, parts: [{ type: "step-start" }] }, // no text/tool: dropped
      { info: { role: "user" } }, // no parts: tolerated, dropped
    ];
    expect(readOpencodeMessages(messages)).toEqual([]);
  });

  it("opencodeSessionId returns the first message's session id, or undefined", () => {
    expect(
      opencodeSessionId([
        { info: { role: "assistant" } },
        { info: { role: "user", sessionID: "ses_9" } },
      ])
    ).toBe("ses_9");
    expect(opencodeSessionId([{ info: { role: "user" } }])).toBeUndefined();
    expect(opencodeSessionId([])).toBeUndefined();
  });
});
