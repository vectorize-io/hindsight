import { describe, expect, it } from "vitest";
import { readOpencode2Messages, type Oc2Message } from "./transcript-opencode2";

describe("readOpencode2Messages", () => {
  it("keeps user/assistant prose + compact action turns; drops other types, parts and tool outputs", () => {
    // Shapes taken from a live `ctx.session.context({sessionID})` response (opencode2
    // 0.0.0-beta-18743): a user message carries `text`, an assistant message a `content` array.
    const messages: Oc2Message[] = [
      // non-conversational message type: dropped
      { type: "agent-selected", text: "build" },
      {
        type: "user",
        text: "add retry backoff to the uploader",
        time: { created: 1_700_000_000_000 },
      },
      {
        type: "assistant",
        content: [
          { type: "reasoning", text: "thinking…" },
          { type: "text", text: "I'll add exponential backoff." },
          {
            type: "tool",
            name: "shell",
            state: {
              status: "completed",
              input: { command: "npm test" },
              content: [{ type: "text", text: "12 passed" }],
            },
          },
        ],
      },
      // an errored tool call still renders only the compact action line (no error text)
      {
        type: "assistant",
        content: [
          { type: "tool", name: "read", state: { status: "error", input: { path: "nope.ts" } } },
        ],
      },
    ];

    expect(readOpencode2Messages(messages)).toEqual([
      {
        role: "user",
        content: "add retry backoff to the uploader",
        timestamp: new Date(1_700_000_000_000).toISOString(),
      },
      { role: "assistant", content: "I'll add exponential backoff." },
      { role: "action", content: "shell npm test" },
      { role: "action", content: "read nope.ts" },
    ]);
  });

  it("strips injected memory so a write-back never re-ingests what we injected", () => {
    const messages: Oc2Message[] = [
      { type: "user", text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?" },
      // a message that is NOTHING but injected memory renders empty and is dropped entirely
      { type: "user", text: "<hindsight_knowledge>\nall of it\n</hindsight_knowledge>" },
    ];
    expect(readOpencode2Messages(messages)).toEqual([{ role: "user", content: "Why retry?" }]);
  });

  it("joins an assistant's several text parts into one prose turn, actions after", () => {
    const messages: Oc2Message[] = [
      {
        type: "assistant",
        content: [
          { type: "text", text: "first" },
          { type: "tool", name: "glob", state: { input: { pattern: "*.ts" } } },
          { type: "text", text: "second" },
        ],
      },
    ];
    expect(readOpencode2Messages(messages)).toEqual([
      { role: "assistant", content: "first\nsecond" },
      { role: "action", content: "glob *.ts" },
    ]);
  });

  it("never throws on malformed or empty input", () => {
    expect(readOpencode2Messages([])).toEqual([]);
    expect(readOpencode2Messages(undefined as unknown as Oc2Message[])).toEqual([]);
    expect(
      readOpencode2Messages([
        {},
        { type: "assistant" },
        { type: "assistant", content: [null as unknown as never, { type: "tool" }] },
      ])
    ).toEqual([]);
  });
});
