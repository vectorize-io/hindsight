import { describe, expect, it } from "vitest";
import { type PaMessage, readPrimeAgentMessages } from "./transcript-prime-agent";

describe("readPrimeAgentMessages", () => {
  it("keeps user/assistant text + compact action turns; drops other roles/blocks and tool args", () => {
    const messages: PaMessage[] = [
      // non-conversational role: dropped
      { role: "system", content: [{ type: "text", text: "you are prime-agent" }] },
      // string content is accepted as a prose turn
      { role: "user", content: "add retry backoff to the uploader" },
      // assistant message: text + a toolCall (args NOT retained) + a dropped reasoning block
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking…" },
          { type: "text", text: "I'll add exponential backoff." },
          { type: "toolCall", name: "bash", arguments: { command: "npm test" } },
        ],
      },
      // assistant message with only a toolCall: just the compact action line
      {
        role: "assistant",
        content: [{ type: "toolCall", name: "read", arguments: { path: "nope.ts" } }],
      },
    ];

    expect(readPrimeAgentMessages(messages)).toEqual([
      { role: "user", content: "add retry backoff to the uploader" },
      { role: "assistant", content: "I'll add exponential backoff." },
      { role: "action", content: "bash npm test" },
      { role: "action", content: "read nope.ts" },
    ]);
  });

  it("strips injected memory that leaks into a kept message", () => {
    const messages: PaMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?" },
        ],
      },
    ];
    expect(readPrimeAgentMessages(messages)).toEqual([{ role: "user", content: "Why retry?" }]);
  });

  it("never throws on malformed entries", () => {
    const messages = [
      null,
      {},
      { role: "user" },
      { role: "assistant", content: [null, 3, "x"] },
    ] as unknown as PaMessage[];
    expect(() => readPrimeAgentMessages(messages)).not.toThrow();
    expect(readPrimeAgentMessages(messages)).toEqual([]);
  });
});
