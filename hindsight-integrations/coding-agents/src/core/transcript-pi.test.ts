import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { type PiMessage, readPiMessages, readPiTranscript } from "./transcript-pi";

describe("readPiMessages", () => {
  it("keeps user/assistant text + compact action turns; drops other roles/blocks and tool args", () => {
    const messages: PiMessage[] = [
      // non-conversational role: dropped
      { role: "system", content: [{ type: "text", text: "you are pi" }] },
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

    expect(readPiMessages(messages)).toEqual([
      { role: "user", content: "add retry backoff to the uploader" },
      { role: "assistant", content: "I'll add exponential backoff." },
      { role: "action", content: "bash npm test" },
      { role: "action", content: "read nope.ts" },
    ]);
  });

  it("strips injected memory that leaks into a kept message", () => {
    const messages: PiMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?" },
        ],
      },
    ];
    expect(readPiMessages(messages)).toEqual([{ role: "user", content: "Why retry?" }]);
  });

  it("never throws on malformed entries", () => {
    const messages = [
      null,
      {},
      { role: "user" },
      { role: "assistant", content: [null, 3, "x"] },
    ] as unknown as PiMessage[];
    expect(() => readPiMessages(messages)).not.toThrow();
    expect(readPiMessages(messages)).toEqual([]);
  });
});

describe("readPiTranscript", () => {
  let dir: string | undefined;
  afterEach(() => {
    if (dir) rmSync(dir, { recursive: true, force: true });
    dir = undefined;
  });

  function writeLog(...lines: unknown[]): string {
    dir = mkdtempSync(join(tmpdir(), "hs-pi-log-"));
    const file = join(dir, "session.jsonl");
    writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
    return file;
  }

  it("renders a stored session exactly like the live agent_end path, stamped with entry timestamps", () => {
    const file = writeLog(
      { type: "session", version: 3, id: "s1", cwd: "/repo" },
      // Settings entries sit between messages and carry no conversation.
      { type: "model_change", provider: "openai-codex", modelId: "gpt-5.6-sol" },
      {
        type: "message",
        timestamp: "2026-08-24T12:00:01.000Z",
        message: { role: "user", content: [{ type: "text", text: "add retry backoff" }] },
      },
      {
        type: "message",
        timestamp: "2026-08-24T12:00:02.000Z",
        message: {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "internal" },
            { type: "text", text: "Adding exponential backoff." },
            { type: "toolCall", name: "read", arguments: { path: "uploader.ts" } },
          ],
        },
      },
      // Tool OUTPUT is mechanical noise, dropped like the Codex reader drops function_call_output.
      {
        type: "message",
        timestamp: "2026-08-24T12:00:03.000Z",
        message: { role: "toolResult", content: [{ type: "text", text: "file contents" }] },
      }
    );

    expect(readPiTranscript(file)).toEqual([
      { role: "user", content: "add retry backoff", timestamp: "2026-08-24T12:00:01.000Z" },
      {
        role: "assistant",
        content: "Adding exponential backoff.",
        timestamp: "2026-08-24T12:00:02.000Z",
      },
      { role: "action", content: "read uploader.ts", timestamp: "2026-08-24T12:00:02.000Z" },
    ]);
  });

  // An imported session must not feed our own injected memory back into the bank — the same
  // stripInjectedMemory guarantee the live path has, exercised through the file reader.
  it("strips injected memory from a stored user turn", () => {
    const file = writeLog(
      { type: "session", version: 3, id: "s1", cwd: "/repo" },
      {
        type: "message",
        timestamp: "2026-08-24T12:00:01.000Z",
        message: {
          role: "user",
          content: [
            {
              type: "text",
              text: "<hindsight_memories>only 429 and 408 retry</hindsight_memories>\nship it",
            },
          ],
        },
      }
    );

    const turns = readPiTranscript(file);
    expect(turns).toEqual([
      { role: "user", content: "ship it", timestamp: "2026-08-24T12:00:01.000Z" },
    ]);
  });

  it("survives a torn tail line and a missing file instead of throwing", () => {
    dir = mkdtempSync(join(tmpdir(), "hs-pi-log-"));
    const file = join(dir, "session.jsonl");
    writeFileSync(
      file,
      `${JSON.stringify({ type: "session", id: "s1", cwd: "/repo" })}\n` +
        `${JSON.stringify({ type: "message", message: { role: "user", content: "kept" } })}\n` +
        `{"type":"message","message":{"role":"assis`
    );
    expect(readPiTranscript(file)).toEqual([{ role: "user", content: "kept" }]);
    expect(readPiTranscript(join(dir, "gone.jsonl"))).toEqual([]);
  });
});
