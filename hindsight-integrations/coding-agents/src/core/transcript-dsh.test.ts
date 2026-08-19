import { describe, expect, it } from "vitest";
import { readDshEvents, type DshSessionEvent } from "./transcript-dsh";

const at = (iso: string) => Date.parse(iso);

const userMessage = (text: string, source: Record<string, unknown> = { kind: "user" }) => ({
  id: "m-1",
  role: "user",
  content: [{ type: "text", text }],
  source,
});

describe("readDshEvents", () => {
  it("renders human prompts, assistant replies and tool calls in log order", () => {
    const events: DshSessionEvent[] = [
      { type: "turn/start", time: at("2026-08-14T10:00:00Z"), data: { turn: 1 } },
      {
        type: "user/message",
        time: at("2026-08-14T10:00:01Z"),
        data: userMessage("fix the flake"),
      },
      {
        type: "tool/call",
        time: at("2026-08-14T10:00:02Z"),
        data: { turn: 1, step: 1, callId: "c1", name: "read", arguments: '{"path":"src/app.ts"}' },
      },
      {
        type: "assistant/message",
        time: at("2026-08-14T10:00:03Z"),
        data: {
          turn: 1,
          step: 1,
          message: {
            role: "assistant",
            content: [
              { type: "reasoning", text: "thinking out loud" },
              { type: "text", text: "The retry loop was the problem." },
            ],
            source: { kind: "model" },
          },
        },
      },
    ];

    expect(readDshEvents(events)).toEqual([
      { role: "user", content: "fix the flake", timestamp: "2026-08-14T10:00:01.000Z" },
      { role: "action", content: "read src/app.ts", timestamp: "2026-08-14T10:00:02.000Z" },
      {
        role: "assistant",
        content: "The retry loop was the problem.",
        timestamp: "2026-08-14T10:00:03.000Z",
      },
    ]);
  });

  it("drops every plugin-sourced message, ours and the host's alike", () => {
    const events: DshSessionEvent[] = [
      {
        type: "user/message",
        time: at("2026-08-14T10:00:00Z"),
        data: userMessage("<hindsight_memory>recalled</hindsight_memory>", {
          kind: "plugin",
          plugin: "hindsight",
          form: "recall",
        }),
      },
      {
        type: "user/message",
        time: at("2026-08-14T10:00:01Z"),
        data: userMessage("Current runtime context. This snapshot supersedes…", {
          kind: "plugin",
          plugin: "@deepseek-ai/dsh-system-prompt",
        }),
      },
      { type: "user/message", time: at("2026-08-14T10:00:02Z"), data: userMessage("carry on") },
    ];

    expect(readDshEvents(events)).toEqual([
      { role: "user", content: "carry on", timestamp: "2026-08-14T10:00:02.000Z" },
    ]);
  });

  it("keeps a tool call whose arguments the model produced malformed", () => {
    const events: DshSessionEvent[] = [
      {
        type: "tool/call",
        time: at("2026-08-14T10:00:00Z"),
        data: { name: "bash", arguments: "{oops" },
      },
    ];
    expect(readDshEvents(events)).toEqual([
      { role: "action", content: "bash {oops", timestamp: "2026-08-14T10:00:00.000Z" },
    ]);
  });

  it("ignores log-only events, unknown types and malformed entries", () => {
    const events = [
      null,
      "not an event",
      { type: "assistant/chunk", time: at("2026-08-14T10:00:00Z"), data: { chunk: {} } },
      { type: "tool/result", time: at("2026-08-14T10:00:01Z"), data: { message: { content: [] } } },
      { type: "user/message", data: undefined },
      { type: "assistant/message", time: at("2026-08-14T10:00:02Z"), data: {} },
    ] as unknown as DshSessionEvent[];

    expect(readDshEvents(events)).toEqual([]);
  });

  it("survives an empty or absent log", () => {
    expect(readDshEvents([])).toEqual([]);
    expect(readDshEvents(undefined as unknown as DshSessionEvent[])).toEqual([]);
  });
});
