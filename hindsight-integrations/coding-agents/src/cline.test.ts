import { describe, expect, it, vi } from "vitest";
import { CLINE_HOOK_BUDGET_MS, clineTranscript, createClineHooks } from "./cline";

const message = (id: string, role: "user" | "assistant" | "tool", text: string) => ({
  id,
  role,
  content: [{ type: "text", text }],
  createdAt: Date.parse("2026-07-31T10:00:00Z"),
});

describe("Cline native plugin adapter", () => {
  it("injects the shared RuntimeCore context once for each new user message", async () => {
    const onPrompt = vi.fn(async () => {});
    const core = {
      onPrompt,
      getInjection: vi.fn(() => "<hindsight_memories>remember this</hindsight_memories>"),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createClineHooks(core as never, "session-1");
    const snapshot = { agentId: "agent-1", messages: [message("u-1", "user", "plan the change")] };
    const request = { messages: snapshot.messages };

    const first = await hooks.beforeModel({ snapshot, request });
    const second = await hooks.beforeModel({ snapshot, request });

    expect(onPrompt).toHaveBeenCalledOnce();
    expect(onPrompt).toHaveBeenCalledWith("session-1", "plan the change");
    expect(first?.messages.at(-1)).toMatchObject({
      role: "user",
      content: [{ type: "text", text: "<hindsight_memories>remember this</hindsight_memories>" }],
      metadata: { hindsight_coding_agents_injection: true },
    });
    expect(second?.messages).toHaveLength(2);
  });

  it("waits for the shared SessionStart decision before the first reflect", async () => {
    let releaseSessionStart: (() => void) | undefined;
    const sessionStart = new Promise<void>((resolve) => {
      releaseSessionStart = resolve;
    });
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createClineHooks(core as never, "session-1", sessionStart);
    const pending = hooks.beforeModel({
      snapshot: { agentId: "agent-1", messages: [message("u-1", "user", "first prompt")] },
      request: { messages: [message("u-1", "user", "first prompt")] },
    });

    await Promise.resolve();
    expect(core.onPrompt).not.toHaveBeenCalled();
    releaseSessionStart?.();
    await pending;
    expect(core.onPrompt).toHaveBeenCalledWith("session-1", "first prompt");
  });

  it(
    "does not let a slow shared reflect exceed Cline's sandbox hook budget",
    async () => {
      const core = {
        onPrompt: vi.fn(() => new Promise<void>(() => {})),
        getInjection: vi.fn(() => undefined),
        onTranscript: vi.fn(async () => {}),
      };
      const hooks = createClineHooks(core as never, "session-1");

      await expect(
        hooks.beforeModel({
          snapshot: { agentId: "agent-1", messages: [message("u-1", "user", "slow reflect")] },
          request: { messages: [message("u-1", "user", "slow reflect")] },
        })
      ).resolves.toBeUndefined();
    },
    CLINE_HOOK_BUDGET_MS + 1_000
  );

  // seedIfCold waits on a cold local daemon (#3524), which outlasts the sandbox's 3s RPC abort by
  // design — so the SessionStart await has to be bounded exactly like the reflect above.
  it(
    "does not let a cold daemon start exceed Cline's sandbox hook budget",
    async () => {
      const core = {
        onPrompt: vi.fn(async () => {}),
        getInjection: vi.fn(() => undefined),
        onTranscript: vi.fn(async () => {}),
      };
      const hooks = createClineHooks(core as never, "session-1", new Promise<void>(() => {}));

      await expect(
        hooks.beforeModel({
          snapshot: { agentId: "agent-1", messages: [message("u-1", "user", "cold daemon")] },
          request: { messages: [message("u-1", "user", "cold daemon")] },
        })
      ).resolves.toBeUndefined();
    },
    CLINE_HOOK_BUDGET_MS + 1_000
  );

  it("retains only user-visible text and excludes tool arguments/results", async () => {
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createClineHooks(core as never);
    const snapshot = {
      agentId: "agent-1",
      conversationId: "conversation-1",
      messages: [
        message("u-1", "user", "remember the preference"),
        {
          id: "a-1",
          role: "assistant" as const,
          content: [
            { type: "text", text: "I will do that." },
            { type: "tool-call", toolName: "read_file", input: { path: "README.md" } },
          ],
          createdAt: Date.parse("2026-07-31T10:01:00Z"),
        },
        {
          id: "t-1",
          role: "tool" as const,
          content: [{ type: "tool-result", toolName: "read_file", output: "secret file output" }],
          createdAt: Date.parse("2026-07-31T10:01:01Z"),
        },
      ],
    };

    await hooks.afterRun({ snapshot });

    expect(core.onTranscript).toHaveBeenCalledWith("conversation-1", [
      expect.objectContaining({ role: "user", content: "remember the preference" }),
      expect.objectContaining({ role: "assistant", content: "I will do that." }),
    ]);
    expect(clineTranscript(snapshot.messages)).toEqual([
      expect.objectContaining({ role: "user", content: "remember the preference" }),
      expect.objectContaining({ role: "assistant", content: "I will do that." }),
    ]);
  });
});
