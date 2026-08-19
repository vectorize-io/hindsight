import { z } from "zod";
import { describe, expect, it, vi } from "vitest";
import type { ToolSpec } from "./core/knowledge-tools";
import { createPrimeAgentHooks, toPrimeAgentTool } from "./prime-agent";

describe("Prime Agent extension adapter", () => {
  it("recalls on each prompt and appends the injection to the system prompt", async () => {
    const onPrompt = vi.fn(async () => {});
    const core = {
      onPrompt,
      getInjection: vi.fn(() => "<hindsight_memories>remember this</hindsight_memories>"),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPrimeAgentHooks(core as never);

    const result = await hooks.beforeAgentStart(
      { prompt: "  plan the change  ", systemPrompt: "You are Prime Agent." },
      "session-1"
    );

    expect(onPrompt).toHaveBeenCalledOnce();
    expect(onPrompt).toHaveBeenCalledWith("session-1", "plan the change");
    expect(result?.systemPrompt).toBe(
      "You are Prime Agent.\n\n<hindsight_memories>remember this</hindsight_memories>"
    );
  });

  it("injects nothing when the core has no injection for this turn", async () => {
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPrimeAgentHooks(core as never);
    const result = await hooks.beforeAgentStart({ prompt: "hi", systemPrompt: "sys" }, "session-1");
    expect(result).toBeUndefined();
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
    const hooks = createPrimeAgentHooks(core as never, sessionStart);
    const pending = hooks.beforeAgentStart(
      { prompt: "first prompt", systemPrompt: "sys" },
      "session-1"
    );

    await Promise.resolve();
    expect(core.onPrompt).not.toHaveBeenCalled();
    releaseSessionStart?.();
    await pending;
    expect(core.onPrompt).toHaveBeenCalledWith("session-1", "first prompt");
  });

  it("writes back the converted transcript on agent_end", async () => {
    const onTranscript = vi.fn(async () => {});
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript,
    };
    const hooks = createPrimeAgentHooks(core as never);

    await hooks.agentEnd(
      {
        messages: [
          { role: "user", content: [{ type: "text", text: "remember the preference" }] },
          {
            role: "assistant",
            content: [
              { type: "text", text: "I will do that." },
              { type: "toolCall", name: "read", arguments: { path: "README.md" } },
            ],
          },
        ],
      },
      "session-1"
    );

    expect(onTranscript).toHaveBeenCalledWith("session-1", [
      { role: "user", content: "remember the preference" },
      { role: "assistant", content: "I will do that." },
      { role: "action", content: "read README.md" },
    ]);
  });

  it("does not write back an empty exchange", async () => {
    const onTranscript = vi.fn(async () => {});
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript,
    };
    const hooks = createPrimeAgentHooks(core as never);
    await hooks.agentEnd({ messages: [] }, "session-1");
    expect(onTranscript).not.toHaveBeenCalled();
  });

  it("adapts a knowledge ToolSpec into a Prime Agent native tool with a JSON-Schema parameters object", async () => {
    const spec: ToolSpec = {
      name: "hindsight_search_knowledge_pages",
      description: "Search the knowledge pages",
      inputSchema: { query: z.string(), limit: z.number().optional() },
      handler: async () => ({ content: [{ type: "text", text: "page A\npage B" }] }),
    };

    const def = toPrimeAgentTool(spec);
    expect(def.name).toBe("hindsight_search_knowledge_pages");
    expect(def.label).toBe("hindsight_search_knowledge_pages");
    expect(def.parameters.type).toBe("object");
    expect((def.parameters.properties as Record<string, unknown>).query).toBeDefined();

    const result = await def.execute("call-1", { query: "x" });
    expect(result.content).toEqual([{ type: "text", text: "page A\npage B" }]);
  });
});
