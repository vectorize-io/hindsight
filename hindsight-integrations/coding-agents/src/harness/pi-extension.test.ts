import { z } from "zod";
import { describe, expect, it, vi } from "vitest";
import type { ToolSpec } from "../core/knowledge-tools";
import { createPiExtension, createPiHooks, toPiTool } from "./pi-extension";

// The factory resolves a bank per directory; stand in for config + runtime so the test can opt a
// single directory in and see which one the adapter asked about.
const OPTED_IN = "/work/opted-in-repo";
const ALSO_OPTED_IN = "/work/second-repo";
const resolved: string[] = [];
vi.mock("../core/host-client", () => ({
  resolveHostMemory: (_harness: string, directory: string) => {
    resolved.push(directory);
    return {
      cfg: { disabled: ![OPTED_IN, ALSO_OPTED_IN].includes(directory) },
      bankId: `bank:${directory}`,
      client: {},
    };
  },
}));
vi.mock("../core/runtime", () => ({
  RuntimeCore: class {
    constructor(
      _client: unknown,
      readonly bankId: string
    ) {}
    toolSpecs() {
      return [
        {
          name: "hindsight_search_knowledge_pages",
          description: "d",
          inputSchema: {},
          handler: async () => ({ content: [{ type: "text", text: `searched ${this.bankId}` }] }),
        },
      ];
    }
    seedIfCold = vi.fn(async () => {});
    onPrompt = vi.fn(async () => {});
    getInjection = vi.fn(() => `injection for ${this.bankId}`);
    onTranscript = vi.fn(async () => {});
  },
}));

function fakePi() {
  const tools: string[] = [];
  const handlers: Record<string, (event: never, ctx: never) => unknown> = {};
  return {
    tools,
    handlers,
    pi: {
      registerTool: (def: { name: string }) => tools.push(def.name),
      on: (event: string, handler: (event: never, ctx: never) => unknown) => {
        handlers[event] = handler;
      },
    },
  };
}
const ctxFor = (cwd: string) => ({ cwd, sessionManager: { getSessionId: () => `s:${cwd}` } });

describe("pi extension adapter", () => {
  it.each(["before", "after"])("preserves sections added %s memory injection", async (order) => {
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => "<hindsight_memories>remember this</hindsight_memories>"),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPiHooks(core, "pi");
    const sections: Record<string, string> = { base: "You are pi." };
    if (order === "before") sections.other = "SECTION-MARKER";
    const result = await hooks.beforeAgentStart(
      {
        prompt: "hi",
        systemPrompt: Object.values(sections).join("\n\n"),
        systemPromptOptions: { sections },
      },
      "session-1"
    );
    if (order === "after") sections.other = "SECTION-MARKER";
    const prompt = result?.systemPrompt ?? Object.values(sections).join("\n\n");
    expect(prompt).toContain("You are pi.");
    expect(prompt).toContain("SECTION-MARKER");
    expect(prompt).toContain("<hindsight_memories>remember this</hindsight_memories>");
    expect(result).toBeUndefined();
  });

  it("recalls on each prompt and appends the injection to the system prompt", async () => {
    const onPrompt = vi.fn(async () => {});
    const core = {
      onPrompt,
      getInjection: vi.fn(() => "<hindsight_memories>remember this</hindsight_memories>"),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPiHooks(core as never, "pi");

    const result = await hooks.beforeAgentStart(
      { prompt: "  plan the change  ", systemPrompt: "You are pi." },
      "session-1"
    );

    expect(onPrompt).toHaveBeenCalledOnce();
    expect(onPrompt).toHaveBeenCalledWith("session-1", "plan the change");
    expect(result?.systemPrompt).toBe(
      "You are pi.\n\n<hindsight_memories>remember this</hindsight_memories>"
    );
  });

  it("injects nothing when the core has no injection for this turn", async () => {
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPiHooks(core as never, "pi");
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
    const hooks = createPiHooks(core as never, "pi", sessionStart);
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
    const hooks = createPiHooks(core as never, "pi");

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

    // `true`: agent_end sees the finished run — the reply's turn is complete (usage stats).
    expect(onTranscript).toHaveBeenCalledWith(
      "session-1",
      [
        { role: "user", content: "remember the preference" },
        { role: "assistant", content: "I will do that." },
        { role: "action", content: "read README.md" },
      ],
      true
    );
  });

  it("does not write back an empty exchange", async () => {
    const onTranscript = vi.fn(async () => {});
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript,
    };
    const hooks = createPiHooks(core as never, "pi");
    await hooks.agentEnd({ messages: [] }, "session-1");
    expect(onTranscript).not.toHaveBeenCalled();
  });

  it("adapts a knowledge ToolSpec into a pi native tool with a JSON-Schema parameters object", async () => {
    const spec: ToolSpec = {
      name: "hindsight_search_knowledge_pages",
      description: "Search the knowledge pages",
      inputSchema: { query: z.string(), limit: z.number().optional() },
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
      handler: async () => ({ content: [{ type: "text", text: "page A\npage B" }] }),
    };

    const def = toPiTool(spec);
    expect(def.name).toBe("hindsight_search_knowledge_pages");
    expect(def.label).toBe("hindsight_search_knowledge_pages");
    expect(def.parameters.type).toBe("object");
    expect((def.parameters.properties as Record<string, unknown>).query).toBeDefined();

    const result = await def.execute("call-1", { query: "x" });
    expect(result.content).toEqual([{ type: "text", text: "page A\npage B" }]);
  });

  it("resolves the bank from the session cwd, not the process cwd the host was launched in", async () => {
    resolved.length = 0;
    // pi-web-ui: one long-lived process, launched somewhere that is not the opted-in workspace.
    vi.spyOn(process, "cwd").mockReturnValue("/srv/pi-web-ui");
    const host = fakePi();
    createPiExtension("pi")(host.pi as never);

    const ctx = ctxFor(OPTED_IN);
    await host.handlers.session_start?.({ type: "session_start" } as never, ctx as never);
    const result = (await host.handlers.before_agent_start?.(
      { type: "before_agent_start", prompt: "hi", systemPrompt: "sys" } as never,
      ctx as never
    )) as { systemPrompt?: string } | undefined;

    expect(resolved).not.toContain("/srv/pi-web-ui");
    expect(host.tools).toContain("hindsight_search_knowledge_pages");
    expect(result?.systemPrompt).toContain(`bank:${OPTED_IN}`);
    vi.restoreAllMocks();
  });

  it("runs a tool against the bank of the session that called it", async () => {
    const host = fakePi();
    const registered: { execute: (...a: unknown[]) => Promise<{ content: { text: string }[] }> }[] =
      [];
    host.pi.registerTool = ((def: (typeof registered)[number]) => registered.push(def)) as never;
    createPiExtension("pi")(host.pi as never);

    await host.handlers.session_start?.({} as never, ctxFor(OPTED_IN) as never);
    await host.handlers.session_start?.({} as never, ctxFor(ALSO_OPTED_IN) as never);

    expect(registered).toHaveLength(1);
    const run = (cwd: string) => registered[0].execute("c", {}, undefined, undefined, { cwd });
    expect((await run(OPTED_IN)).content[0].text).toBe(`searched bank:${OPTED_IN}`);
    expect((await run(ALSO_OPTED_IN)).content[0].text).toBe(`searched bank:${ALSO_OPTED_IN}`);
  });

  it("leaves a directory that is not opted in without tools or hooks", async () => {
    const host = fakePi();
    createPiExtension("pi")(host.pi as never);
    const ctx = ctxFor("/work/not-opted-in");
    await host.handlers.session_start?.({} as never, ctx as never);
    const result = await host.handlers.before_agent_start?.(
      { type: "before_agent_start", prompt: "hi", systemPrompt: "sys" } as never,
      ctx as never
    );
    expect(host.tools).toEqual([]);
    expect(result).toBeUndefined();
  });
});
