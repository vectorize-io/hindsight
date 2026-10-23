import { z } from "zod";
import { describe, expect, it, vi } from "vitest";
import type { ToolSpec } from "../core/knowledge-tools";
import { createPiExtension, createPiHooks, toPiTool } from "./pi-extension";

describe("pi extension adapter", () => {
  it("recalls on each prompt and adds the injection as a hidden message, not to the system prompt", async () => {
    // The block changes between turns, so a system prompt carrying it broke the provider's prefix
    // cache (#5914); returning a systemPrompt also dropped later extensions' sections (#4841).
    const onPrompt = vi.fn(async () => {});
    const core = {
      onPrompt,
      getInjection: vi.fn(() => "<hindsight_memories>remember this</hindsight_memories>"),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPiHooks(core as never, "pi");
    const sections: Record<string, string> = { base: "You are pi." };
    const event = {
      prompt: "  plan the change  ",
      systemPrompt: "You are pi.",
      systemPromptOptions: { sections },
    };

    const result = await hooks.beforeAgentStart(event, "session-1");

    expect(onPrompt).toHaveBeenCalledOnce();
    expect(onPrompt).toHaveBeenCalledWith("session-1", "plan the change");
    expect(result).toEqual({
      message: {
        customType: "hindsight",
        content: "<hindsight_memories>remember this</hindsight_memories>",
        display: false,
      },
    });
    expect(result).not.toHaveProperty("systemPrompt");
    expect(sections).toEqual({ base: "You are pi." });
  });

  it("injects nothing when the core has no injection for this turn", async () => {
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPiHooks(core as never, "pi");
    const result = await hooks.beforeAgentStart({ prompt: "hi" }, "session-1");
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
    const pending = hooks.beforeAgentStart({ prompt: "first prompt" }, "session-1");

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

  it("writes back the whole session branch, not just the latest run's messages (#4963)", async () => {
    const onTranscript = vi.fn(async () => {});
    const flush = vi.fn(async () => {});
    const core = { onPrompt: vi.fn(), getInjection: vi.fn(), onTranscript, flush };
    const hooks = createPiHooks(core as never, "pi");
    const msg = (role: string, text: string) => ({
      type: "message",
      timestamp: "2026-10-01T00:00:00.000Z",
      message: { role, content: [{ type: "text", text }] },
    });

    // The second run's event holds only its own exchange; the session holds both runs.
    await hooks.agentEnd({ messages: [{ role: "user", content: "beta" }] }, "session-1", [
      { type: "session" },
      msg("user", "alpha"),
      msg("assistant", "noted alpha"),
      { type: "compaction" },
      msg("user", "beta"),
    ]);
    const turns = (onTranscript.mock.calls[0] as unknown[])[1] as { content: string }[];
    expect(turns.map((t) => t.content)).toEqual(["alpha", "noted alpha", "beta"]);

    // Exit drains the write-back the last agent_end started (#5081).
    await hooks.sessionShutdown();
    expect(flush).toHaveBeenCalledOnce();
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
});

describe("pi extension factory", () => {
  /** A fake RuntimeCore per repository; its one tool answers with the repository it is bound to. */
  function fakeCore(repo: string) {
    return {
      seedIfCold: vi.fn(async () => {}),
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => undefined),
      onTranscript: vi.fn(async () => {}),
      flush: vi.fn(async () => {}),
      toolSpecs: (): ToolSpec[] => [
        {
          name: "hindsight_reflect",
          description: "reflect",
          inputSchema: { query: z.string() },
          annotations: {
            readOnlyHint: true,
            destructiveHint: false,
            idempotentHint: true,
            openWorldHint: false,
          },
          handler: async () => ({ content: [{ type: "text", text: `bank of ${repo}` }] }),
        },
      ],
    };
  }

  function fakePi() {
    const handlers: Record<string, (event: unknown, ctx: unknown) => unknown> = {};
    const tools: ReturnType<typeof toPiTool>[] = [];
    const api = {
      on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => {
        handlers[event] = handler;
      },
      registerTool: (def: ReturnType<typeof toPiTool>) => tools.push(def),
    };
    return { handlers, tools, api };
  }

  const ctx = (cwd: string, session: string) => ({
    cwd,
    sessionManager: { getSessionId: () => session },
  });

  // #5039: pi's web UI serves every session from one process launched elsewhere, so the launch
  // directory must not pick the bank, nor (when it is not opted in) disable every session.
  it("resolves each session's repository from ctx.cwd, not the process cwd", async () => {
    const cores = new Map<string, ReturnType<typeof fakeCore>>();
    const makeRuntime = (_harness: string, dir: string) => {
      if (dir === process.cwd()) return undefined; // the launch directory is not opted in
      const core = fakeCore(dir);
      cores.set(dir, core);
      return core as never;
    };
    const pi = fakePi();
    createPiExtension("pi", makeRuntime)(pi.api as never);
    expect(pi.tools).toHaveLength(0);

    pi.handlers.session_start({}, ctx("/repo-a", "s-a"));
    expect(pi.tools.map((t) => t.name)).toEqual(["hindsight_reflect"]);

    await pi.handlers.before_agent_start({ prompt: "hello a" }, ctx("/repo-a", "s-a"));
    await pi.handlers.before_agent_start({ prompt: "hello b" }, ctx("/repo-b", "s-b"));
    expect(cores.get("/repo-a")?.onPrompt).toHaveBeenCalledExactlyOnceWith("s-a", "hello a");
    expect(cores.get("/repo-b")?.onPrompt).toHaveBeenCalledExactlyOnceWith("s-b", "hello b");

    // Registered once, but each call runs against the calling session's bank.
    const run = async (cwd: string) =>
      (await pi.tools[0].execute("c", { query: "q" }, undefined, undefined, ctx(cwd, "s")))
        .content[0].text;
    expect(await run("/repo-b")).toBe("bank of /repo-b");
    expect(await run(process.cwd())).toBe("Hindsight memory is disabled for this workspace.");
    expect(pi.tools).toHaveLength(1);

    // Exit drains every workspace's in-flight write-back, not just the launch one's (#5081).
    await pi.handlers.session_shutdown({}, undefined);
    expect(cores.get("/repo-a")?.flush).toHaveBeenCalledOnce();
    expect(cores.get("/repo-b")?.flush).toHaveBeenCalledOnce();
  });

  it("registers the launch directory's tools at load, as the CLI always has", () => {
    const pi = fakePi();
    createPiExtension("pi", (_h, dir) => fakeCore(dir) as never)(pi.api as never);
    expect(pi.tools.map((t) => t.name)).toEqual(["hindsight_reflect"]);
  });
});
