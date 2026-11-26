import { z } from "zod";
import { describe, expect, it, vi } from "vitest";
import type { ToolSpec } from "../core/page-tools";
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

  it("handles a custom-message run, which has no prompt, without throwing (#5592)", async () => {
    // pi's goal resume and `pi.sendMessage(…, { triggerTurn: true })` start a run with
    // `messageType: "custom"` and `message` instead of `prompt`.
    const onPrompt = vi.fn(async () => {});
    const core = {
      onPrompt,
      getInjection: vi.fn(() => undefined),
      onTranscript: vi.fn(async () => {}),
    };
    const hooks = createPiHooks(core as never, "pi");

    await hooks.beforeAgentStart(
      { messageType: "custom", message: { content: "Goal resumed" } },
      "s"
    );
    await hooks.beforeAgentStart({ messageType: "user", prompt: " next " }, "s");

    // The custom message is not the user's words: no recall for it, and the last block is dropped.
    expect(onPrompt.mock.calls).toEqual([
      ["s", ""],
      ["s", "next"],
    ]);
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

  it("adapts a page ToolSpec into a pi native tool with a JSON-Schema parameters object", async () => {
    const spec: ToolSpec = {
      name: "hindsight_search_pages",
      description: "Search the pages",
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
    expect(def.name).toBe("hindsight_search_pages");
    expect(def.label).toBe("hindsight_search_pages");
    expect(def.parameters.type).toBe("object");
    expect((def.parameters.properties as Record<string, unknown>).query).toBeDefined();

    const result = await def.execute("call-1", { query: "x" });
    expect(result.content).toEqual([{ type: "text", text: "page A\npage B" }]);
  });
});

describe("pi extension factory", () => {
  /** A fake RuntimeCore per repository; its one tool answers with the repository it is bound to. */
  function fakeCore(repo: string) {
    let notify: ((title: string, message: string) => void) | undefined;
    return {
      /** Raise a notice the way RuntimeCore does: through whatever notifier the adapter set. */
      raise: (title: string, message: string) => notify?.(title, message),
      setNotifier: vi.fn((n: (title: string, message: string) => void) => (notify = n)),
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

  const ctx = (cwd: string, session: string, extra: Record<string, unknown> = {}) => ({
    cwd,
    sessionManager: { getSessionId: () => session },
    ...extra,
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

  /**
   * #5557: the core raises notices (the session-start status, the per-turn recall preview) through
   * `setNotifier`, and this adapter never set one, so pi users saw none of them.
   */
  describe("notices", () => {
    /** One extension per test, with every repository's fake core reachable by directory. */
    function setup() {
      const cores = new Map<string, ReturnType<typeof fakeCore>>();
      const pi = fakePi();
      createPiExtension("pi", (_h, dir) => {
        const core = fakeCore(dir);
        // Like RuntimeCore.seedIfCold: the status notice is raised as the workspace is resolved,
        // which for the launch directory is extension load, before any event carries a ctx.
        core.seedIfCold.mockImplementation(async () => core.raise("Hindsight", `Is on in ${dir}`));
        cores.set(dir, core);
        return core as never;
      })(pi.api as never);
      const ui = () => ({ notify: vi.fn() });
      return { cores, pi, ui, launch: process.cwd() };
    }

    it("holds the notice raised at load and shows it on the first event with a UI", async () => {
      const { pi, ui, launch } = setup();
      const shown = ui();
      pi.handlers.session_start({}, ctx(launch, "s", { hasUI: true, ui: shown }));
      expect(shown.notify).toHaveBeenCalledExactlyOnceWith(`Hindsight: Is on in ${launch}`, "info");
    });

    it("shows a notice raised during a turn right away, on that turn's UI", async () => {
      const { cores, pi, ui, launch } = setup();
      const shown = ui();
      cores
        .get(launch)!
        .onPrompt.mockImplementation(async () =>
          cores.get(launch)!.raise("Hindsight · recalled past decisions", "we round half up")
        );
      await pi.handlers.before_agent_start(
        { prompt: "how do we round?" },
        ctx(launch, "s", { hasUI: true, ui: shown })
      );
      expect(shown.notify.mock.calls).toEqual([
        [`Hindsight: Is on in ${launch}`, "info"],
        ["Hindsight · recalled past decisions: we round half up", "info"],
      ]);
    });

    it("shows nothing in print or json mode, and holds nothing back for later", async () => {
      const { cores, pi, ui, launch } = setup();
      const silent = ui();
      pi.handlers.session_start({}, ctx(launch, "s", { hasUI: false, ui: silent }));
      cores.get(launch)!.raise("Hindsight · recalled past decisions", "later");
      expect(silent.notify).not.toHaveBeenCalled();
      // A UI that shows up afterwards (pi's web UI serves sessions from one process) gets only what
      // is raised from then on, not a backlog from the headless run.
      const shown = ui();
      pi.handlers.session_start({}, ctx(launch, "s2", { hasUI: true, ui: shown }));
      expect(shown.notify).not.toHaveBeenCalled();
      cores.get(launch)!.raise("Hindsight", "now");
      expect(shown.notify).toHaveBeenCalledExactlyOnceWith("Hindsight: now", "info");
    });

    it("works with a host that passes no UI at all", async () => {
      const { cores, pi, launch } = setup();
      pi.handlers.session_start({}, ctx(launch, "s")); // an older pi-family host: no ui, no hasUI
      expect(() => cores.get(launch)!.raise("Hindsight", "x")).not.toThrow();
      await expect(
        pi.handlers.before_agent_start({ prompt: "hi" }, ctx(launch, "s"))
      ).resolves.toBeUndefined();
    });

    it("never lets a broken notice line break the turn", async () => {
      const { cores, pi, launch } = setup();
      const broken = {
        notify: vi.fn(() => {
          throw new Error("ui gone");
        }),
      };
      cores
        .get(launch)!
        .onPrompt.mockImplementation(async () => cores.get(launch)!.raise("Hindsight", "x"));
      await expect(
        pi.handlers.before_agent_start({ prompt: "hi" }, ctx(launch, "s", { ui: broken }))
      ).resolves.toBeUndefined();
      expect(broken.notify).toHaveBeenCalledTimes(2); // the held status, then the turn's notice
    });

    it("holds at most 8 notices while no event has arrived", async () => {
      const { cores, pi, ui, launch } = setup();
      for (let i = 0; i < 20; i++) cores.get(launch)!.raise("Hindsight", `n${i}`);
      const shown = ui();
      pi.handlers.session_start({}, ctx(launch, "s", { ui: shown }));
      expect(shown.notify).toHaveBeenCalledTimes(8); // the load-time status plus n0..n6
      expect(shown.notify.mock.calls.at(-1)).toEqual(["Hindsight: n6", "info"]);
    });

    it("shows each repository's notices in that repository's session only", async () => {
      const { cores, pi, ui, launch } = setup();
      const inA = ui();
      const inB = ui();
      pi.handlers.session_start({}, ctx(launch, "s-a", { ui: inA }));
      pi.handlers.session_start({}, ctx("/repo-b", "s-b", { ui: inB }));
      expect(inB.notify).toHaveBeenCalledExactlyOnceWith("Hindsight: Is on in /repo-b", "info");
      cores.get(launch)!.raise("Hindsight", "for a");
      expect(inA.notify.mock.calls.at(-1)).toEqual(["Hindsight: for a", "info"]);
      expect(inB.notify).toHaveBeenCalledOnce();
    });
  });
});
