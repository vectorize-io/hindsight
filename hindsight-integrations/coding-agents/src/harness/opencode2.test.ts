/**
 * The opencode2 adapter's hook wiring, driven against a fake plugin context.
 *
 * The shapes here are the ones a live opencode2 (0.0.0-beta-18743) actually hands a plugin, so
 * these tests fail if the v2 contract we mapped onto RuntimeCore drifts under us: v2 offers no
 * "your hook never fired" signal, so a silently mis-wired hook is otherwise indistinguishable from
 * an agent with no memory.
 */
import { describe, expect, it, vi } from "vitest";
import type { RuntimeCore } from "../core/runtime";
import type { ToolSpec } from "../core/knowledge-tools";
import { wireOpencode2Runtime } from "./opencode2";

type PromptHook = (input: { sessionID: string; prompt?: { text?: string } }) => Promise<void>;
type ContextHook = (input: {
  sessionID: string;
  system: { type: string; text: string }[];
}) => Promise<void>;
type ToolDraft = { add(tool: Record<string, unknown>): void };

/**
 * A fake ctx that records what the adapter registers and lets a test drive each hook.
 *
 * Events are pushed by the test through `emit`, never queued up front: the write-back is gated on
 * ids the prompt hook admitted, so a test that cannot say "these hooks ran, THEN this event
 * arrived" would be asserting on a race rather than on the guard.
 */
function fakeContext() {
  const hooks: Record<string, unknown> = {};
  const tools: Record<string, any>[] = [];
  let subscribeSignal: AbortSignal | undefined;
  const contextCalls: string[] = [];
  const pending: unknown[] = [];
  let waiting: (() => void) | undefined;
  return {
    tools,
    contextCalls,
    prompt: () => hooks.prompt as PromptHook,
    context: () => hooks.context as ContextHook,
    /** Deliver events to the adapter's subscription, then let its loop drain. */
    async emit(...events: unknown[]) {
      for (const e of events) {
        pending.push(e);
        waiting?.();
      }
      // Two macrotask hops: one to hand each event to the loop, one for its handler to run.
      await new Promise((r) => setTimeout(r, 0));
      await new Promise((r) => setTimeout(r, 0));
    },
    get subscribeSignal() {
      return subscribeSignal;
    },
    ctx: {
      session: {
        hook: async (name: string, cb: unknown) => {
          hooks[name] = cb;
          return { dispose: async () => {} };
        },
        context: async ({ sessionID }: { sessionID: string }) => {
          contextCalls.push(sessionID);
          return [{ type: "user", text: "hi" }];
        },
      },
      tool: {
        transform: async (cb: (draft: ToolDraft) => void) => {
          cb({ add: (t) => tools.push(t as Record<string, any>) });
          return { dispose: async () => {} };
        },
      },
      event: {
        subscribe: (opts?: { signal?: AbortSignal }) => {
          subscribeSignal = opts?.signal;
          return (async function* () {
            while (!opts?.signal?.aborted) {
              if (!pending.length) await new Promise<void>((r) => (waiting = r));
              while (pending.length) yield pending.shift();
            }
          })();
        },
      },
    } as never,
  };
}

function fakeCore(overrides: Partial<Record<string, unknown>> = {}) {
  const spec: ToolSpec = {
    name: "hindsight_recall",
    description: "recall",
    inputSchema: {},
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: false,
    },
    handler: async () => ({ content: [{ type: "text", text: "ok" }] }),
  };
  const core = {
    harness: "opencode2",
    toolSpecs: () => [spec],
    setTranscriptSource: vi.fn(),
    onPrompt: vi.fn(async () => {}),
    getInjection: vi.fn(() => "<hindsight_memories>remember</hindsight_memories>"),
    onSessionIdle: vi.fn(async () => {}),
    ...overrides,
  };
  return core as unknown as RuntimeCore & typeof core;
}

describe("opencode2 adapter", () => {
  it("registers every hindsight tool with codemode disabled", async () => {
    // NOT decoration: a v2 tool left on the default is only reachable through the host's `execute`
    // code-execution tool, so a model calling it by name — which is what the skill and the injected
    // preamble tell it to do — gets "Unknown tool: hindsight_recall".
    const fake = fakeContext();
    await wireOpencode2Runtime(fakeCore(), fake.ctx);
    expect(fake.tools).toHaveLength(1);
    expect(fake.tools[0].name).toBe("hindsight_recall");
    expect(fake.tools[0].options).toEqual({ codemode: false });
    // v2 wants a whole schema, not v1's raw Zod shape.
    expect(typeof fake.tools[0].input?.parse).toBe("function");
  });

  it("surfaces a tool's MCP-shaped result as v2 text content", async () => {
    const fake = fakeContext();
    await wireOpencode2Runtime(fakeCore(), fake.ctx);
    await expect(fake.tools[0].execute({})).resolves.toEqual({ content: "ok" });
  });

  it("recalls on the user's prompt text", async () => {
    const fake = fakeContext();
    const core = fakeCore();
    await wireOpencode2Runtime(core, fake.ctx);
    await fake.prompt()({ sessionID: "ses_1", prompt: { text: "why do we retry?" } });
    expect(core.onPrompt).toHaveBeenCalledWith("ses_1", "why do we retry?");
  });

  it("pushes this turn's injection into the system prompt, keyed by session", async () => {
    const fake = fakeContext();
    const core = fakeCore();
    await wireOpencode2Runtime(core, fake.ctx);
    const system = [{ type: "text", text: "you are opencode" }];
    await fake.context()({ sessionID: "ses_1", system });
    expect(core.getInjection).toHaveBeenCalledWith("ses_1");
    expect(system).toEqual([
      { type: "text", text: "you are opencode" },
      { type: "text", text: "<hindsight_memories>remember</hindsight_memories>" },
    ]);
  });

  it("leaves the system prompt untouched when there is nothing to inject", async () => {
    const fake = fakeContext();
    await wireOpencode2Runtime(fakeCore({ getInjection: () => undefined }), fake.ctx);
    const system: { type: string; text: string }[] = [];
    await fake.context()({ sessionID: "ses_1", system });
    expect(system).toEqual([]);
  });

  it("writes back on BOTH idle signals and ignores everything else", async () => {
    // The interactive TUI settles on session.idle; a one-shot `opencode2 run` exits before that
    // ever fires and only emits session.execution.succeeded. Missing either one loses a session's
    // final exchange — the exact gap `onSessionIdle` exists to close.
    const fake = fakeContext();
    const core = fakeCore();
    await wireOpencode2Runtime(core, fake.ctx);
    // Both ids have to be admitted by our own prompt hook first — see the isolation test below.
    await fake.prompt()({ sessionID: "ses_run", prompt: { text: "a" } });
    await fake.prompt()({ sessionID: "ses_tui", prompt: { text: "b" } });
    await fake.emit(
      { type: "session.text.delta", data: { sessionID: "ses_run" } },
      { type: "session.execution.succeeded", data: { sessionID: "ses_run" } },
      { type: "session.idle", data: { sessionID: "ses_tui" } },
      { type: "session.idle", data: {} } // no session id: skipped, must not throw
    );
    expect(core.onSessionIdle.mock.calls).toEqual([["ses_run"], ["ses_tui"]]);
  });

  it("never writes back a session another project's plugin instance owns", async () => {
    // Bank isolation, not tidiness. v2's background service hosts every open project at once and
    // `event.subscribe()` is GLOBAL — a plugin loaded for project A really does receive
    // session.idle for project B's sessions, with no location on the envelope to tell them apart
    // (verified against 0.0.0-beta-18743). The session hooks, by contrast, only ever fire for this
    // instance's own location. Retaining an unadmitted id would pull another project's transcript
    // into THIS project's bank.
    const fake = fakeContext();
    const core = fakeCore();
    await wireOpencode2Runtime(core, fake.ctx);
    await fake.prompt()({ sessionID: "ses_mine", prompt: { text: "only this one is ours" } });
    await fake.emit(
      { type: "session.idle", data: { sessionID: "ses_mine" } },
      { type: "session.idle", data: { sessionID: "ses_other_project" } }
    );
    expect(core.onSessionIdle.mock.calls).toEqual([["ses_mine"]]);
  });

  it("reads the transcript back through session.context", async () => {
    const fake = fakeContext();
    const core = fakeCore();
    await wireOpencode2Runtime(core, fake.ctx);
    const source = core.setTranscriptSource.mock.calls[0][0] as (s: string) => Promise<unknown>;
    await expect(source("ses_1")).resolves.toEqual([{ role: "user", content: "hi" }]);
    expect(fake.contextCalls).toEqual(["ses_1"]);
  });

  it("aborts the event subscription on teardown", async () => {
    const fake = fakeContext();
    const cleanup = await wireOpencode2Runtime(fakeCore(), fake.ctx);
    expect(fake.subscribeSignal?.aborted).toBe(false);
    cleanup();
    expect(fake.subscribeSignal?.aborted).toBe(true);
  });
});
