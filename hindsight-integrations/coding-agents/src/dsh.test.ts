import { describe, expect, it, vi } from "vitest";
import { createDshHooks, toDshParameters, type Workspace } from "./dsh";
import type { ToolSpec } from "./core/knowledge-tools";
import { z } from "zod";

const session = (id: string, extra: Record<string, unknown> = {}) => ({
  header: { id, cwd: "/repo", ...extra },
  events: [] as unknown[],
});

const agentWith = (id: string, extra: Record<string, unknown> = {}) =>
  ({ session: session(id, extra) }) as never;

const userMessage = (text: string) => ({
  id: `m-${text}`,
  role: "user" as const,
  content: [{ type: "text", text }],
  source: { kind: "user" },
});

const pluginMessage = (text: string) => ({
  id: `p-${text}`,
  role: "user" as const,
  content: [{ type: "text", text }],
  source: { kind: "plugin", plugin: "@deepseek-ai/dsh-system-prompt" },
});

function fakeWorkspace(core: Partial<Workspace["core"]>): Workspace {
  return { core: core as Workspace["core"], root: "/repo", seeded: Promise.resolve() };
}

const enter = (messages: unknown[]) => async () => ({ kind: "enter" as const, messages }) as never;

describe("dsh pre-step injection", () => {
  it("recalls on the human prompt and appends the memory as a sourced message", async () => {
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => "<hindsight_memory>past decision</hindsight_memory>"),
      seedIfCold: vi.fn(async () => {}),
    };
    const hooks = createDshHooks(() => fakeWorkspace(core));

    const decision = await hooks.preStep(
      { agent: agentWith("s-1"), signal: new AbortController().signal },
      enter([userMessage("why did we roll back?"), pluginMessage("Current runtime context…")])
    );

    expect(core.onPrompt).toHaveBeenCalledWith("s-1", "why did we roll back?");
    expect(decision.kind).toBe("enter");
    const appended = (decision as { messages: Record<string, any>[] }).messages.at(-1)!;
    expect(appended).toMatchObject({
      role: "user",
      content: [{ type: "text", text: "<hindsight_memory>past decision</hindsight_memory>" }],
      source: { kind: "plugin", plugin: "hindsight", form: "recall" },
    });
    expect(appended.id).toEqual(expect.any(String));
  });

  it("does not recall again on a tool continuation that claimed no new input", async () => {
    const core = {
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => "memory"),
      seedIfCold: vi.fn(async () => {}),
    };
    const hooks = createDshHooks(() => fakeWorkspace(core));

    const decision = await hooks.preStep(
      { agent: agentWith("s-1"), signal: new AbortController().signal },
      enter([pluginMessage("Current runtime context…")])
    );

    expect(core.onPrompt).not.toHaveBeenCalled();
    expect((decision as { messages: unknown[] }).messages).toHaveLength(1);
  });

  it("leaves a rejected step and an aborted step untouched", async () => {
    const core = { onPrompt: vi.fn(async () => {}), getInjection: vi.fn(() => "memory") };
    const hooks = createDshHooks(() => fakeWorkspace(core));

    const rejected = await hooks.preStep(
      { agent: agentWith("s-1"), signal: new AbortController().signal },
      async () => ({ kind: "reject" }) as never
    );
    const aborted = AbortSignal.abort();
    const cancelled = await hooks.preStep(
      { agent: agentWith("s-1"), signal: aborted },
      enter([userMessage("hello")])
    );

    expect(rejected).toEqual({ kind: "reject" });
    expect((cancelled as { messages: unknown[] }).messages).toHaveLength(1);
    expect(core.onPrompt).not.toHaveBeenCalled();
  });

  it("passes the step through untouched when memory is off for the workspace", async () => {
    const hooks = createDshHooks(() => undefined);
    const decision = await hooks.preStep(
      { agent: agentWith("s-1"), signal: new AbortController().signal },
      enter([userMessage("hello")])
    );
    expect((decision as { messages: unknown[] }).messages).toHaveLength(1);
  });
});

describe("dsh write-back", () => {
  it("hands the completed exchange to the shared idle path at the stop boundary", async () => {
    const core = { onSessionIdle: vi.fn(async () => {}), seedIfCold: vi.fn(async () => {}) };
    const hooks = createDshHooks(() => fakeWorkspace(core));

    await hooks.turnStopping({ agent: agentWith("s-2") });

    expect(core.onSessionIdle).toHaveBeenCalledWith("s-2");
  });

  it("skips subagent sessions the parent conversation already covers", async () => {
    const core = {
      onSessionIdle: vi.fn(async () => {}),
      onPrompt: vi.fn(async () => {}),
      getInjection: vi.fn(() => "memory"),
      seedIfCold: vi.fn(async () => {}),
    };
    // The real resolver applies this rule; assert it rather than the fake, so the test still means
    // something if the hooks stop consulting the resolver for subagents.
    const hooks = createDshHooks((agent) =>
      (agent as unknown as { session: { header: { origin?: string } } }).session.header.origin ===
      "subagent"
        ? undefined
        : fakeWorkspace(core)
    );

    await hooks.turnStopping({ agent: agentWith("child", { origin: "subagent" }) });

    expect(core.onSessionIdle).not.toHaveBeenCalled();
  });
});

describe("dsh session start", () => {
  it("starts the repo seed exactly once, however many sessions open", () => {
    const core = { seedIfCold: vi.fn(async () => {}) };
    const workspace = { core: core as never, root: "/repo" } as Workspace;
    const hooks = createDshHooks(() => workspace);

    hooks.sessionStart({ agent: agentWith("s-1") });
    hooks.sessionStart({ agent: agentWith("s-2") });

    expect(core.seedIfCold).toHaveBeenCalledOnce();
    expect(core.seedIfCold).toHaveBeenCalledWith("/repo");
  });
});

describe("toDshParameters", () => {
  const spec = (inputSchema: ToolSpec["inputSchema"]): ToolSpec => ({
    name: "hindsight_capture_initiative",
    description: "…",
    inputSchema,
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: false,
    },
    handler: async () => ({ content: [] }),
  });

  it("projects the Zod raw shape onto dsh's parameter JSON Schema", () => {
    expect(
      toDshParameters(
        spec({
          title: z.string(),
          summary: z.string().describe("what and why"),
          relates_to_page_id: z.string().optional(),
        })
      )
    ).toEqual({
      type: "object",
      properties: {
        title: { type: "string" },
        summary: { type: "string", description: "what and why" },
        relates_to_page_id: { type: "string" },
      },
      required: ["title", "summary"],
    });
  });

  it("omits `required` for a tool that takes no arguments", () => {
    expect(toDshParameters(spec({}))).toEqual({ type: "object", properties: {} });
  });

  it("refuses a parameter shape the projection cannot express", () => {
    expect(() => toDshParameters(spec({ limit: z.number() as never }))).toThrow(
      /string parameters only/
    );
    // The guard must see THROUGH `.optional()`, not treat every optional as a string.
    expect(() => toDshParameters(spec({ limit: z.number().optional() as never }))).toThrow(
      /string parameters only/
    );
  });
});
