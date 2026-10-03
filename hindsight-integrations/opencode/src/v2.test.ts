import { describe, it, expect, vi } from "vitest";

// Mock the V2 SDK so the entry can be loaded without OpenCode running.
vi.mock("@opencode/plugin", () => ({
  Plugin: { define: <T,>(definition: T): T => definition },
}));

vi.mock("@vectorize-io/hindsight-client", () => {
  const MockHindsightClient = vi.fn(function (this: any) {
    this.retain = vi.fn().mockResolvedValue({});
    this.recall = vi.fn().mockResolvedValue({ results: [] });
    this.reflect = vi.fn().mockResolvedValue({ text: "" });
    this.createBank = vi.fn().mockResolvedValue({});
  });
  return { HindsightClient: MockHindsightClient };
});

import plugin from "./index.v2.js";
import { HindsightClient } from "@vectorize-io/hindsight-client";

interface TestCtx {
  options: Record<string, unknown>;
  location: { directory: string };
  tool: { transform: (cb: (editor: { add: (t: unknown) => void }) => void) => Promise<void> };
  session: {
    hook: (name: string, cb: unknown) => Promise<void>;
    context: (input: { sessionID: string }) => Promise<unknown[]>;
  };
  event: { subscribe: (options?: { signal?: AbortSignal }) => AsyncIterable<unknown> };
  hooks: Record<string, unknown>;
  tools: Array<{ name: string }>;
}

function makeCtx(): TestCtx {
  const hooks: Record<string, unknown> = {};
  const tools: Array<{ name: string }> = [];
  return {
    options: {},
    location: { directory: "/tmp/test-project" },
    tool: {
      transform: async (cb) => {
        cb({ add: (t) => tools.push(t as { name: string }) });
      },
    },
    session: {
      hook: async (name, cb) => {
        hooks[name] = cb;
      },
      context: async () => [],
    },
    // An empty async iterable stands in for the server event stream.
    event: {
      subscribe: () =>
        (async function* () {
          /* no events */
        })(),
    },
    hooks,
    tools,
  };
}

describe("Hindsight V2 plugin entry", () => {
  it("exports a definition with an id and setup", () => {
    expect(plugin.id).toBe("hindsight");
    expect(typeof plugin.setup).toBe("function");
  });

  it("registers the three memory tools via ctx.tool.transform", async () => {
    const ctx = makeCtx();
    await plugin.setup(ctx as never);

    expect(ctx.tools.map((t) => t.name).sort()).toEqual([
      "hindsight_recall",
      "hindsight_reflect",
      "hindsight_retain",
    ]);
  });

  it("registers the context and compaction session hooks", async () => {
    const ctx = makeCtx();
    await plugin.setup(ctx as never);

    expect(typeof ctx.hooks["context"]).toBe("function");
    expect(typeof ctx.hooks["compaction"]).toBe("function");
  });

  // Guards the real OpenCode v2 contract (see #5136): plugins receive
  // `session.execution.succeeded` — NOT `session.idle` — and v2 session
  // messages use `type` + top-level `text`/`content`, not `role`/`parts`.
  it("auto-retains on session.execution.succeeded with v2 message shapes", async () => {
    const messages = [
      { type: "user", text: "I prefer dark mode and use VS Code." },
      { type: "assistant", content: [{ type: "text", text: "Noted." }] },
    ];
    const ctx = makeCtx();
    ctx.options = { retainEveryNTurns: 1 };
    ctx.session.context = async () => messages;
    ctx.event.subscribe = () =>
      (async function* () {
        yield {
          type: "session.execution.succeeded",
          data: { sessionID: "ses_contract_test" },
        };
      })();

    await plugin.setup(ctx as never);
    await new Promise((r) => setTimeout(r, 20)); // let the event loop drain

    const instances = (HindsightClient as unknown as { mock: { instances: any[] } }).mock
      .instances;
    const client = instances.at(-1);
    expect(client.retain).toHaveBeenCalled();
    const content = String(client.retain.mock.calls.at(-1)?.[1]);
    expect(content).toContain("dark mode");
  });

  it("still accepts the legacy session.idle event", async () => {
    const ctx = makeCtx();
    ctx.options = { retainEveryNTurns: 1 };
    ctx.session.context = async () => [
      { type: "user", text: "legacy path" },
      { type: "assistant", content: [{ type: "text", text: "ok" }] },
    ];
    ctx.event.subscribe = () =>
      (async function* () {
        yield { type: "session.idle", properties: { sessionID: "ses_legacy" } };
      })();

    await plugin.setup(ctx as never);
    await new Promise((r) => setTimeout(r, 20));

    const instances = (HindsightClient as unknown as { mock: { instances: any[] } }).mock
      .instances;
    expect(instances.at(-1).retain).toHaveBeenCalled();
  });
});
