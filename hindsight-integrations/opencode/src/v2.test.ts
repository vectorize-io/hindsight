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
});
