import { afterEach, describe, expect, it, vi } from "vitest";
import { HindsightClient, HindsightError } from "@vectorize-io/hindsight-client";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import plugin from "./index.js";
import type { MoltbotPluginAPI, PluginTool, PluginToolContext, ServiceConfig } from "./types.js";

type ToolFactory = (ctx: PluginToolContext) => PluginTool[] | null;

let service: ServiceConfig | undefined;
let directory: string | undefined;

/** Load the whole plugin against a mocked OpenClaw API, the way the gateway does. */
async function load(config: Record<string, unknown>) {
  directory = mkdtempSync(join(tmpdir(), "hindsight-tools-"));
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ api_version: "0.10.0", features: {} }), {
          headers: { "Content-Type": "application/json" },
        })
    )
  );
  const hooks = new Map<string, Parameters<MoltbotPluginAPI["on"]>[1]>();
  let factory: ToolFactory | undefined;
  let names: string[] | undefined;
  const api: MoltbotPluginAPI = {
    config: {
      plugins: {
        entries: {
          "hindsight-openclaw": {
            config: {
              hindsightApiUrl: "http://localhost:8888",
              autoRetain: false,
              retainQueuePath: join(directory, "queue.jsonl"),
              logLevel: "error",
              ...config,
            },
          },
        },
      },
    },
    registerService: (registered) => {
      service = registered;
    },
    on: (name, handler) => {
      hooks.set(name, handler);
    },
    registerTool: (f, opts) => {
      factory = f;
      names = opts?.names;
    },
    logger: { info: () => {}, warn: () => {}, error: () => {} },
  };
  plugin(api);
  await service!.start();
  return { hooks, factory: factory!, names };
}

function tool(tools: PluginTool[] | null, name: string): PluginTool {
  const found = tools?.find((t) => t.name === name);
  if (!found) throw new Error(`tool ${name} not offered`);
  return found;
}

function mockClient() {
  return {
    retain: vi
      .spyOn(HindsightClient.prototype, "retain")
      .mockResolvedValue({ success: true } as never),
    recall: vi.spyOn(HindsightClient.prototype, "recall").mockResolvedValue({
      results: [
        { id: "m1", text: "Prefers persimmons" },
        { id: "m2", text: "Lives in Rome" },
      ],
    } as never),
    reflect: vi
      .spyOn(HindsightClient.prototype, "reflect")
      .mockResolvedValue({ text: "They like autumn fruit." } as never),
  };
}

afterEach(async () => {
  await service?.stop();
  service = undefined;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  if (directory) rmSync(directory, { recursive: true, force: true });
});

describe("Hermes's tools in OpenClaw (#5635)", () => {
  it("offers Hermes's three tools, none of which takes a bank", async () => {
    const { factory, names } = await load({});
    const tools = factory({ agentId: "main", sessionKey: "agent:main:main" });
    expect(names?.slice(0, 3)).toEqual([
      "hindsight_retain",
      "hindsight_recall",
      "hindsight_reflect",
    ]);
    // Deleting knowledge is opt-in; everything else is offered.
    expect(tools?.map((t) => t.name)).toEqual(
      names?.filter((n) => n !== "hindsight_delete_knowledge_node")
    );
    for (const t of tools ?? []) {
      expect(Object.keys(t.parameters.properties as object)).not.toContain("bank_id");
    }
  });

  it("works on the agent's agentBankMap bank", async () => {
    const client = mockClient();
    const { factory } = await load({
      agentBankMap: { support: "team-bank" },
      retainTags: ["source_system:openclaw"],
    });
    const tools = factory({ agentId: "support", sessionKey: "agent:support:main" });

    const stored = await tool(tools, "hindsight_retain").execute("c1", {
      content: "The user prefers persimmons",
      context: "user preference",
      tags: ["fruit", "source_system:openclaw"],
      occurred_at: "2026-08-20",
    });
    expect(stored.content[0].text).toBe("Memory queued for storage.");
    expect(client.retain).toHaveBeenCalledWith(
      "team-bank",
      "The user prefers persimmons",
      expect.objectContaining({
        context: "user preference",
        tags: ["source_system:openclaw", "fruit"],
        timestamp: "2026-08-20",
        async: true,
      })
    );

    const recalled = await tool(tools, "hindsight_recall").execute("c2", { query: "fruit" });
    expect(client.recall).toHaveBeenCalledWith(
      "team-bank",
      "fruit",
      expect.objectContaining({ budget: "mid", maxTokens: 1024, types: ["observation"] })
    );
    expect(recalled.content[0].text).toBe("1. Prefers persimmons\n2. Lives in Rome");

    const reflected = await tool(tools, "hindsight_reflect").execute("c3", { query: "Why?" });
    expect(client.reflect).toHaveBeenCalledWith(
      "team-bank",
      "Why?",
      expect.objectContaining({ budget: "mid" })
    );
    expect(reflected.content[0].text).toBe("They like autumn fruit.");
  });

  it("works on the bank auto-recall used for the same session", async () => {
    const client = mockClient();
    const { hooks, factory } = await load({});
    const sessionKey = "agent:main:telegram:direct:alice";
    await hooks.get("before_prompt_build")!(
      { rawMessage: "what fruit do I like?", messages: [] },
      {
        agentId: "main",
        sessionKey,
        messageProvider: "telegram",
        channelId: "alice",
        senderId: "alice",
      }
    );
    const autoRecallBank = client.recall.mock.calls[0][0];
    expect(autoRecallBank).toContain("alice");

    // The tool context carries no channel or sender: the plugin fills them from the session.
    const tools = factory({ agentId: "main", sessionKey });
    await tool(tools, "hindsight_recall").execute("c1", { query: "fruit" });
    expect(client.recall.mock.calls[1][0]).toBe(autoRecallBank);
  });

  it("refuses a session without a stable identity instead of guessing a bank", async () => {
    const client = mockClient();
    const { factory } = await load({ dynamicBankGranularity: ["user"] });
    const tools = factory({ agentId: "main", sessionKey: "agent:main:main" });
    await expect(tool(tools, "hindsight_recall").execute("c1", { query: "fruit" })).rejects.toThrow(
      /not available in this session/
    );
    expect(client.recall).not.toHaveBeenCalled();
  });

  it("answers 'nothing found' for a bank nothing was retained to yet", async () => {
    vi.spyOn(HindsightClient.prototype, "recall").mockRejectedValue(
      new HindsightError("not found", 404)
    );
    const { factory } = await load({ bankId: "fresh", dynamicBankId: false });
    const tools = factory({ agentId: "main", sessionKey: "agent:main:main" });
    const result = await tool(tools, "hindsight_recall").execute("c1", { query: "fruit" });
    expect(result.content[0].text).toBe("No relevant memories found.");
  });

  it("offers no tools in memoryMode 'context'", async () => {
    const { factory } = await load({ memoryMode: "context" });
    expect(factory({ agentId: "main", sessionKey: "agent:main:main" })).toBeNull();
  });

  it("honours the session filters auto-recall and auto-retain use", async () => {
    const { factory } = await load({
      ignoreSessionPatterns: ["agent:*:ignored:**"],
      statelessSessionPatterns: ["agent:*:readonly:**"],
      skipStatelessSessions: false,
    });
    expect(factory({ agentId: "main", sessionKey: "agent:main:ignored:x" })).toBeNull();
    expect(
      factory({ agentId: "main", sessionKey: "agent:main:readonly:x" })?.map((t) => t.name)
    ).toEqual([
      "hindsight_recall",
      "hindsight_reflect",
      "hindsight_get_knowledge_base_tree",
      "hindsight_search_knowledge_base",
      "hindsight_get_knowledge_page",
    ]);
  });

  it("offers no tools to a stateless session while skipStatelessSessions is on", async () => {
    const { factory } = await load({ statelessSessionPatterns: ["agent:*:readonly:**"] });
    expect(factory({ agentId: "main", sessionKey: "agent:main:readonly:x" })).toBeNull();
  });

  it("drops auto-recall but keeps the tools in memoryMode 'tools'", async () => {
    const client = mockClient();
    const { hooks, factory } = await load({
      memoryMode: "tools",
      bankId: "solo",
      dynamicBankId: false,
    });
    await hooks.get("before_prompt_build")!(
      { rawMessage: "what fruit do I like?", messages: [] },
      { agentId: "main", sessionKey: "agent:main:main", messageProvider: "webchat" }
    );
    expect(client.recall).not.toHaveBeenCalled();

    const tools = factory({ agentId: "main", sessionKey: "agent:main:main" });
    await tool(tools, "hindsight_recall").execute("c1", { query: "fruit" });
    expect(client.recall).toHaveBeenCalledWith("solo", "fruit", expect.anything());
  });
});

describe("knowledge-base tools", () => {
  function mockKnowledge() {
    const p = HindsightClient.prototype;
    return {
      tree: vi.spyOn(p, "getKnowledgeBaseTree").mockResolvedValue({ roots: [] } as never),
      search: vi
        .spyOn(p, "searchKnowledgeBase")
        .mockResolvedValue({ results: [{ id: "kp-1" }], total: 1 } as never),
      page: vi.spyOn(p, "getKnowledgePage").mockResolvedValue({ id: "kp-1" } as never),
      folder: vi.spyOn(p, "createKnowledgeFolder").mockResolvedValue({ id: "kf-1" } as never),
      create: vi.spyOn(p, "createKnowledgePage").mockResolvedValue({ page_id: "kp-2" } as never),
      update: vi.spyOn(p, "updateKnowledgeNode").mockResolvedValue({ id: "kp-2" } as never),
      remove: vi.spyOn(p, "deleteKnowledgeNode").mockResolvedValue({} as never),
    };
  }
  const session = { agentId: "support", sessionKey: "agent:support:main" };

  it("reads and writes the agent's own bank", async () => {
    const kb = mockKnowledge();
    const { factory } = await load({ agentBankMap: { support: "team-bank" } });
    const tools = factory(session);

    await tool(tools, "hindsight_get_knowledge_base_tree").execute("c1", {});
    expect(kb.tree).toHaveBeenCalledWith("team-bank", expect.anything());

    await tool(tools, "hindsight_search_knowledge_base").execute("c2", { query: "q", limit: 500 });
    expect(kb.search).toHaveBeenCalledWith(
      "team-bank",
      "q",
      expect.objectContaining({ limit: 50 })
    );

    const page = await tool(tools, "hindsight_get_knowledge_page").execute("c3", {
      page_id: "kp-1",
    });
    expect(kb.page).toHaveBeenCalledWith("team-bank", "kp-1", expect.anything());
    expect(JSON.parse(page.content[0].text)).toEqual({ id: "kp-1" });

    await tool(tools, "hindsight_create_knowledge_folder").execute("c4", { name: "People" });
    expect(kb.folder).toHaveBeenCalledWith("team-bank", "People", expect.anything());

    await tool(tools, "hindsight_create_knowledge_page").execute("c5", {
      name: "Food",
      source_query: "What does the user like to eat?",
      parent_id: "kf-1",
      tags: ["user:alice"],
    });
    expect(kb.create).toHaveBeenCalledWith(
      "team-bank",
      "Food",
      "What does the user like to eat?",
      expect.objectContaining({ parentId: "kf-1", tags: ["user:alice"] })
    );

    await tool(tools, "hindsight_update_knowledge_node").execute("c6", {
      node_id: "kp-2",
      parent_id: "root",
    });
    expect(kb.update).toHaveBeenCalledWith(
      "team-bank",
      "kp-2",
      expect.objectContaining({ parentId: null })
    );
  });

  it("refuses an update that changes nothing", async () => {
    const kb = mockKnowledge();
    const { factory } = await load({ bankId: "b", dynamicBankId: false });
    await expect(
      tool(factory(session), "hindsight_update_knowledge_node").execute("c1", { node_id: "kp-1" })
    ).rejects.toThrow(/Provide name/);
    expect(kb.update).not.toHaveBeenCalled();
  });

  it("offers delete only with allowKnowledgeDelete", async () => {
    const kb = mockKnowledge();
    const { factory } = await load({
      bankId: "b",
      dynamicBankId: false,
      allowKnowledgeDelete: true,
    });
    await tool(factory(session), "hindsight_delete_knowledge_node").execute("c1", {
      node_id: "kf-1",
    });
    expect(kb.remove).toHaveBeenCalledWith("b", "kf-1", expect.anything());
  });

  it("gives a read-only session the reading tools only", async () => {
    const { factory } = await load({
      statelessSessionPatterns: ["agent:*:readonly:**"],
      skipStatelessSessions: false,
      allowKnowledgeDelete: true,
    });
    expect(
      factory({ agentId: "main", sessionKey: "agent:main:readonly:x" })?.map((t) => t.name)
    ).toEqual([
      "hindsight_recall",
      "hindsight_reflect",
      "hindsight_get_knowledge_base_tree",
      "hindsight_search_knowledge_base",
      "hindsight_get_knowledge_page",
    ]);
  });
});
