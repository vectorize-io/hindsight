import { afterEach, describe, expect, it, vi } from "vitest";
import { stopLogger } from "./logger.js";
import type { MoltbotPluginAPI, PluginToolContext } from "./types.js";

const { createKnowledgeTools } = vi.hoisted(() => ({
  createKnowledgeTools: vi.fn(),
}));

vi.mock("@vectorize-io/hindsight-agent-sdk", () => ({
  TOOL_NAMES: [
    "agent_knowledge_list_pages",
    "agent_knowledge_get_page",
    "agent_knowledge_create_page",
    "agent_knowledge_update_page",
    "agent_knowledge_delete_page",
    "agent_knowledge_recall",
    "agent_knowledge_reflect",
    "agent_knowledge_ingest",
  ],
  createKnowledgeTools,
}));

import plugin from "./index.js";

function makeToolApi(rawConfig: Record<string, unknown>): {
  api: MoltbotPluginAPI;
  factory: (ctx: PluginToolContext) => Array<{
    name: string;
    execute: (id: string, params: Record<string, unknown>) => Promise<unknown>;
  }>;
} {
  let factory: (ctx: PluginToolContext) => Array<{
    name: string;
    execute: (id: string, params: Record<string, unknown>) => Promise<unknown>;
  }> = () => [];
  const api = {
    config: { plugins: { entries: { "hindsight-openclaw": { config: rawConfig } } } },
    registerService: () => undefined,
    on: () => undefined,
    registerTool: (next: typeof factory) => {
      factory = next;
    },
    logger: { info: () => undefined, warn: () => undefined, error: () => undefined },
  } as unknown as MoltbotPluginAPI;
  return {
    api,
    get factory() {
      return factory;
    },
  } as unknown as {
    api: MoltbotPluginAPI;
    factory: typeof factory;
  };
}

describe("knowledge tool recall defaults (#5057)", () => {
  afterEach(() => {
    createKnowledgeTools.mockReset();
    stopLogger();
  });

  it("passes plugin recallTypes, preferObservations, minScores, and budget to the SDK", () => {
    createKnowledgeTools.mockReturnValue([]);
    const harness = makeToolApi({
      enableKnowledgeTools: true,
      dynamicBankId: false,
      bankId: "shared-bank",
      recallTypes: ["observation", "world", "experience"],
      preferObservations: true,
      recallMinScores: { reranker: 0.3, semantic: null },
      recallBudget: "high",
    });

    plugin(harness.api);
    harness.factory({ agentId: "main", sessionKey: "agent:main:main" });

    expect(createKnowledgeTools).toHaveBeenCalledWith(
      expect.objectContaining({
        bankId: "shared-bank",
        recallFactTypes: ["observation", "world", "experience"],
        preferObservations: true,
        minScores: { reranker: 0.3, semantic: null },
        recallBudget: "high",
      })
    );
  });

  it("defaults manual recall to observation, preferObservations false, and budget mid", () => {
    createKnowledgeTools.mockReturnValue([]);
    const harness = makeToolApi({
      enableKnowledgeTools: true,
      dynamicBankId: false,
      bankId: "shared-bank",
    });

    plugin(harness.api);
    harness.factory({});

    expect(createKnowledgeTools).toHaveBeenCalledWith(
      expect.objectContaining({
        recallFactTypes: ["observation"],
        preferObservations: false,
        recallBudget: "mid",
        minScores: undefined,
      })
    );
  });

  it("forwards explicit recall arguments to the SDK tool unchanged", async () => {
    const execute = vi.fn(async () => ({ content: [{ type: "text", text: "{}" }] }));
    createKnowledgeTools.mockReturnValue([
      {
        name: "agent_knowledge_recall",
        label: "Search memories",
        description: "search",
        parameters: {},
        execute,
      },
    ]);
    const harness = makeToolApi({
      enableKnowledgeTools: true,
      dynamicBankId: false,
      bankId: "shared-bank",
      recallTypes: ["observation"],
      preferObservations: true,
      recallBudget: "high",
    });

    plugin(harness.api);
    const tools = harness.factory({});
    await tools[0].execute("call-1", {
      query: "raw facts",
      fact_types: ["world"],
      prefer_observations: false,
      min_scores: { final: 0.5 },
      budget: "low",
    });

    expect(execute).toHaveBeenCalledWith({
      query: "raw facts",
      fact_types: ["world"],
      prefer_observations: false,
      min_scores: { final: 0.5 },
      budget: "low",
    });
  });
});
