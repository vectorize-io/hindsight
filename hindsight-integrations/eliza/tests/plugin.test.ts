import type { IAgentRuntime, Memory, Provider, Evaluator } from "@elizaos/core";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createHindsightPlugin,
  type HindsightClient,
  type RecallResponse,
  type RetainResponse,
} from "../src/index.js";

const AGENT_ID = "00000000-0000-0000-0000-0000000000a9" as const;
const USER_ID = "00000000-0000-0000-0000-0000000000u5" as const;

function mockClient(): HindsightClient {
  return {
    recall: vi.fn(
      async (): Promise<RecallResponse> => ({
        results: [
          { id: "1", text: "User prefers dark mode" },
          { id: "2", text: "User lives in Berlin" },
        ],
      })
    ),
    retain: vi.fn(
      async (bankId: string): Promise<RetainResponse> => ({
        success: true,
        bank_id: bankId,
        items_count: 1,
        async: true,
      })
    ),
  };
}

function userMessage(text: string): Memory {
  return {
    entityId: USER_ID,
    agentId: AGENT_ID,
    roomId: "00000000-0000-0000-0000-0000000000r1",
    content: { text },
  } as Memory;
}

const runtime = { agentId: AGENT_ID } as IAgentRuntime;

function getProvider(plugin: ReturnType<typeof createHindsightPlugin>): Provider {
  const provider = plugin.providers?.[0];
  if (!provider) throw new Error("provider missing");
  return provider;
}

function getEvaluator(plugin: ReturnType<typeof createHindsightPlugin>): Evaluator {
  const evaluator = plugin.evaluators?.[0];
  if (!evaluator) throw new Error("evaluator missing");
  return evaluator;
}

describe("createHindsightPlugin", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = mockClient();
  });

  it("registers a recall provider and a retain evaluator by default", () => {
    const plugin = createHindsightPlugin({ client });
    expect(plugin.providers).toHaveLength(1);
    expect(plugin.evaluators).toHaveLength(1);
    expect(getProvider(plugin).name).toBe("HINDSIGHT_MEMORY");
    expect(getEvaluator(plugin).name).toBe("HINDSIGHT_RETAIN");
  });

  it("can disable recall or retain", () => {
    const plugin = createHindsightPlugin({
      client,
      recall: { enabled: false },
      retain: { enabled: false },
    });
    expect(plugin.providers).toHaveLength(0);
    expect(plugin.evaluators).toHaveLength(0);
  });

  describe("recall provider", () => {
    it("recalls memories and formats them for the prompt", async () => {
      const provider = getProvider(createHindsightPlugin({ client }));
      const result = await provider.get(runtime, userMessage("What theme do I like?"), {} as never);

      expect(client.recall).toHaveBeenCalledWith(
        USER_ID,
        "What theme do I like?",
        expect.any(Object)
      );
      expect(result.text).toContain("User prefers dark mode");
      expect(result.text).toContain("User lives in Berlin");
      expect(result.values?.hindsightMemoryCount).toBe(2);
    });

    it("defaults the bank to the message entityId but honours an override", async () => {
      const provider = getProvider(createHindsightPlugin({ client, bank: "team-bank" }));
      await provider.get(runtime, userMessage("hi"), {} as never);
      expect(client.recall).toHaveBeenCalledWith("team-bank", "hi", expect.any(Object));
    });

    it("returns empty text for an empty message without calling recall", async () => {
      const provider = getProvider(createHindsightPlugin({ client }));
      const result = await provider.get(runtime, userMessage("   "), {} as never);
      expect(client.recall).not.toHaveBeenCalled();
      expect(result.text).toBe("");
    });

    it("never throws when recall fails", async () => {
      (client.recall as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("down"));
      const provider = getProvider(createHindsightPlugin({ client }));
      const result = await provider.get(runtime, userMessage("hello"), {} as never);
      expect(result.text).toBe("");
      expect(result.data?.hindsightError).toBe("down");
    });

    it("passes recall options through to the client", async () => {
      const provider = getProvider(
        createHindsightPlugin({
          client,
          recall: { budget: "high", types: ["world"], includeEntities: true, maxTokens: 500 },
        })
      );
      await provider.get(runtime, userMessage("q"), {} as never);
      expect(client.recall).toHaveBeenCalledWith(USER_ID, "q", {
        budget: "high",
        types: ["world"],
        includeEntities: true,
        maxTokens: 500,
      });
    });
  });

  describe("retain evaluator", () => {
    it("retains the incoming user message", async () => {
      const evaluator = getEvaluator(createHindsightPlugin({ client, retain: { async: false } }));
      const message = userMessage("Remember I like dark mode");
      await evaluator.handler(runtime, message, undefined, undefined, undefined, undefined);
      expect(client.retain).toHaveBeenCalledWith(USER_ID, "Remember I like dark mode", {
        async: false,
        tags: undefined,
        metadata: undefined,
      });
    });

    it("validates only messages that have text", async () => {
      const evaluator = getEvaluator(createHindsightPlugin({ client }));
      expect(await evaluator.validate(runtime, userMessage("hi"))).toBe(true);
      expect(await evaluator.validate(runtime, userMessage("  "))).toBe(false);
    });

    it("skips the agent's own message by default", async () => {
      const evaluator = getEvaluator(createHindsightPlugin({ client, retain: { async: false } }));
      const agentMsg = { ...userMessage("my reply"), entityId: AGENT_ID } as Memory;
      await evaluator.handler(runtime, agentMsg, undefined, undefined, undefined, undefined);
      expect(client.retain).not.toHaveBeenCalled();
    });

    it("retains agent replies when includeAgentMessages is set", async () => {
      const evaluator = getEvaluator(
        createHindsightPlugin({ client, retain: { async: false, includeAgentMessages: true } })
      );
      const message = userMessage("hello");
      const responses = [{ ...userMessage("agent answer"), entityId: AGENT_ID } as Memory];
      await evaluator.handler(runtime, message, undefined, undefined, undefined, responses);
      expect(client.retain).toHaveBeenCalledWith(USER_ID, "hello", expect.any(Object));
      expect(client.retain).toHaveBeenCalledWith(USER_ID, "agent answer", expect.any(Object));
    });

    it("does not reject the turn when retain fails (async mode)", async () => {
      (client.retain as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("boom"));
      const evaluator = getEvaluator(createHindsightPlugin({ client }));
      await expect(
        evaluator.handler(runtime, userMessage("hi"), undefined, undefined, undefined, undefined)
      ).resolves.toBeUndefined();
    });
  });
});
