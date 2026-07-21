import type { AgentMemoryProvider } from "@framers/agentos";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createHindsightMemory,
  type HindsightClient,
  type RecallResponse,
  type RetainResponse,
} from "../src/index.js";

function mockClient(): HindsightClient {
  return {
    recall: vi.fn(
      async (): Promise<RecallResponse> => ({
        results: [
          { id: "1", text: "User prefers dark mode", type: "world" },
          { id: "2", text: "User lives in Berlin", type: "experience" },
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

/**
 * Minimal re-implementation of AgentOS's own `applyMemoryProvider` wiring:
 * call `getContext` before generation and inject the returned block as a
 * system message, then call `observe` for the user turn and assistant reply
 * after. Lets the test exercise the provider exactly the way AgentOS does.
 */
async function runTurn(
  provider: AgentMemoryProvider,
  userText: string,
  assistantText: string,
  tokenBudget = 2000
): Promise<{ role: string; content: string }[]> {
  const messages: { role: string; content: string }[] = [
    { role: "system", content: "You are Ada." },
    { role: "user", content: userText },
  ];
  if (provider.getContext) {
    const ctx = await provider.getContext(userText, { tokenBudget });
    if (ctx?.contextText) {
      messages.splice(1, 0, { role: "system", content: ctx.contextText });
    }
  }
  if (provider.observe) {
    await provider.observe("user", userText);
    await provider.observe("assistant", assistantText);
  }
  return messages;
}

describe("createHindsightMemory", () => {
  let client: HindsightClient;

  beforeEach(() => {
    client = mockClient();
  });

  it("returns a provider exposing getContext and observe by default", () => {
    const provider = createHindsightMemory({ client });
    expect(typeof provider.getContext).toBe("function");
    expect(typeof provider.observe).toBe("function");
  });

  it("can disable recall or retain", () => {
    const provider = createHindsightMemory({
      client,
      recall: { enabled: false },
      retain: { enabled: false },
    });
    expect(provider.getContext).toBeUndefined();
    expect(provider.observe).toBeUndefined();
  });

  describe("getContext (recall)", () => {
    it("recalls memories and formats them as a context block", async () => {
      const provider = createHindsightMemory({ client });
      const ctx = await provider.getContext!("What theme do I like?", { tokenBudget: 500 });
      expect(client.recall).toHaveBeenCalledWith("default", "What theme do I like?", {
        types: undefined,
        maxTokens: 500,
        budget: undefined,
        includeEntities: undefined,
      });
      expect(ctx?.contextText).toContain("User prefers dark mode");
      expect(ctx?.contextText).toContain("User lives in Berlin");
    });

    it("uses the configured bank", async () => {
      const provider = createHindsightMemory({ client, bank: "team-bank" });
      await provider.getContext!("hi");
      expect(client.recall).toHaveBeenCalledWith("team-bank", "hi", expect.any(Object));
    });

    it("prefers a configured maxTokens over the hook tokenBudget", async () => {
      const provider = createHindsightMemory({ client, recall: { maxTokens: 128 } });
      await provider.getContext!("q", { tokenBudget: 4096 });
      expect(client.recall).toHaveBeenCalledWith(
        "default",
        "q",
        expect.objectContaining({ maxTokens: 128 })
      );
    });

    it("labels fact types when labelTypes is set", async () => {
      const provider = createHindsightMemory({ client, recall: { labelTypes: true } });
      const ctx = await provider.getContext!("q");
      expect(ctx?.contextText).toContain("[world] User prefers dark mode");
      expect(ctx?.contextText).toContain("[experience] User lives in Berlin");
    });

    it("returns null for an empty query without calling recall", async () => {
      const provider = createHindsightMemory({ client });
      const ctx = await provider.getContext!("   ");
      expect(client.recall).not.toHaveBeenCalled();
      expect(ctx).toBeNull();
    });

    it("returns null when recall yields no memories", async () => {
      (client.recall as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ results: [] });
      const provider = createHindsightMemory({ client });
      expect(await provider.getContext!("q")).toBeNull();
    });

    it("never throws when recall fails", async () => {
      (client.recall as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("down"));
      const provider = createHindsightMemory({ client });
      expect(await provider.getContext!("hello")).toBeNull();
    });

    it("passes recall options through to the client", async () => {
      const provider = createHindsightMemory({
        client,
        recall: { budget: "high", types: ["world"], includeEntities: true, maxTokens: 500 },
      });
      await provider.getContext!("q", { tokenBudget: 4096 });
      expect(client.recall).toHaveBeenCalledWith("default", "q", {
        budget: "high",
        types: ["world"],
        includeEntities: true,
        maxTokens: 500,
      });
    });
  });

  describe("observe (retain)", () => {
    it("retains the user turn", async () => {
      const provider = createHindsightMemory({ client, retain: { async: false } });
      await provider.observe!("user", "Remember I like dark mode");
      expect(client.retain).toHaveBeenCalledWith("default", "Remember I like dark mode", {
        async: false,
        tags: undefined,
        metadata: undefined,
      });
    });

    it("skips the assistant reply by default", async () => {
      const provider = createHindsightMemory({ client, retain: { async: false } });
      await provider.observe!("assistant", "my reply");
      expect(client.retain).not.toHaveBeenCalled();
    });

    it("retains assistant replies when includeAgentMessages is set", async () => {
      const provider = createHindsightMemory({
        client,
        retain: { async: false, includeAgentMessages: true },
      });
      await provider.observe!("assistant", "agent answer");
      expect(client.retain).toHaveBeenCalledWith("default", "agent answer", expect.any(Object));
    });

    it("ignores empty content", async () => {
      const provider = createHindsightMemory({ client, retain: { async: false } });
      await provider.observe!("user", "   ");
      expect(client.retain).not.toHaveBeenCalled();
    });

    it("passes tags and metadata through", async () => {
      const provider = createHindsightMemory({
        client,
        retain: { async: false, tags: ["source:agentos"], metadata: { env: "prod" } },
      });
      await provider.observe!("user", "hi");
      expect(client.retain).toHaveBeenCalledWith("default", "hi", {
        async: false,
        tags: ["source:agentos"],
        metadata: { env: "prod" },
      });
    });

    it("does not reject when retain fails (async mode)", async () => {
      (client.retain as ReturnType<typeof vi.fn>).mockRejectedValue(new Error("boom"));
      const provider = createHindsightMemory({ client });
      await expect(provider.observe!("user", "hi")).resolves.toBeUndefined();
    });
  });

  describe("AgentOS turn wiring", () => {
    it("injects recalled memory into the prompt and retains the user turn", async () => {
      const provider = createHindsightMemory({ client, bank: "ada", retain: { async: false } });
      const messages = await runTurn(provider, "What theme do I like?", "You like dark mode.");

      // Recall block injected after the leading system message, before the user turn.
      expect(messages[1].role).toBe("system");
      expect(messages[1].content).toContain("User prefers dark mode");
      expect(messages[0].content).toBe("You are Ada.");

      expect(client.recall).toHaveBeenCalledWith("ada", "What theme do I like?", expect.any(Object));
      // User turn retained; assistant reply skipped by default.
      expect(client.retain).toHaveBeenCalledTimes(1);
      expect(client.retain).toHaveBeenCalledWith("ada", "What theme do I like?", expect.any(Object));
    });

    it("retains both turns when includeAgentMessages is set", async () => {
      const provider = createHindsightMemory({
        client,
        bank: "ada",
        retain: { async: false, includeAgentMessages: true },
      });
      await runTurn(provider, "hi", "hello there");
      expect(client.retain).toHaveBeenCalledWith("ada", "hi", expect.any(Object));
      expect(client.retain).toHaveBeenCalledWith("ada", "hello there", expect.any(Object));
    });
  });
});
