import { describe, it, expect, vi, beforeEach } from "vitest";
import { withHindsightChat } from "./wrapper.js";
import type { HindsightClient, ChatThread, ChatMessage, HindsightChatContext } from "./types.js";

// --- Mocks ---

function mockClient(overrides?: Partial<HindsightClient>): HindsightClient {
  return {
    retain: vi.fn().mockResolvedValue({
      success: true,
      bank_id: "test-bank",
      items_count: 1,
      async: false,
    }),
    recall: vi.fn().mockResolvedValue({
      results: [{ id: "mem-1", text: "User likes TypeScript", type: "experience" }],
      entities: {
        "ent-1": {
          entity_id: "ent-1",
          canonical_name: "User",
          observations: [{ text: "Prefers dark mode" }],
        },
      },
    }),
    reflect: vi.fn().mockResolvedValue({
      text: "User is a TypeScript developer",
      based_on: [],
    }),
    ...overrides,
  };
}

function mockThread(): ChatThread {
  return {
    post: vi.fn().mockResolvedValue(undefined),
    subscribe: vi.fn().mockResolvedValue(undefined),
    unsubscribe: vi.fn().mockResolvedValue(undefined),
    isSubscribed: vi.fn().mockResolvedValue(false),
    startTyping: vi.fn().mockResolvedValue(undefined),
    state: Promise.resolve(null),
    setState: vi.fn().mockResolvedValue(undefined),
  };
}

function mockMessage(overrides?: Partial<ChatMessage>): ChatMessage {
  return {
    author: { userId: "user-123", name: "Alice", isMe: false },
    text: "What do you know about me?",
    threadId: "thread-1",
    ...overrides,
  };
}

describe("withHindsightChat", () => {
  let client: HindsightClient;
  let thread: ChatThread;
  let message: ChatMessage;

  beforeEach(() => {
    client = mockClient();
    thread = mockThread();
    message = mockMessage();
  });

  describe("bankId resolution", () => {
    it("uses static bankId", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "static-bank" }, handler);

      await wrapped(thread, message);

      expect(client.recall).toHaveBeenCalledWith("static-bank", message.text, expect.any(Object));
      expect(handler).toHaveBeenCalledWith(
        thread,
        message,
        expect.objectContaining({ bankId: "static-bank" })
      );
    });

    it("uses dynamic bankId from message", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        { client, bankId: (msg) => `bank-${msg.author.userId}` },
        handler
      );

      await wrapped(thread, message);

      expect(client.recall).toHaveBeenCalledWith("bank-user-123", message.text, expect.any(Object));
      expect(handler).toHaveBeenCalledWith(
        thread,
        message,
        expect.objectContaining({ bankId: "bank-user-123" })
      );
    });
  });

  describe("auto-recall", () => {
    it("recalls by default", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      expect(client.recall).toHaveBeenCalledOnce();
      expect(client.recall).toHaveBeenCalledWith("bank", message.text, {
        budget: "mid",
        maxTokens: undefined,
        types: undefined,
        includeEntities: true,
      });

      const ctx: HindsightChatContext = handler.mock.calls[0][2];
      expect(ctx.memories).toHaveLength(1);
      expect(ctx.memories[0].text).toBe("User likes TypeScript");
      expect(ctx.entities).not.toBeNull();
    });

    it("can be disabled", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        { client, bankId: "bank", recall: { enabled: false } },
        handler
      );

      await wrapped(thread, message);

      expect(client.recall).not.toHaveBeenCalled();

      const ctx: HindsightChatContext = handler.mock.calls[0][2];
      expect(ctx.memories).toEqual([]);
      expect(ctx.entities).toBeNull();
    });

    it("passes recall options through", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        {
          client,
          bankId: "bank",
          recall: {
            budget: "high",
            maxTokens: 500,
            types: ["experience"],
            includeEntities: false,
          },
        },
        handler
      );

      await wrapped(thread, message);

      expect(client.recall).toHaveBeenCalledWith("bank", message.text, {
        budget: "high",
        maxTokens: 500,
        types: ["experience"],
        includeEntities: false,
      });
    });

    it("skips recall for empty message text", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, mockMessage({ text: "" }));

      expect(client.recall).not.toHaveBeenCalled();
    });

    it("handles recall errors gracefully", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      client = mockClient({
        recall: vi.fn().mockRejectedValue(new Error("Network error")),
      });
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      expect(warnSpy).toHaveBeenCalledWith(
        "[hindsight-chat] Auto-recall failed:",
        expect.any(Error)
      );
      // Handler still runs with empty memories
      const ctx: HindsightChatContext = handler.mock.calls[0][2];
      expect(ctx.memories).toEqual([]);
      warnSpy.mockRestore();
    });
  });

  describe("auto-retain", () => {
    it("does not retain by default", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      expect(client.retain).not.toHaveBeenCalled();
    });

    it("retains when enabled", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        { client, bankId: "bank", retain: { enabled: true } },
        handler
      );

      await wrapped(thread, message);

      expect(client.retain).toHaveBeenCalledWith("bank", message.text, {
        tags: undefined,
        metadata: undefined,
        async: true,
      });
    });

    it("passes retain options through", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        {
          client,
          bankId: "bank",
          retain: {
            enabled: true,
            tags: ["slack"],
            metadata: { source: "chat" },
            async: false,
          },
        },
        handler
      );

      await wrapped(thread, message);

      expect(client.retain).toHaveBeenCalledWith("bank", message.text, {
        tags: ["slack"],
        metadata: { source: "chat" },
        async: false,
      });
    });

    it("skips retain for bot messages (isMe)", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        { client, bankId: "bank", retain: { enabled: true } },
        handler
      );

      await wrapped(thread, mockMessage({ author: { userId: "bot", isMe: true } }));

      expect(client.retain).not.toHaveBeenCalled();
    });

    it("skips retain for empty text", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        { client, bankId: "bank", retain: { enabled: true } },
        handler
      );

      await wrapped(thread, mockMessage({ text: "" }));

      expect(client.retain).not.toHaveBeenCalled();
    });

    it("handles retain errors gracefully", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      client = mockClient({
        retain: vi.fn().mockRejectedValue(new Error("Retain failed")),
      });
      const handler = vi.fn();
      const wrapped = withHindsightChat(
        { client, bankId: "bank", retain: { enabled: true } },
        handler
      );

      // Should not throw
      await wrapped(thread, message);

      expect(warnSpy).toHaveBeenCalledWith(
        "[hindsight-chat] Auto-retain failed:",
        expect.any(Error)
      );
      // Handler still runs
      expect(handler).toHaveBeenCalled();
      warnSpy.mockRestore();
    });
  });

  describe("context methods", () => {
    it("memoriesAsSystemPrompt() formats recalled memories", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      const ctx: HindsightChatContext = handler.mock.calls[0][2];
      const prompt = ctx.memoriesAsSystemPrompt();
      expect(prompt).toContain("<memories>");
      expect(prompt).toContain("User likes TypeScript");
      expect(prompt).toContain("<entity_observations>");
    });

    it("ctx.retain() delegates to client", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      const ctx: HindsightChatContext = handler.mock.calls[0][2];
      await ctx.retain("New memory content", { tags: ["test"] });

      expect(client.retain).toHaveBeenCalledWith("bank", "New memory content", {
        tags: ["test"],
      });
    });

    it("ctx.recall() delegates to client", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      const ctx: HindsightChatContext = handler.mock.calls[0][2];
      await ctx.recall("search query", { budget: "high" });

      // Second call (first was auto-recall)
      expect(client.recall).toHaveBeenCalledTimes(2);
      expect(client.recall).toHaveBeenLastCalledWith("bank", "search query", {
        budget: "high",
      });
    });

    it("ctx.reflect() delegates to client", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      const ctx: HindsightChatContext = handler.mock.calls[0][2];
      await ctx.reflect("What does the user prefer?");

      expect(client.reflect).toHaveBeenCalledWith("bank", "What does the user prefer?", undefined);
    });
  });

  describe("handler invocation", () => {
    it("passes thread and message through", async () => {
      const handler = vi.fn();
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      expect(handler).toHaveBeenCalledWith(thread, message, expect.any(Object));
    });

    it("awaits async handlers", async () => {
      let completed = false;
      const handler = async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        completed = true;
      };
      const wrapped = withHindsightChat({ client, bankId: "bank" }, handler);

      await wrapped(thread, message);

      expect(completed).toBe(true);
    });
  });
});
