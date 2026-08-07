import { afterEach, describe, it, expect, vi } from "vitest";
import { createTools } from "./tools.js";
import { makeConfig } from "./test-helpers.js";

const mockContext = {
  sessionID: "sess-1",
  messageID: "msg-1",
  agent: "default",
  directory: "/tmp",
  worktree: "/tmp",
  abort: new AbortController().signal,
  metadata: vi.fn(),
  ask: vi.fn(),
};

describe("createTools", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates all four tools", () => {
    const client = { retain: vi.fn(), recall: vi.fn(), reflect: vi.fn() } as any;
    const tools = createTools(client, "test-bank", makeConfig());

    expect(tools.hindsight_retain).toBeDefined();
    expect(tools.hindsight_operation_status).toBeDefined();
    expect(tools.hindsight_recall).toBeDefined();
    expect(tools.hindsight_reflect).toBeDefined();
  });

  it("all tools have description and execute", () => {
    const client = { retain: vi.fn(), recall: vi.fn(), reflect: vi.fn() } as any;
    const tools = createTools(client, "test-bank", makeConfig());

    for (const tool of Object.values(tools)) {
      expect(tool.description).toBeTruthy();
      expect(typeof tool.execute).toBe("function");
    }
  });

  describe("hindsight_retain", () => {
    it("calls client.retain with correct bank and content", async () => {
      const client = {
        retain: vi.fn().mockResolvedValue({ operation_id: "op-123" }),
        recall: vi.fn(),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      const result = await tools.hindsight_retain.execute(
        { content: "User likes TypeScript" },
        mockContext
      );

      expect(client.retain).toHaveBeenCalledWith("test-bank", "User likes TypeScript", {
        context: "opencode",
        tags: undefined,
        metadata: undefined,
        async: true,
      });
      expect(result).toBe(
        "Memory queued successfully. Operation ID: op-123. " +
          "Use hindsight_operation_status to poll for completion."
      );
    });

    it("passes optional context", async () => {
      const client = {
        retain: vi.fn().mockResolvedValue({ operation_id: "op-123" }),
        recall: vi.fn(),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await tools.hindsight_retain.execute(
        { content: "Fact", context: "from conversation" },
        mockContext
      );

      expect(client.retain).toHaveBeenCalledWith("test-bank", "Fact", {
        context: "from conversation",
        tags: undefined,
        metadata: undefined,
        async: true,
      });
    });

    it("includes tags and metadata from config", async () => {
      const client = {
        retain: vi.fn().mockResolvedValue({ operation_id: "op-123" }),
        recall: vi.fn(),
        reflect: vi.fn(),
      } as any;
      const config = makeConfig({
        retainTags: ["coding"],
        retainMetadata: { source: "opencode" },
      });
      const tools = createTools(client, "test-bank", config);

      await tools.hindsight_retain.execute({ content: "Fact" }, mockContext);

      expect(client.retain).toHaveBeenCalledWith("test-bank", "Fact", {
        context: "opencode",
        tags: ["coding"],
        metadata: { source: "opencode" },
        async: true,
      });
    });

    it("rejects an async response without an operation ID", async () => {
      const client = {
        retain: vi.fn().mockResolvedValue({ async: true }),
        recall: vi.fn(),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await expect(
        tools.hindsight_retain.execute({ content: "Fact" }, mockContext)
      ).rejects.toThrow("Async retain did not return an operation ID.");
    });
  });

  describe("hindsight_operation_status", () => {
    it("fetches operation status with authentication", async () => {
      const operation = {
        operation_id: "550e8400-e29b-41d4-a716-446655440000",
        status: "processing",
        error_message: null,
      };
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue(operation),
      });
      vi.stubGlobal("fetch", fetchMock);
      const client = { retain: vi.fn(), recall: vi.fn(), reflect: vi.fn() } as any;
      const config = makeConfig({
        hindsightApiUrl: "https://hindsight.example/",
        hindsightApiToken: "secret-token",
      });
      const tools = createTools(client, "test bank", config);

      const result = await tools.hindsight_operation_status.execute(
        { operationId: "550e8400-e29b-41d4-a716-446655440000" },
        mockContext
      );

      expect(fetchMock).toHaveBeenCalledWith(
        "https://hindsight.example/v1/default/banks/test%20bank/operations/550e8400-e29b-41d4-a716-446655440000",
        { headers: { Authorization: "Bearer secret-token" } }
      );
      expect(JSON.parse(result)).toEqual(operation);
    });

    it("returns the API not_found status", async () => {
      const operation = {
        operation_id: "550e8400-e29b-41d4-a716-446655440000",
        status: "not_found",
      };
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: true,
          json: vi.fn().mockResolvedValue(operation),
        })
      );
      const client = { retain: vi.fn(), recall: vi.fn(), reflect: vi.fn() } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      const result = await tools.hindsight_operation_status.execute(
        { operationId: operation.operation_id },
        mockContext
      );

      expect(JSON.parse(result)).toEqual(operation);
    });

    it("propagates operation status errors", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue({
          ok: false,
          status: 500,
          text: vi.fn().mockResolvedValue("internal error"),
        })
      );
      const client = { retain: vi.fn(), recall: vi.fn(), reflect: vi.fn() } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await expect(
        tools.hindsight_operation_status.execute(
          { operationId: "550e8400-e29b-41d4-a716-446655440000" },
          mockContext
        )
      ).rejects.toThrow("Operation status failed (500): internal error");
    });
  });

  describe("hindsight_recall", () => {
    it("calls client.recall and formats results", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn().mockResolvedValue({
          results: [{ text: "User likes Python", type: "world", mentioned_at: "2025-01-01" }],
        }),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      const result = await tools.hindsight_recall.execute(
        { query: "user preferences" },
        mockContext
      );

      expect(client.recall).toHaveBeenCalledWith("test-bank", "user preferences", {
        budget: "mid",
        maxTokens: 1024,
        types: ["world", "experience"],
      });
      expect(result).toContain("User likes Python");
      expect(result).toContain("[world]");
    });

    it("returns no-results message when empty", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn().mockResolvedValue({ results: [] }),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      const result = await tools.hindsight_recall.execute({ query: "unknown" }, mockContext);
      expect(result).toBe("No relevant memories found.");
    });

    it("uses config budget settings", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn().mockResolvedValue({ results: [] }),
        reflect: vi.fn(),
      } as any;
      const config = makeConfig({ recallBudget: "high", recallMaxTokens: 4096 });
      const tools = createTools(client, "test-bank", config);

      await tools.hindsight_recall.execute({ query: "test" }, mockContext);

      expect(client.recall).toHaveBeenCalledWith("test-bank", "test", {
        budget: "high",
        maxTokens: 4096,
        types: ["world", "experience"],
      });
    });
  });

  describe("hindsight_reflect", () => {
    it("calls client.reflect and returns text", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn(),
        reflect: vi.fn().mockResolvedValue({ text: "The user is a Python developer." }),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      const result = await tools.hindsight_reflect.execute(
        { query: "What do I know about this user?" },
        mockContext
      );

      expect(client.reflect).toHaveBeenCalledWith("test-bank", "What do I know about this user?", {
        context: undefined,
        budget: "mid",
      });
      expect(result).toBe("The user is a Python developer.");
    });

    it("returns fallback when no text", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn(),
        reflect: vi.fn().mockResolvedValue({ text: "" }),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      const result = await tools.hindsight_reflect.execute({ query: "something" }, mockContext);
      expect(result).toBe("No relevant information found to reflect on.");
    });

    it("passes context to reflect", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn(),
        reflect: vi.fn().mockResolvedValue({ text: "Answer" }),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await tools.hindsight_reflect.execute(
        { query: "Q", context: "We are building an app" },
        mockContext
      );

      expect(client.reflect).toHaveBeenCalledWith("test-bank", "Q", {
        context: "We are building an app",
        budget: "mid",
      });
    });
  });

  describe("error propagation", () => {
    it("propagates retain errors", async () => {
      const client = {
        retain: vi.fn().mockRejectedValue(new Error("Network error")),
        recall: vi.fn(),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await expect(
        tools.hindsight_retain.execute({ content: "test" }, mockContext)
      ).rejects.toThrow("Network error");
    });

    it("propagates recall errors", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn().mockRejectedValue(new Error("Timeout")),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await expect(tools.hindsight_recall.execute({ query: "test" }, mockContext)).rejects.toThrow(
        "Timeout"
      );
    });

    it("propagates reflect errors", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn(),
        reflect: vi.fn().mockRejectedValue(new Error("Server error")),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await expect(tools.hindsight_reflect.execute({ query: "test" }, mockContext)).rejects.toThrow(
        "Server error"
      );
    });
  });

  describe("bank mission setup", () => {
    it("calls ensureBankMission before retain when missionsSet provided", async () => {
      const client = {
        retain: vi.fn().mockResolvedValue({ operation_id: "op-123" }),
        recall: vi.fn(),
        reflect: vi.fn(),
        createBank: vi.fn().mockResolvedValue({}),
      } as any;
      const missionsSet = new Set<string>();
      const config = makeConfig({ bankMission: "Extract technical decisions" });
      const tools = createTools(client, "test-bank", config, missionsSet);

      await tools.hindsight_retain.execute({ content: "fact" }, mockContext);

      expect(client.createBank).toHaveBeenCalledWith("test-bank", {
        reflectMission: "Extract technical decisions",
        retainMission: undefined,
      });
      expect(missionsSet.has("test-bank")).toBe(true);
      expect(client.retain).toHaveBeenCalled();
    });

    it("calls ensureBankMission before reflect when missionsSet provided", async () => {
      const client = {
        retain: vi.fn(),
        recall: vi.fn(),
        reflect: vi.fn().mockResolvedValue({ text: "answer" }),
        createBank: vi.fn().mockResolvedValue({}),
      } as any;
      const missionsSet = new Set<string>();
      const config = makeConfig({ bankMission: "Synthesize project context" });
      const tools = createTools(client, "test-bank", config, missionsSet);

      await tools.hindsight_reflect.execute({ query: "summary" }, mockContext);

      expect(client.createBank).toHaveBeenCalled();
      expect(client.reflect).toHaveBeenCalled();
    });

    it("skips mission setup when missionsSet not provided (backward compat)", async () => {
      const client = {
        retain: vi.fn().mockResolvedValue({ operation_id: "op-123" }),
        recall: vi.fn(),
        reflect: vi.fn(),
      } as any;
      const tools = createTools(client, "test-bank", makeConfig());

      await tools.hindsight_retain.execute({ content: "fact" }, mockContext);

      expect(client.retain).toHaveBeenCalled();
      // No createBank call since missionsSet wasn't passed
    });
  });

  it("always uses constructor bankId", async () => {
    const client = {
      retain: vi.fn().mockResolvedValue({ operation_id: "op-123" }),
      recall: vi.fn().mockResolvedValue({ results: [] }),
      reflect: vi.fn().mockResolvedValue({ text: "ok" }),
    } as any;
    const tools = createTools(client, "fixed-bank", makeConfig());

    await tools.hindsight_retain.execute({ content: "x" }, mockContext);
    await tools.hindsight_recall.execute({ query: "x" }, mockContext);
    await tools.hindsight_reflect.execute({ query: "x" }, mockContext);

    expect(client.retain.mock.calls[0][0]).toBe("fixed-bank");
    expect(client.recall.mock.calls[0][0]).toBe("fixed-bank");
    expect(client.reflect.mock.calls[0][0]).toBe("fixed-bank");
  });
});
