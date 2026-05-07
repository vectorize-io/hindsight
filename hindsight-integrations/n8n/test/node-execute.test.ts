import { describe, expect, it, vi, beforeEach } from "vitest";
import type { IExecuteFunctions, INodeExecutionData } from "n8n-workflow";

import { Hindsight } from "../nodes/Hindsight/Hindsight.node";

const API_URL = "https://api.example.com";

function createMockExecuteFunctions(
  params: Record<string, unknown>,
  options: {
    apiUrl?: string;
    apiKey?: string;
    requestImpl?: (...args: unknown[]) => unknown;
  } = {}
): IExecuteFunctions {
  const httpRequestWithAuthentication = vi.fn(options.requestImpl ?? (() => ({})));
  const fns = {
    getInputData: () => [{ json: {} }] as INodeExecutionData[],
    getCredentials: vi.fn().mockResolvedValue({
      apiUrl: options.apiUrl ?? API_URL,
      apiKey: options.apiKey ?? "hsk_test123",
    }),
    getNodeParameter: vi
      .fn()
      .mockImplementation((name: string, _index: number, fallback?: unknown) => {
        return params[name] ?? fallback;
      }),
    getNode: vi.fn().mockReturnValue({ name: "Hindsight" }),
    continueOnFail: vi.fn().mockReturnValue(false),
    helpers: {
      httpRequestWithAuthentication,
    },
  };
  return fns as unknown as IExecuteFunctions;
}

function getRequestMock(fns: IExecuteFunctions): ReturnType<typeof vi.fn> {
  return (
    fns as unknown as { helpers: { httpRequestWithAuthentication: ReturnType<typeof vi.fn> } }
  ).helpers.httpRequestWithAuthentication;
}

describe("Hindsight node execute()", () => {
  const node = new Hindsight();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls retain endpoint with correct URL and body", async () => {
    const mockFns = createMockExecuteFunctions(
      {
        operation: "retain",
        bankId: "bank-1",
        content: "User prefers dark mode",
        retainTags: "pref,ui",
      },
      {
        requestImpl: () => ({ operation_id: "op-1", status: "accepted" }),
      }
    );

    const result = await node.execute.call(mockFns);

    const req = getRequestMock(mockFns);
    expect(req).toHaveBeenCalledTimes(1);
    expect(req).toHaveBeenCalledWith("hindsightApi", {
      method: "POST",
      url: `${API_URL}/v1/default/banks/bank-1/memories`,
      body: { items: [{ content: "User prefers dark mode", tags: ["pref", "ui"] }] },
      json: true,
    });
    expect(result[0][0].json).toEqual({ operation_id: "op-1", status: "accepted" });
  });

  it("retain omits tags from item when tags is empty", async () => {
    const mockFns = createMockExecuteFunctions(
      {
        operation: "retain",
        bankId: "bank-1",
        content: "Hello world",
        retainTags: "",
      },
      {
        requestImpl: () => ({ operation_id: "op-2", status: "accepted" }),
      }
    );

    await node.execute.call(mockFns);

    const req = getRequestMock(mockFns);
    expect(req).toHaveBeenCalledWith("hindsightApi", {
      method: "POST",
      url: `${API_URL}/v1/default/banks/bank-1/memories`,
      body: { items: [{ content: "Hello world" }] },
      json: true,
    });
  });

  it("calls recall endpoint with correct URL and body", async () => {
    const mockFns = createMockExecuteFunctions(
      {
        operation: "recall",
        bankId: "bank-1",
        recallQuery: "what are user preferences?",
        recallBudget: "high",
        recallMaxTokens: 2048,
        recallTags: "pref",
      },
      {
        requestImpl: () => ({
          results: [{ text: "User prefers dark mode", score: 0.95 }],
        }),
      }
    );

    const result = await node.execute.call(mockFns);

    const req = getRequestMock(mockFns);
    expect(req).toHaveBeenCalledWith("hindsightApi", {
      method: "POST",
      url: `${API_URL}/v1/default/banks/bank-1/memories/recall`,
      body: {
        query: "what are user preferences?",
        max_tokens: 2048,
        budget: "high",
        tags: ["pref"],
      },
      json: true,
    });
    expect(result[0][0].json).toEqual({
      results: [{ text: "User prefers dark mode", score: 0.95 }],
    });
  });

  it("recall omits tags from body when tags filter is empty", async () => {
    const mockFns = createMockExecuteFunctions(
      {
        operation: "recall",
        bankId: "bank-1",
        recallQuery: "hello",
        recallBudget: "mid",
        recallMaxTokens: 4096,
        recallTags: "",
      },
      {
        requestImpl: () => ({ results: [] }),
      }
    );

    await node.execute.call(mockFns);

    const req = getRequestMock(mockFns);
    expect(req).toHaveBeenCalledWith("hindsightApi", {
      method: "POST",
      url: `${API_URL}/v1/default/banks/bank-1/memories/recall`,
      body: {
        query: "hello",
        max_tokens: 4096,
        budget: "mid",
      },
      json: true,
    });
  });

  it("calls reflect endpoint with correct URL and body", async () => {
    const mockFns = createMockExecuteFunctions(
      {
        operation: "reflect",
        bankId: "bank-1",
        reflectQuery: "summarize user preferences",
        reflectBudget: "low",
      },
      {
        requestImpl: () => ({
          text: "The user prefers dark mode and minimal UI.",
          citations: [],
        }),
      }
    );

    const result = await node.execute.call(mockFns);

    const req = getRequestMock(mockFns);
    expect(req).toHaveBeenCalledWith("hindsightApi", {
      method: "POST",
      url: `${API_URL}/v1/default/banks/bank-1/reflect`,
      body: {
        query: "summarize user preferences",
        budget: "low",
      },
      json: true,
    });
    expect(result[0][0].json).toEqual({
      text: "The user prefers dark mode and minimal UI.",
      citations: [],
    });
  });

  it("throws NodeOperationError when bankId is empty", async () => {
    const mockFns = createMockExecuteFunctions({
      operation: "retain",
      bankId: "",
      content: "test",
      retainTags: "",
    });

    await expect(node.execute.call(mockFns)).rejects.toThrow("bankId is required");
  });

  it("returns error json when continueOnFail is true and request throws", async () => {
    const mockFns = createMockExecuteFunctions(
      {
        operation: "retain",
        bankId: "bank-1",
        content: "test",
        retainTags: "",
      },
      {
        requestImpl: () => {
          throw new Error("Network error");
        },
      }
    );
    (mockFns.continueOnFail as ReturnType<typeof vi.fn>).mockReturnValue(true);

    const result = await node.execute.call(mockFns);

    expect(result[0][0].json).toEqual({ error: "Network error" });
  });

  it("strips trailing slash from apiUrl when building request URL", async () => {
    const mockFns = createMockExecuteFunctions(
      {
        operation: "retain",
        bankId: "bank-1",
        content: "test",
        retainTags: "",
      },
      {
        apiUrl: "http://localhost:8888/",
        apiKey: "",
        requestImpl: () => ({ operation_id: "op-3", status: "accepted" }),
      }
    );

    await node.execute.call(mockFns);

    const req = getRequestMock(mockFns);
    expect(req).toHaveBeenCalledWith(
      "hindsightApi",
      expect.objectContaining({
        url: "http://localhost:8888/v1/default/banks/bank-1/memories",
      })
    );
  });
});
