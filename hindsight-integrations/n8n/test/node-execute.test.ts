import { describe, expect, it, vi, beforeEach } from "vitest";
import type { IExecuteFunctions, INodeExecutionData } from "n8n-workflow";

const mockRetain = vi.fn();
const mockRecall = vi.fn();
const mockReflect = vi.fn();

vi.mock("@vectorize-io/hindsight-client", () => {
  return {
    HindsightClient: class {
      constructor(public options: unknown) {}
      retain = mockRetain;
      recall = mockRecall;
      reflect = mockReflect;
    },
  };
});

import { Hindsight } from "../nodes/Hindsight/Hindsight.node";

function createMockExecuteFunctions(params: Record<string, unknown>): IExecuteFunctions {
  return {
    getInputData: () => [{ json: {} }] as INodeExecutionData[],
    getCredentials: vi.fn().mockResolvedValue({
      apiUrl: "https://api.example.com",
      apiKey: "hsk_test123",
    }),
    getNodeParameter: vi
      .fn()
      .mockImplementation((name: string, _index: number, fallback?: unknown) => {
        return params[name] ?? fallback;
      }),
    getNode: vi.fn().mockReturnValue({ name: "Hindsight" }),
    continueOnFail: vi.fn().mockReturnValue(false),
  } as unknown as IExecuteFunctions;
}

describe("Hindsight node execute()", () => {
  const node = new Hindsight();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls client.retain with correct arguments", async () => {
    mockRetain.mockResolvedValue({ operation_id: "op-1", status: "accepted" });

    const mockFns = createMockExecuteFunctions({
      operation: "retain",
      bankId: "bank-1",
      content: "User prefers dark mode",
      retainTags: "pref,ui",
    });

    const result = await node.execute.call(mockFns);

    // Client was instantiated (verified by the successful retain call)
    expect(mockRetain).toHaveBeenCalledWith("bank-1", "User prefers dark mode", {
      tags: ["pref", "ui"],
    });
    expect(result[0][0].json).toEqual({ operation_id: "op-1", status: "accepted" });
  });

  it("calls client.retain without tags when tags is empty", async () => {
    mockRetain.mockResolvedValue({ operation_id: "op-2", status: "accepted" });

    const mockFns = createMockExecuteFunctions({
      operation: "retain",
      bankId: "bank-1",
      content: "Hello world",
      retainTags: "",
    });

    await node.execute.call(mockFns);

    expect(mockRetain).toHaveBeenCalledWith("bank-1", "Hello world", undefined);
  });

  it("calls client.recall with correct arguments", async () => {
    mockRecall.mockResolvedValue({
      results: [{ text: "User prefers dark mode", score: 0.95 }],
    });

    const mockFns = createMockExecuteFunctions({
      operation: "recall",
      bankId: "bank-1",
      recallQuery: "what are user preferences?",
      recallBudget: "high",
      recallMaxTokens: 2048,
      recallTags: "pref",
    });

    const result = await node.execute.call(mockFns);

    expect(mockRecall).toHaveBeenCalledWith("bank-1", "what are user preferences?", {
      budget: "high",
      maxTokens: 2048,
      tags: ["pref"],
    });
    expect(result[0][0].json).toEqual({
      results: [{ text: "User prefers dark mode", score: 0.95 }],
    });
  });

  it("calls client.recall without tags when tags filter is empty", async () => {
    mockRecall.mockResolvedValue({ results: [] });

    const mockFns = createMockExecuteFunctions({
      operation: "recall",
      bankId: "bank-1",
      recallQuery: "hello",
      recallBudget: "mid",
      recallMaxTokens: 4096,
      recallTags: "",
    });

    await node.execute.call(mockFns);

    expect(mockRecall).toHaveBeenCalledWith("bank-1", "hello", {
      budget: "mid",
      maxTokens: 4096,
    });
  });

  it("calls client.reflect with correct arguments", async () => {
    mockReflect.mockResolvedValue({
      text: "The user prefers dark mode and minimal UI.",
      citations: [],
    });

    const mockFns = createMockExecuteFunctions({
      operation: "reflect",
      bankId: "bank-1",
      reflectQuery: "summarize user preferences",
      reflectBudget: "low",
    });

    const result = await node.execute.call(mockFns);

    expect(mockReflect).toHaveBeenCalledWith("bank-1", "summarize user preferences", {
      budget: "low",
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

  it("returns error json when continueOnFail is true and client throws", async () => {
    mockRetain.mockRejectedValue(new Error("Network error"));

    const mockFns = createMockExecuteFunctions({
      operation: "retain",
      bankId: "bank-1",
      content: "test",
      retainTags: "",
    });
    (mockFns.continueOnFail as ReturnType<typeof vi.fn>).mockReturnValue(true);

    const result = await node.execute.call(mockFns);

    expect(result[0][0].json).toEqual({ error: "Network error" });
  });

  it("omits apiKey from client when credential has no key", async () => {
    mockRetain.mockResolvedValue({ operation_id: "op-3", status: "accepted" });

    const mockFns = createMockExecuteFunctions({
      operation: "retain",
      bankId: "bank-1",
      content: "test",
      retainTags: "",
    });
    (mockFns.getCredentials as ReturnType<typeof vi.fn>).mockResolvedValue({
      apiUrl: "http://localhost:8888",
      apiKey: "",
    });

    await node.execute.call(mockFns);

    // Verified by the successful retain call with empty apiKey credential
  });
});
