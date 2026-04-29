import { describe, it, expect, vi, beforeEach } from "vitest";
import { createKnowledgeTools, TOOL_NAMES } from "../src/index.js";
import type { KnowledgeTool } from "../src/index.js";

// Mock fetch globally — the SDK uses @vectorize-io/hindsight-client which calls fetch.
// The generated client passes a Request object (not a plain URL string).
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockResponse(data: unknown, status = 200) {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
    headers: new Headers({ "content-type": "application/json" }),
    clone: function () {
      return this;
    },
  });
}

/** Extract the URL string from the first argument (Request object or string). */
function getUrl(call: any[]): string {
  const arg = call[0];
  return typeof arg === "string" ? arg : (arg?.url ?? String(arg));
}

/** Extract request init/options from the fetch call. */
function getOpts(call: any[]): any {
  const arg = call[0];
  if (typeof arg === "object" && arg?.method) return arg;
  return call[1] ?? {};
}

/** Extract the body as a parsed object from a fetch call. */
async function getBody(call: any[]): Promise<any> {
  const opts = getOpts(call);
  if (typeof opts.body === "string") return JSON.parse(opts.body);
  // Request objects have a .json() or .text() method
  if (typeof opts.json === "function") return opts.json();
  if (typeof opts.text === "function") return JSON.parse(await opts.text());
  return undefined;
}

describe("createKnowledgeTools", () => {
  let tools: KnowledgeTool[];

  beforeEach(() => {
    mockFetch.mockReset();
    tools = createKnowledgeTools({
      apiUrl: "http://localhost:9077",
      apiToken: "test-token",
      bankId: "test-bank",
    });
  });

  it("returns all 7 tools", () => {
    expect(tools).toHaveLength(7);
    const names = tools.map((t) => t.name);
    expect(names).toEqual([...TOOL_NAMES]);
  });

  it("each tool has required fields", () => {
    for (const tool of tools) {
      expect(tool.name).toBeTruthy();
      expect(tool.label).toBeTruthy();
      expect(tool.description).toBeTruthy();
      expect(tool.parameters).toBeDefined();
      expect(typeof tool.execute).toBe("function");
    }
  });

  it("list_pages calls the correct endpoint", async () => {
    mockFetch.mockReturnValueOnce(mockResponse({ models: [] }));

    const tool = tools.find((t) => t.name === "agent_knowledge_list_pages")!;
    const result = await tool.execute({});

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const url = getUrl(mockFetch.mock.calls[0]);
    expect(url).toContain("/v1/default/banks/test-bank/mental-models");
    expect(url).toContain("detail=metadata");
    expect(result.content[0].type).toBe("text");
  });

  it("get_page calls the correct endpoint", async () => {
    mockFetch.mockReturnValueOnce(mockResponse({ id: "my-page", content: "hello" }));

    const tool = tools.find((t) => t.name === "agent_knowledge_get_page")!;
    await tool.execute({ page_id: "my-page" });

    const url = getUrl(mockFetch.mock.calls[0]);
    expect(url).toContain("/v1/default/banks/test-bank/mental-models/my-page");
  });

  it("create_page sends correct body with defaults", async () => {
    mockFetch.mockReturnValueOnce(mockResponse({ id: "prefs" }));

    const tool = tools.find((t) => t.name === "agent_knowledge_create_page")!;
    await tool.execute({
      page_id: "prefs",
      name: "Preferences",
      source_query: "What are the user's preferences?",
    });

    const url = getUrl(mockFetch.mock.calls[0]);
    expect(url).toContain("/v1/default/banks/test-bank/mental-models");
    const body = await getBody(mockFetch.mock.calls[0]);
    expect(body.id).toBe("prefs");
    expect(body.name).toBe("Preferences");
    expect(body.source_query).toBe("What are the user's preferences?");
    expect(body.max_tokens).toBe(4096);
    expect(body.trigger.mode).toBe("delta");
    expect(body.trigger.refresh_after_consolidation).toBe(true);
    expect(body.trigger.exclude_mental_models).toBe(true);
    expect(body.trigger.fact_types).toEqual(["observation"]);
  });

  it("update_page sends PATCH", async () => {
    mockFetch.mockReturnValueOnce(mockResponse({ id: "prefs" }));

    const tool = tools.find((t) => t.name === "agent_knowledge_update_page")!;
    await tool.execute({ page_id: "prefs", name: "New Name" });

    const url = getUrl(mockFetch.mock.calls[0]);
    const opts = getOpts(mockFetch.mock.calls[0]);
    expect(url).toContain("/v1/default/banks/test-bank/mental-models/prefs");
    expect(opts.method).toBe("PATCH");
  });

  it("delete_page sends DELETE", async () => {
    mockFetch.mockReturnValueOnce(mockResponse({}));

    const tool = tools.find((t) => t.name === "agent_knowledge_delete_page")!;
    const result = await tool.execute({ page_id: "old-page" });

    const url = getUrl(mockFetch.mock.calls[0]);
    const opts = getOpts(mockFetch.mock.calls[0]);
    expect(url).toContain("/v1/default/banks/test-bank/mental-models/old-page");
    expect(opts.method).toBe("DELETE");
    expect(JSON.parse(result.content[0].text)).toEqual({ success: true });
  });

  it("recall sends POST with query", async () => {
    mockFetch.mockReturnValueOnce(mockResponse({ results: [{ text: "found" }] }));

    const tool = tools.find((t) => t.name === "agent_knowledge_recall")!;
    await tool.execute({ query: "what happened?" });

    const url = getUrl(mockFetch.mock.calls[0]);
    const opts = getOpts(mockFetch.mock.calls[0]);
    expect(url).toContain("/v1/default/banks/test-bank/memories/recall");
    expect(opts.method).toBe("POST");
  });

  it("ingest sends POST with content and document_id", async () => {
    mockFetch.mockReturnValueOnce(mockResponse({ status: "queued" }));

    const tool = tools.find((t) => t.name === "agent_knowledge_ingest")!;
    await tool.execute({ title: "My Document", content: "Full text here" });

    const url = getUrl(mockFetch.mock.calls[0]);
    expect(url).toContain("/v1/default/banks/test-bank/memories");
    const body = await getBody(mockFetch.mock.calls[0]);
    expect(body.items[0].document_id).toBe("my-document");
    expect(body.items[0].content).toBe("Full text here");
    expect(body.async).toBe(true);
  });

  it("works without apiToken", () => {
    const noAuthTools = createKnowledgeTools({
      apiUrl: "http://localhost:9077",
      bankId: "test-bank",
    });
    expect(noAuthTools).toHaveLength(7);
  });
});

describe("TOOL_NAMES", () => {
  it("exports all 7 tool names", () => {
    expect(TOOL_NAMES).toHaveLength(7);
    expect(TOOL_NAMES).toContain("agent_knowledge_list_pages");
    expect(TOOL_NAMES).toContain("agent_knowledge_ingest");
  });
});
