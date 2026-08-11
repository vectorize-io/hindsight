import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { buildKnowledgeTools } from "./knowledge-tools";
import type { HindsightClient } from "./hindsight";

/** Minimal stub of the HindsightClient surface the tools call — no SDK, no network. */
function stubClient(overrides: Partial<Record<string, ReturnType<typeof vi.fn>>> = {}) {
  return {
    listPages: vi.fn(async () => ({ pages: [] })),
    getPage: vi.fn(async (_id: string) => ({ id: _id })),
    recall: vi.fn(async (_q: string, _o: unknown) => [{ text: "a fact" }]),
    captureInitiative: vi.fn(async (_a: unknown) => ({ page_id: "initiative-x" })),
    retain: vi.fn(async (..._args: unknown[]) => undefined),
    ...overrides,
  } as unknown as HindsightClient;
}

function findTool(tools: ReturnType<typeof buildKnowledgeTools>, name: string) {
  const t = tools.find((t) => t.name === name);
  if (!t) throw new Error(`tool not found: ${name}`);
  return t;
}

const EXPECTED_TOOLS = [
  "hindsight_sync_status",
  "hindsight_diagnose",
  "hindsight_search_knowledge_pages",
  "hindsight_list_knowledge_pages",
  "hindsight_read_knowledge_page",
  "hindsight_reflect",
  "hindsight_capture_initiative",
  "hindsight_ingest_document",
];

describe("buildKnowledgeTools", () => {
  it("returns exactly the eight expected tools (as a set)", () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    expect(tools.map((t) => t.name).sort()).toEqual([...EXPECTED_TOOLS].sort());
  });

  it("hindsight_sync_status is the FIRST tool in the list", () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    expect(tools[0].name).toBe("hindsight_sync_status");
  });

  it("does not expose the removed raw page-CRUD tools", () => {
    const client = stubClient();
    const names = buildKnowledgeTools(client, "repo-a").map((t) => t.name);
    expect(names).not.toContain("create_page");
    expect(names).not.toContain("update_page");
    expect(names).not.toContain("delete_page");
    expect(names).not.toContain("agent_knowledge_create_page");
    expect(names).not.toContain("agent_knowledge_update_page");
    expect(names).not.toContain("agent_knowledge_delete_page");
  });

  it("hindsight_sync_status returns the syncStatus JSON for the given repoDir", async () => {
    const repoDir = mkdtempSync(join(tmpdir(), "hs-status-nogit-"));
    try {
      const repoName = basename(repoDir);
      const client = stubClient({
        listDocumentIds: vi.fn(async (tag: string) =>
          tag === "source:git"
            ? new Set([`gitlog:${repoName}`, "git:sha1", "git:sha2"])
            : new Set(["chat:s1"])
        ),
        listPages: vi.fn(async () => ({ items: [{ id: "p1", name: "Component map" }] })),
        activeOperations: vi.fn(async () => 0),
      });
      const tools = buildKnowledgeTools(client, "repo-a", { repoDir });
      const tool = findTool(tools, "hindsight_sync_status");
      const result = await tool.handler({});
      expect(result.isError).toBeFalsy();
      expect(JSON.parse(result.content[0].text)).toEqual({
        bank: "repo-a",
        gitlogPresent: true,
        gitDiffDocs: 2,
        gitDiffTarget: null, // temp dir is not a git repo
        surveyBaseline: null,
        surveyDocs: 0,
        surveyCommitsBehind: null,
        chatDocs: 1,
        pagesCount: 1,
        activeOps: 0,
        synced: true,
      });
    } finally {
      rmSync(repoDir, { recursive: true, force: true });
    }
  });

  it("hindsight_sync_status returns isError:true when the client's listDocumentIds throws", async () => {
    const repoDir = mkdtempSync(join(tmpdir(), "hs-status-nogit-"));
    try {
      const client = stubClient({
        listDocumentIds: vi.fn(async () => {
          throw new Error("server down");
        }),
        listPages: vi.fn(async () => ({ items: [] })),
        activeOperations: vi.fn(async () => 0),
      });
      const tools = buildKnowledgeTools(client, "repo-a", { repoDir });
      const tool = findTool(tools, "hindsight_sync_status");
      const result = await tool.handler({});
      expect(result.isError).toBe(true);
      expect(JSON.parse(result.content[0].text)).toEqual({ error: "server down" });
    } finally {
      rmSync(repoDir, { recursive: true, force: true });
    }
  });

  it("hindsight_diagnose returns safe runtime configuration without touching the client", async () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_diagnose");
    const result = await tool.handler({});
    expect(result.isError).toBeFalsy();
    expect(JSON.parse(result.content[0].text)).toMatchObject({
      bank_id: "repo-a",
      harness: "unknown",
      config: { api_token_configured: false },
    });
    expect(client.listPages).not.toHaveBeenCalled();
    expect(client.getPage).not.toHaveBeenCalled();
  });

  it("hindsight_search_knowledge_pages calls the server hybrid search and returns ranked hits", async () => {
    const client = stubClient({
      searchKnowledgePages: vi.fn(async () => [
        { id: "p1", name: "Uploader guide", snippet: "Uploads retry with backoff…", score: 0.031 },
        { id: "p2", name: "Auth notes", snippet: "Tokens rotate daily.", score: 0.012 },
      ]),
    });
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_search_knowledge_pages");
    const result = await tool.handler({ query: "upload retries" });
    expect(result.isError).toBeFalsy();
    expect(client.searchKnowledgePages).toHaveBeenCalledWith("upload retries", 3);
    expect(JSON.parse(result.content[0].text)).toEqual([
      {
        page: "Uploader guide",
        page_id: "p1",
        snippet: "Uploads retry with backoff…",
        score: 0.031,
      },
      { page: "Auth notes", page_id: "p2", snippet: "Tokens rotate daily.", score: 0.012 },
    ]);
  });

  it("hindsight_search_knowledge_pages returns isError:true when the server search throws", async () => {
    const client = stubClient({
      searchKnowledgePages: vi.fn(async () => {
        throw new Error("search down");
      }),
    });
    const tool = findTool(
      buildKnowledgeTools(client, "repo-a"),
      "hindsight_search_knowledge_pages"
    );
    const result = await tool.handler({ query: "x" });
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toContain("search down");
  });

  it("hindsight_list_knowledge_pages calls client.listPages() with no args", async () => {
    const client = stubClient({ listPages: vi.fn(async () => ({ pages: [{ id: "p1" }] })) });
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_list_knowledge_pages");
    const result = await tool.handler({});
    expect(client.listPages).toHaveBeenCalledWith();
    expect(JSON.parse(result.content[0].text)).toEqual({ pages: [{ id: "p1" }] });
  });

  it("hindsight_read_knowledge_page calls client.getPage(page_id)", async () => {
    const client = stubClient({ getPage: vi.fn(async (id: string) => ({ id, name: "X" })) });
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_read_knowledge_page");
    const result = await tool.handler({ page_id: "p1" });
    expect(client.getPage).toHaveBeenCalledWith("p1");
    expect(JSON.parse(result.content[0].text)).toEqual({ id: "p1", name: "X" });
  });

  it("hindsight_reflect calls client.reflect(query, {budget: high}) and returns the synthesis", async () => {
    const client = stubClient({
      reflect: vi.fn(async () => "the decided rule is X=3"),
    });
    const tool = findTool(buildKnowledgeTools(client, "repo-a"), "hindsight_reflect");
    const result = await tool.handler({ query: "why is X 3?" });
    expect(result.isError).toBeFalsy();
    expect(client.reflect).toHaveBeenCalledWith("why is X 3?", { budget: "high" });
    expect(JSON.parse(result.content[0].text)).toBe("the decided rule is X=3");
  });

  it("hindsight_capture_initiative calls client.captureInitiative({title, summary, relatesToPageId}) and returns the page id", async () => {
    const client = stubClient({
      captureInitiative: vi.fn(async (_a: unknown) => ({ page_id: "initiative-retry-backoff" })),
    });
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_capture_initiative");
    const result = await tool.handler({
      title: "Retry backoff for the uploader",
      summary: "Add exponential backoff so transient upload failures self-heal.",
      relates_to_page_id: "initiative-uploader",
    });
    expect(client.captureInitiative).toHaveBeenCalledWith({
      title: "Retry backoff for the uploader",
      summary: "Add exponential backoff so transient upload failures self-heal.",
      relatesToPageId: "initiative-uploader",
    });
    expect(JSON.parse(result.content[0].text)).toEqual({ page_id: "initiative-retry-backoff" });
  });

  it("hindsight_capture_initiative passes relatesToPageId: undefined for a brand-new initiative", async () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_capture_initiative");
    await tool.handler({ title: "New thing", summary: "why" });
    expect(client.captureInitiative).toHaveBeenCalledWith({
      title: "New thing",
      summary: "why",
      relatesToPageId: undefined,
    });
  });

  it("applies retain attribution to initiative capture and manual document ingestion", async () => {
    const client = stubClient();
    const stampFor = vi.fn(() => ({
      tags: ["project:repo-a"],
      metadata: { project: "repo-a" },
    }));
    const tools = buildKnowledgeTools(client, "repo-a", { harness: "codex", stampFor });

    await findTool(tools, "hindsight_capture_initiative").handler({
      title: "New thing",
      summary: "why",
    });
    expect(client.captureInitiative).toHaveBeenCalledWith({
      title: "New thing",
      summary: "why",
      relatesToPageId: undefined,
      stamp: { tags: ["project:repo-a"], metadata: { project: "repo-a" } },
    });

    await findTool(tools, "hindsight_ingest_document").handler({ title: "Notes", content: "x" });
    expect(client.retain).toHaveBeenCalledWith(
      "x",
      "ingested document",
      "notes",
      ["project:repo-a", "source:upload", "harness:codex"],
      "document",
      { metadata: { project: "repo-a", harness: "codex" } }
    );
    expect(stampFor).toHaveBeenCalledTimes(2);
  });

  it("hindsight_ingest_document slugifies the title and calls client.retain(...) with the 'document' strategy", async () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_ingest_document");
    const result = await tool.handler({ title: "My Title", content: "some content" });
    expect(client.retain).toHaveBeenCalledWith(
      "some content",
      "ingested document",
      "my-title",
      ["source:upload"],
      "document",
      {} // no harness in these tests: nothing to stamp
    );
    expect(JSON.parse(result.content[0].text)).toEqual({ ok: true, doc_id: "my-title" });
  });

  it("hindsight_ingest_document collapses internal whitespace runs in the title into single hyphens", async () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_ingest_document");
    await tool.handler({ title: "Repo   Core  Concepts", content: "x" });
    expect(client.retain).toHaveBeenCalledWith(
      "x",
      "ingested document",
      "repo-core-concepts",
      ["source:upload"],
      "document",
      {} // no harness in these tests: nothing to stamp
    );
  });

  it("hindsight_ingest_document strips punctuation from the title into a safe slug", async () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_ingest_document");
    await tool.handler({ title: "Repo: Component Map! (v2/final)", content: "x" });
    expect(client.retain).toHaveBeenCalledWith(
      "x",
      "ingested document",
      "repo-component-map-v2-final",
      ["source:upload"],
      "document",
      {} // no harness in these tests: nothing to stamp
    );
  });

  it("hindsight_ingest_document falls back to 'doc' when the title has no safe characters", async () => {
    const client = stubClient();
    const tools = buildKnowledgeTools(client, "repo-a");
    const tool = findTool(tools, "hindsight_ingest_document");
    await tool.handler({ title: "!!!///???", content: "x" });
    expect(client.retain).toHaveBeenCalledWith(
      "x",
      "ingested document",
      "doc",
      ["source:upload"],
      "document",
      {} // no harness in these tests: nothing to stamp
    );
  });

  for (const name of [
    "hindsight_list_knowledge_pages",
    "hindsight_read_knowledge_page",
    "hindsight_reflect",
    "hindsight_capture_initiative",
    "hindsight_ingest_document",
  ] as const) {
    it(`${name} returns isError:true with the error text when the client method throws`, async () => {
      const boom = new Error("boom: not found");
      const client = stubClient({
        listPages: vi.fn(async () => {
          throw boom;
        }),
        getPage: vi.fn(async () => {
          throw boom;
        }),
        reflect: vi.fn(async () => {
          throw boom;
        }),
        captureInitiative: vi.fn(async () => {
          throw boom;
        }),
        retain: vi.fn(async () => {
          throw boom;
        }),
      });
      const tools = buildKnowledgeTools(client, "repo-a");
      const tool = findTool(tools, name);
      const args =
        name === "hindsight_reflect"
          ? { query: "q" }
          : name === "hindsight_capture_initiative"
            ? { title: "T", summary: "S" }
            : name === "hindsight_ingest_document"
              ? { title: "T", content: "C" }
              : name === "hindsight_read_knowledge_page"
                ? { page_id: "p1" }
                : {};
      const result = await tool.handler(args);
      expect(result.isError).toBe(true);
      expect(JSON.parse(result.content[0].text)).toEqual({ error: "boom: not found" });
    });
  }
});
