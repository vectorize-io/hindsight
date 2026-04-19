/**
 * Tests for paperclip-plugin-hindsight.
 *
 * Uses @paperclipai/plugin-sdk's createTestHarness to simulate the Paperclip
 * host environment without requiring a running Paperclip instance.
 *
 * Hindsight API calls are intercepted via global fetch mocking.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createTestHarness } from "@paperclipai/plugin-sdk";
import manifest from "../src/manifest.js";
import plugin from "../src/worker.js";

// ---------------------------------------------------------------------------
// Fetch mock helpers
// ---------------------------------------------------------------------------

function mockFetch(responses: Array<{ url: string | RegExp; body: unknown; status?: number }>) {
  return vi.fn(async (url: string) => {
    const match = responses.find((r) =>
      typeof r.url === "string" ? url.includes(r.url) : r.url.test(url)
    );
    if (!match) {
      return new Response(JSON.stringify({ error: "unmatched url" }), { status: 404 });
    }
    return new Response(JSON.stringify(match.body), {
      status: match.status ?? 200,
    });
  });
}

// ---------------------------------------------------------------------------
// Harness setup
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG = {
  hindsightApiUrl: "http://localhost:8888",
  bankGranularity: ["company", "agent"],
  recallBudget: "mid",
  autoRetain: true,
};

function buildHarness(config: Record<string, unknown> = DEFAULT_CONFIG) {
  return createTestHarness({ manifest, config, capabilities: manifest.capabilities });
}

async function setupPlugin(harness: ReturnType<typeof buildHarness>) {
  await plugin.definition.setup(harness.ctx);
}

// ---------------------------------------------------------------------------
// Bank ID derivation
// ---------------------------------------------------------------------------

describe("bank ID derivation", () => {
  it("default: company + agent", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankGranularity: ["company", "agent"] }
      )
    ).toBe("paperclip::co-1::ag-1");
  });

  it("company only", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankGranularity: ["company"] })
    ).toBe("paperclip::co-1");
  });

  it("agent only", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankGranularity: ["agent"] })
    ).toBe("paperclip::ag-1");
  });
});

// ---------------------------------------------------------------------------
// agent.run.started — recall
// ---------------------------------------------------------------------------

describe("agent.run.started", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = mockFetch([
      { url: /recall/, body: { results: [{ text: "User prefers TypeScript" }] } },
    ]);
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("calls recall and caches memories in plugin state", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);

    await harness.emit(
      "agent.run.started",
      {
        agentId: "ag-1",
        runId: "run-1",
        issueTitle: "Refactor auth module",
        issueDescription: "Migrate to JWT",
      },
      { companyId: "co-1" }
    );

    const recallCall = fetchMock.mock.calls.find(([url]: [string]) => url.includes("recall"));
    expect(recallCall).toBeDefined();
    expect(recallCall?.[0]).toContain("paperclip%3A%3Aco-1%3A%3Aag-1");

    const state = harness.getState({
      scopeKind: "run",
      scopeId: "run-1",
      stateKey: "recalled-memories",
    });
    expect(state).toContain("TypeScript");
  });

  it("skips recall when no issue context provided", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);

    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-2" },
      { companyId: "co-1" }
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("does not throw when Hindsight is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("", { status: 503 }))
    );
    const harness = buildHarness();
    await setupPlugin(harness);

    await expect(
      harness.emit(
        "agent.run.started",
        { agentId: "ag-1", runId: "run-3", issueTitle: "Fix bug" },
        { companyId: "co-1" }
      )
    ).resolves.not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// agent.run.finished — auto-retain
// ---------------------------------------------------------------------------

describe("agent.run.finished", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = mockFetch([{ url: /memories$/, body: { success: true } }]);
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("retains run output with runId as document ID", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);

    await harness.emit(
      "agent.run.finished",
      {
        agentId: "ag-1",
        runId: "run-1",
        output: "Refactored auth. Migrated to JWT with 24h expiry.",
      },
      { companyId: "co-1" }
    );

    const retainCall = fetchMock.mock.calls.find(([url]: [string]) => /memories$/.test(url));
    expect(retainCall).toBeDefined();

    const body = JSON.parse(retainCall?.[1]?.body as string) as {
      items: Array<{ content: string; document_id?: string }>;
    };
    expect(body.items[0]?.content).toContain("JWT");
    expect(body.items[0]?.document_id).toBe("run-1");
  });

  it("skips retain when output is empty", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);

    await harness.emit(
      "agent.run.finished",
      { agentId: "ag-1", runId: "run-2", output: "" },
      { companyId: "co-1" }
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("skips retain when autoRetain is false", async () => {
    const harness = buildHarness({ ...DEFAULT_CONFIG, autoRetain: false });
    await setupPlugin(harness);

    await harness.emit(
      "agent.run.finished",
      { agentId: "ag-1", runId: "run-3", output: "Some output" },
      { companyId: "co-1" }
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// hindsight_recall tool
// ---------------------------------------------------------------------------

describe("hindsight_recall tool", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns cached memories from run start without additional API call", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 200 }))
    );
    const harness = buildHarness();
    await setupPlugin(harness);

    // Emit agent.run.started so recall fires and caches state
    vi.stubGlobal(
      "fetch",
      mockFetch([{ url: /recall/, body: { results: [{ text: "User prefers dark mode" }] } }])
    );
    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-1", issueTitle: "Update UI" },
      { companyId: "co-1" }
    );

    // Now recall tool should return cached state, not hit the API again
    const callsBefore = (vi.mocked(fetch) as ReturnType<typeof vi.fn>).mock.calls.length;
    const result = await harness.executeTool(
      "hindsight_recall",
      { query: "preferences" },
      { agentId: "ag-1", runId: "run-1", companyId: "co-1", projectId: "proj-1" }
    );

    expect((result as { content: string }).content).toContain("dark mode");
    const callsAfter = (vi.mocked(fetch) as ReturnType<typeof vi.fn>).mock.calls.length;
    // No new recall call — returned from cache
    expect(callsAfter).toBe(callsBefore);
  });

  it("falls back to live recall when no cached state", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch([{ url: /recall/, body: { results: [{ text: "Agent is a Python specialist" }] } }])
    );
    const harness = buildHarness();
    await setupPlugin(harness);

    const result = await harness.executeTool(
      "hindsight_recall",
      { query: "specialization" },
      { agentId: "ag-1", runId: "run-2", companyId: "co-1", projectId: "proj-1" }
    );

    expect((result as { content: string }).content).toContain("Python specialist");
  });
});

// ---------------------------------------------------------------------------
// hindsight_retain tool
// ---------------------------------------------------------------------------

describe("hindsight_retain tool", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stores content via Hindsight retain endpoint", async () => {
    const fetchMock = mockFetch([{ url: /memories$/, body: { success: true } }]);
    vi.stubGlobal("fetch", fetchMock);
    const harness = buildHarness();
    await setupPlugin(harness);

    const result = await harness.executeTool(
      "hindsight_retain",
      { content: "Decision: use Postgres not MySQL" },
      { agentId: "ag-1", runId: "run-1", companyId: "co-1", projectId: "proj-1" }
    );

    expect((result as { content: string }).content).toBe("Memory saved.");
    const call = fetchMock.mock.calls.find(([url]: [string]) => /memories$/.test(url));
    const body = JSON.parse(call?.[1]?.body as string) as {
      items: Array<{ content: string }>;
    };
    expect(body.items[0]?.content).toContain("Postgres");
  });
});

// ---------------------------------------------------------------------------
// onValidateConfig
// ---------------------------------------------------------------------------

describe("onValidateConfig", () => {
  it("fails when hindsightApiUrl is missing", async () => {
    const result = await plugin.definition.onValidateConfig!({ hindsightApiUrl: "" });
    expect(result.ok).toBe(false);
    expect(result.errors?.some((e) => e.includes("hindsightApiUrl"))).toBe(true);
  });

  it("fails when Hindsight is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("", { status: 503 }))
    );
    const result = await plugin.definition.onValidateConfig!({
      hindsightApiUrl: "http://localhost:8888",
    });
    expect(result.ok).toBe(false);
    vi.unstubAllGlobals();
  });

  it("passes with a reachable Hindsight instance", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("{}", { status: 200 }))
    );
    const result = await plugin.definition.onValidateConfig!({
      hindsightApiUrl: "http://localhost:8888",
    });
    expect(result.ok).toBe(true);
    vi.unstubAllGlobals();
  });
});
