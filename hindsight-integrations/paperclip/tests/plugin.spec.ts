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
  return createTestHarness({
    manifest,
    config,
    capabilities: [...manifest.capabilities, "issues.create", "issue.comments.create"],
  });
}

async function setupPlugin(harness: ReturnType<typeof buildHarness>) {
  await plugin.definition.setup(harness.ctx);
}

async function seedIssue(
  harness: ReturnType<typeof buildHarness>,
  opts: { companyId: string; title: string; description?: string; assigneeAgentId?: string }
) {
  return harness.ctx.issues.create({
    companyId: opts.companyId,
    title: opts.title,
    description: opts.description,
    assigneeAgentId: opts.assigneeAgentId,
  });
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

  it("company + agent + user", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1", userId: "alice@acme.com" },
        { bankGranularity: ["company", "agent", "user"] }
      )
    ).toBe("paperclip::co-1::ag-1::user::alice@acme.com");
  });

  it("user granularity without userId falls back to company+agent", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankGranularity: ["company", "agent", "user"] }
      )
    ).toBe("paperclip::co-1::ag-1");
  });

  it("static bankId overrides derivation when dynamicBankId is not true", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankId: "spool-farm", bankGranularity: ["company", "agent"] }
      )
    ).toBe("spool-farm");
  });

  it("static bankId is trimmed", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId({ companyId: "co-1", agentId: "ag-1" }, { bankId: "  shared-bank  " })
    ).toBe("shared-bank");
  });

  it("empty/whitespace bankId falls through to dynamic derivation", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankId: "   ", bankGranularity: ["company", "agent"] }
      )
    ).toBe("paperclip::co-1::ag-1");
  });

  it("dynamicBankId=true ignores static bankId", async () => {
    const { deriveBankId } = await import("../src/bank.js");
    expect(
      deriveBankId(
        { companyId: "co-1", agentId: "ag-1" },
        { bankId: "spool-farm", dynamicBankId: true, bankGranularity: ["company", "agent"] }
      )
    ).toBe("paperclip::co-1::ag-1");
  });
});

// ---------------------------------------------------------------------------
// extractUserFromIssue
// ---------------------------------------------------------------------------

describe("extractUserFromIssue", () => {
  it("extracts email from originId", async () => {
    const { extractUserFromIssue } = await import("../src/bank.js");
    expect(extractUserFromIssue({ originId: "slack::alice@acme.com" })).toBe("alice@acme.com");
  });

  it("prefers creatorEmail over originId", async () => {
    const { extractUserFromIssue } = await import("../src/bank.js");
    expect(
      extractUserFromIssue({
        creatorEmail: "bob@acme.com",
        originId: "slack::alice@acme.com",
      })
    ).toBe("bob@acme.com");
  });

  it("returns undefined when no email found", async () => {
    const { extractUserFromIssue } = await import("../src/bank.js");
    expect(extractUserFromIssue({ originId: "slack::channel-123" })).toBeUndefined();
  });

  it("returns undefined when issue has no originId or creatorEmail", async () => {
    const { extractUserFromIssue } = await import("../src/bank.js");
    expect(extractUserFromIssue({})).toBeUndefined();
  });

  it("handles multi-segment originId", async () => {
    const { extractUserFromIssue } = await import("../src/bank.js");
    expect(extractUserFromIssue({ originId: "zendesk::org-42::ticket-7::user@corp.io" })).toBe(
      "user@corp.io"
    );
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

  it("fetches the issue, calls recall, and caches memories in plugin state", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);
    const issue = await seedIssue(harness, {
      companyId: "co-1",
      title: "Refactor auth module",
      description: "Migrate to JWT",
    });

    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-1", issueId: issue.id },
      { companyId: "co-1" }
    );

    const recallCall = fetchMock.mock.calls.find(([url]: [string]) => url.includes("recall"));
    expect(recallCall).toBeDefined();
    expect(recallCall?.[0]).toContain("paperclip%3A%3Aco-1%3A%3Aag-1");

    const recallBody = JSON.parse(recallCall?.[1]?.body as string) as { query: string };
    expect(recallBody.query).toContain("Refactor auth module");
    expect(recallBody.query).toContain("Migrate to JWT");

    const state = harness.getState({
      scopeKind: "run",
      scopeId: "run-1",
      stateKey: "recalled-memories",
    });
    expect(state).toContain("TypeScript");
  });

  it("uses user-scoped bank ID when bankGranularity includes 'user'", async () => {
    const harness = buildHarness({
      ...DEFAULT_CONFIG,
      bankGranularity: ["company", "agent", "user"],
    });
    await setupPlugin(harness);
    const issue = await seedIssue(harness, {
      companyId: "co-1",
      title: "Fix user-specific bug",
      description: "Details here",
    });
    // Simulate originId with user email (set via harness internals)
    (issue as Record<string, unknown>).originId = "slack::alice@acme.com";

    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-user-1", issueId: issue.id },
      { companyId: "co-1" }
    );

    const recallCall = fetchMock.mock.calls.find(([url]: [string]) => url.includes("recall"));
    expect(recallCall).toBeDefined();
    // Bank ID should include user segment
    expect(recallCall?.[0]).toContain(
      encodeURIComponent("paperclip::co-1::ag-1::user::alice@acme.com")
    );

    // Verify userId was cached in state for tool calls
    const cachedUserId = harness.getState({
      scopeKind: "run",
      scopeId: "run-user-1",
      stateKey: "user-id",
    });
    expect(cachedUserId).toBe("alice@acme.com");
  });

  it("routes recall to static bankId when dynamicBankId is not true", async () => {
    const harness = buildHarness({
      ...DEFAULT_CONFIG,
      bankId: "spool-farm",
      dynamicBankId: false,
    });
    await setupPlugin(harness);
    const issue = await seedIssue(harness, {
      companyId: "co-1",
      title: "Shared project context",
    });

    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-static-1", issueId: issue.id },
      { companyId: "co-1" }
    );

    const recallCall = fetchMock.mock.calls.find(([url]: [string]) => url.includes("recall"));
    expect(recallCall).toBeDefined();
    expect(recallCall?.[0]).toContain(encodeURIComponent("spool-farm"));
    // Should NOT contain the dynamic derivation prefix
    expect(recallCall?.[0]).not.toContain(encodeURIComponent("paperclip::"));
  });

  it("skips recall when no issueId is provided", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);

    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-2" },
      { companyId: "co-1" }
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("skips recall when the issue has no title or description", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);
    const issue = await seedIssue(harness, { companyId: "co-1", title: "" });

    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-2b", issueId: issue.id },
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
    const issue = await seedIssue(harness, { companyId: "co-1", title: "Fix bug" });

    await expect(
      harness.emit(
        "agent.run.started",
        { agentId: "ag-1", runId: "run-3", issueId: issue.id },
        { companyId: "co-1" }
      )
    ).resolves.not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// issue.comment.created — auto-retain
// ---------------------------------------------------------------------------

describe("issue.comment.created", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = mockFetch([{ url: /memories$/, body: { success: true } }]);
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("retains the full comment body with the commentId as document ID", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);
    const issue = await seedIssue(harness, {
      companyId: "co-1",
      title: "Refactor auth",
      assigneeAgentId: "ag-1",
    });
    const comment = await harness.ctx.issues.createComment(
      issue.id,
      "Refactored auth. Migrated to JWT with 24h expiry.",
      "co-1",
      { authorAgentId: "ag-1" }
    );

    await harness.emit(
      "issue.comment.created",
      { commentId: comment.id, agentId: "ag-1", runId: "run-1" },
      { companyId: "co-1", entityId: issue.id }
    );

    const retainCall = fetchMock.mock.calls.find(([url]: [string]) => /memories$/.test(url));
    expect(retainCall).toBeDefined();
    const body = JSON.parse(retainCall?.[1]?.body as string) as {
      items: Array<{ content: string; document_id?: string }>;
    };
    expect(body.items[0]?.content).toContain("JWT");
    expect(body.items[0]?.document_id).toBe(comment.id);
  });

  it("skips retain when comment has no agent author and issue has no assignee", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);
    const issue = await seedIssue(harness, { companyId: "co-1", title: "x" });
    const comment = await harness.ctx.issues.createComment(issue.id, "User message", "co-1");

    await harness.emit(
      "issue.comment.created",
      { commentId: comment.id, agentId: null },
      { companyId: "co-1", entityId: issue.id }
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("skips retain when autoRetain is false", async () => {
    const harness = buildHarness({ ...DEFAULT_CONFIG, autoRetain: false });
    await setupPlugin(harness);
    const issue = await seedIssue(harness, {
      companyId: "co-1",
      title: "x",
      assigneeAgentId: "ag-1",
    });
    const comment = await harness.ctx.issues.createComment(issue.id, "Some output", "co-1", {
      authorAgentId: "ag-1",
    });

    await harness.emit(
      "issue.comment.created",
      { commentId: comment.id, agentId: "ag-1" },
      { companyId: "co-1", entityId: issue.id }
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("skips retain when commentId is missing", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);
    const issue = await seedIssue(harness, { companyId: "co-1", title: "x" });

    await harness.emit("issue.comment.created", {}, { companyId: "co-1", entityId: issue.id });

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("falls back to the issue assignee for bank attribution when comment has no agent author", async () => {
    const harness = buildHarness();
    await setupPlugin(harness);
    const issue = await seedIssue(harness, {
      companyId: "co-1",
      title: "x",
      assigneeAgentId: "ag-assignee",
    });
    const comment = await harness.ctx.issues.createComment(
      issue.id,
      "User says: please do X",
      "co-1"
    );

    await harness.emit(
      "issue.comment.created",
      { commentId: comment.id, agentId: null },
      { companyId: "co-1", entityId: issue.id }
    );

    const retainCall = fetchMock.mock.calls.find(([url]: [string]) => /memories$/.test(url));
    expect(retainCall).toBeDefined();
    expect(retainCall?.[0]).toContain("paperclip%3A%3Aco-1%3A%3Aag-assignee");
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
    const harness = buildHarness();
    await setupPlugin(harness);
    const issue = await seedIssue(harness, {
      companyId: "co-1",
      title: "Update UI",
    });

    vi.stubGlobal(
      "fetch",
      mockFetch([{ url: /recall/, body: { results: [{ text: "User prefers dark mode" }] } }])
    );
    await harness.emit(
      "agent.run.started",
      { agentId: "ag-1", runId: "run-1", issueId: issue.id },
      { companyId: "co-1" }
    );

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
