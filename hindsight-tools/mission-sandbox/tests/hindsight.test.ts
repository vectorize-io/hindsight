import { afterEach, describe, expect, it, vi } from "vitest";

import { SandboxApi } from "../src/core/hindsight.js";

vi.mock("@vectorize-io/hindsight-client", () => ({
  HindsightClient: class {
    createBank = vi.fn();
    retain = vi.fn();
    updateBankConfig = vi.fn();
  },
}));

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SandboxApi raw endpoints", () => {
  it("reads pending consolidation + total from /stats", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ pending_consolidation: 4, total_observations: 12 }));
    vi.stubGlobal("fetch", fetchMock);

    const api = new SandboxApi("http://localhost:8888/");
    const stats = await api.getStats("bank-1");

    expect(stats).toEqual({ pendingConsolidation: 4, totalObservations: 12 });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8888/v1/default/banks/bank-1/stats",
      expect.objectContaining({ method: "GET" })
    );
  });

  it("returns deleted_count from clearObservations", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ deleted_count: 7 })));
    const api = new SandboxApi("http://localhost:8888");
    expect(await api.clearObservations("bank-1")).toBe(7);
  });

  it("throws with status + body on a failed request", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "boom" }, false, 500)));
    const api = new SandboxApi("http://localhost:8888");
    await expect(api.triggerConsolidation("bank-1")).rejects.toThrow(/500/);
  });

  it("polls until consolidation drains", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ pending_consolidation: 2 }))
      .mockResolvedValueOnce(jsonResponse({ pending_consolidation: 0 }));
    vi.stubGlobal("fetch", fetchMock);

    const api = new SandboxApi("http://localhost:8888");
    await api.waitForConsolidation("bank-1", { pollMs: 1 });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("SandboxApi.dryRunExtract", () => {
  it("POSTs to /memories/dry-run-extract with the mission override and parses facts", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        facts: [
          { text: "Alice lives in Berlin.", fact_type: "world", entities: ["Alice", "Berlin"] },
          { text: "Alice is a nurse.", fact_type: "world", entities: ["Alice"] },
        ],
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const api = new SandboxApi("http://localhost:8888");
    const facts = await api.dryRunExtract("bank-1", "Alice lives in Berlin and is a nurse.", {
      retainMission: "Capture where people live and their jobs.",
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8888/v1/default/banks/bank-1/memories/dry-run-extract");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body.content).toContain("Alice lives in Berlin");
    expect(body.retain_mission).toBe("Capture where people live and their jobs.");

    // Dry-run facts are a subset (no id/document_id), so the mapped rows carry empty id + null docId.
    expect(facts).toHaveLength(2);
    expect(facts[0]).toMatchObject({
      text: "Alice lives in Berlin.",
      factType: "world",
      docId: null,
    });
  });
});
