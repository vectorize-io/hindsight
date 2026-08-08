import { afterEach, describe, expect, it, vi } from "vitest";
import { HindsightClient } from "./hindsight";

afterEach(() => vi.restoreAllMocks());

function retainRequest(content: string) {
  const requests: unknown[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: string, init: RequestInit) => {
      requests.push(JSON.parse(String(init.body)));
      return { ok: true, status: 200, json: async () => ({ operation_id: "op-1" }) } as Response;
    })
  );

  const client = new HindsightClient({ apiUrl: "http://hindsight.test", bank: "coding" });
  return client
    .retain(content, "coding agent session", "conversation:s1", ["source:chat"], "conversation", {
      async: true,
      idempotent: true,
      timestamp: "2026-01-01T00:00:00Z",
      metadata: { source: "chat", session_id: "s1" },
    })
    .then(() => requests[0] as { operation_id?: string });
}

describe("HindsightClient idempotent retain", () => {
  it("adds one stable UUID operation_id for an identical async retain", async () => {
    const first = await retainRequest("first turn");
    const second = await retainRequest("first turn");

    expect(first.operation_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
    );
    expect(second.operation_id).toBe(first.operation_id);
  });

  it("uses a different operation_id when the retained payload changes", async () => {
    const first = await retainRequest("first turn");
    const second = await retainRequest("first turn with another message");

    expect(second.operation_id).not.toBe(first.operation_id);
  });

  it("leaves ordinary retains unchanged", async () => {
    const requests: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        requests.push(JSON.parse(String(init.body)));
        return { ok: true, status: 200, json: async () => ({ operation_id: "op-1" }) } as Response;
      })
    );
    const client = new HindsightClient({ apiUrl: "http://hindsight.test", bank: "coding" });

    await client.retain("one-off", "manual", "document:1", ["source:manual"], "document");

    expect(requests[0]).not.toHaveProperty("operation_id");
  });
});
