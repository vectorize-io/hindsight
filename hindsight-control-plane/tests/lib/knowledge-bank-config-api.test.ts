// @vitest-environment jsdom
/** The knowledge bank config calls: the path through the proxy, and the exact bodies. */
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("sonner", () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));

const { client } = await import("@/lib/api");

const ok = () =>
  new Response(JSON.stringify({ bank_id: "a b", config: {}, overrides: {} }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });

afterEach(() => vi.restoreAllMocks());

describe("knowledge bank config client", () => {
  it("reads, patches and resets /api/knowledge-banks/{bank}/config", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => ok());

    await client.getKnowledgeBankConfig("a b");
    await client.updateKnowledgeBankConfig("a b", { text_passage_tokens: 512, image_tokens: null });
    await client.resetKnowledgeBankConfig("a b");

    const calls = fetchSpy.mock.calls.map(([url, init]) => ({
      url: String(url),
      method: init?.method ?? "GET",
      body: init?.body,
    }));
    expect(calls).toEqual([
      { url: "/api/knowledge-banks/a%20b/config", method: "GET", body: undefined },
      {
        url: "/api/knowledge-banks/a%20b/config",
        method: "PATCH",
        body: JSON.stringify({ updates: { text_passage_tokens: 512, image_tokens: null } }),
      },
      { url: "/api/knowledge-banks/a%20b/config", method: "DELETE", body: undefined },
    ]);
  });

  it("throws the server's 400 reason", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ detail: "image_tokens: Input should be less than or equal to 1120" }),
        {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }
      )
    );
    await expect(client.updateKnowledgeBankConfig("a", { image_tokens: 5000 })).rejects.toThrow(
      "image_tokens: Input should be less than or equal to 1120"
    );
  });
});
