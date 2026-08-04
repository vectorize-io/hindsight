import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchTransport } from "../../src/node/fetch-transport";

function fetchReturning(status: number, body: string) {
  return vi.fn(async () => ({ status, text: async () => body }) as unknown as Response);
}

describe("fetchTransport", () => {
  beforeEach(() => vi.unstubAllGlobals());
  afterEach(() => vi.unstubAllGlobals());

  it("forwards method, headers and body to fetch", async () => {
    const spy = fetchReturning(200, "{}");
    vi.stubGlobal("fetch", spy);

    await fetchTransport({
      url: "https://api.example.com/x",
      method: "POST",
      headers: { Authorization: "Bearer t", "Content-Type": "application/json" },
      body: '{"a":1}',
    });

    expect(spy).toHaveBeenCalledWith("https://api.example.com/x", {
      method: "POST",
      headers: { Authorization: "Bearer t", "Content-Type": "application/json" },
      body: '{"a":1}',
    });
  });

  it("returns status, raw text, and parsed json", async () => {
    vi.stubGlobal("fetch", fetchReturning(201, '{"ok":true}'));
    const resp = await fetchTransport({ url: "u", method: "GET", headers: {} });
    expect(resp.status).toBe(201);
    expect(resp.text).toBe('{"ok":true}');
    expect(resp.json).toEqual({ ok: true });
  });

  it("leaves json undefined for a non-JSON body (e.g. an error page)", async () => {
    vi.stubGlobal("fetch", fetchReturning(500, "Internal Server Error"));
    const resp = await fetchTransport({ url: "u", method: "GET", headers: {} });
    expect(resp.status).toBe(500);
    expect(resp.text).toBe("Internal Server Error");
    expect(resp.json).toBeUndefined();
  });
});
