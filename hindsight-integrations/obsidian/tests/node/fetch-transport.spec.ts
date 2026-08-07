import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HindsightClient } from "../../src/client";
import { fetchTransport } from "../../src/node/fetch-transport";
import type { Transport, TransportRequest } from "../../src/transport";

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

  it("propagates a fetch rejection (network failure) to the caller", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new Error("ECONNREFUSED")))
    );
    await expect(fetchTransport({ url: "u", method: "GET", headers: {} })).rejects.toThrow(
      /ECONNREFUSED/
    );
  });
});

describe("HindsightClient over fetchTransport (full stack)", () => {
  beforeEach(() => vi.unstubAllGlobals());
  afterEach(() => vi.unstubAllGlobals());

  it("surfaces a useful error on a non-2xx response", async () => {
    vi.stubGlobal("fetch", fetchReturning(503, "upstream down"));
    const client = new HindsightClient("https://h", "tok", fetchTransport);
    await expect(client.reflect("b", "q")).rejects.toThrow(/HTTP 503: upstream down/);
  });

  it("health() returns false when the transport rejects, true on 2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new Error("down")))
    );
    expect(await new HindsightClient("https://h", undefined, fetchTransport).health()).toBe(false);

    vi.stubGlobal("fetch", fetchReturning(200, "{}"));
    expect(await new HindsightClient("https://h", undefined, fetchTransport).health()).toBe(true);
  });
});

describe("cross-transport request parity", () => {
  // The same client call must produce an identical request regardless of which
  // transport carries it — this is what guarantees the CLI and the plugin talk
  // to Hindsight the same way.
  function recorder(): { transport: Transport; last: () => TransportRequest } {
    let seen: TransportRequest | undefined;
    return {
      transport: async (req) => {
        seen = req;
        return { status: 200, text: "{}", json: {} };
      },
      last: () => {
        if (!seen) throw new Error("transport not called");
        return seen;
      },
    };
  }

  it("retain builds the same request under two different transports", async () => {
    const a = recorder();
    const b = recorder();
    await new HindsightClient("https://h/", "tok", a.transport).retain("bank", "F/n.md", "body", {
      tags: ["t"],
      metadata: { path: "F/n.md" },
    });
    await new HindsightClient("https://h/", "tok", b.transport).retain("bank", "F/n.md", "body", {
      tags: ["t"],
      metadata: { path: "F/n.md" },
    });
    expect(a.last()).toEqual(b.last());
    expect(a.last().url).toBe("https://h/v1/default/banks/bank/memories");
    expect(a.last().headers.Authorization).toBe("Bearer tok");
  });
});
