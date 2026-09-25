import { afterEach, describe, expect, it, vi } from "vitest";
import {
  HindsightRestClient,
  SENTINEL_OPEN,
  SENTINEL_CLOSE,
  buildRecallMarkdown,
  stripSentinelBlocks,
} from "./client";

function mockFetchOnce(status: number, json: unknown): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async () => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => json,
    text: async () => JSON.stringify(json),
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("HindsightRestClient.recall", () => {
  it("POSTs to the recall path with query/budget/max_tokens and bearer auth", async () => {
    const fetchFn = mockFetchOnce(200, {
      results: [{ id: "1", text: "likes tabs", type: "world" }],
    });
    const client = new HindsightRestClient("https://api.hindsight.vectorize.io", "hsk_k");

    const res = await client.recall("bank-1", "preferences", { budget: "low", maxTokens: 512 });

    expect(res.results[0].text).toBe("likes tabs");
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("https://api.hindsight.vectorize.io/v1/default/banks/bank-1/memories/recall");
    expect(init.method).toBe("POST");
    expect(init.headers["Authorization"]).toBe("Bearer hsk_k");
    expect(JSON.parse(init.body)).toEqual({ query: "preferences", budget: "low", max_tokens: 512 });
  });

  it("defaults budget=mid and max_tokens=1024, omits auth header when no token", async () => {
    const fetchFn = mockFetchOnce(200, { results: [] });
    const client = new HindsightRestClient("http://localhost:8000", null);
    await client.recall("b", "q");
    const init = fetchFn.mock.calls[0][1];
    expect(init.headers["Authorization"]).toBeUndefined();
    expect(JSON.parse(init.body)).toEqual({ query: "q", budget: "mid", max_tokens: 1024 });
  });
});

describe("HindsightRestClient.retain", () => {
  it("POSTs items with async=true to the memories path", async () => {
    const fetchFn = mockFetchOnce(200, { success: true });
    const client = new HindsightRestClient("https://api.hindsight.vectorize.io/", "hsk_k");
    await client.retain("my bank", [{ content: "fact", context: "eve" }]);
    const [url, init] = fetchFn.mock.calls[0];
    // trailing slash on baseUrl is normalized; bank is URL-encoded
    expect(url).toBe("https://api.hindsight.vectorize.io/v1/default/banks/my%20bank/memories");
    expect(JSON.parse(init.body)).toEqual({
      items: [{ content: "fact", context: "eve" }],
      async: true,
    });
  });
});

describe("HindsightRestClient.recall on a missing bank", () => {
  it("treats a 404 as no memories, since retain creates the bank on first write", async () => {
    mockFetchOnce(404, { detail: "Bank 'b' not found" });
    const client = new HindsightRestClient("http://localhost:8000", null);
    await expect(client.recall("b", "q")).resolves.toEqual({ results: [] });
  });

  it("still surfaces other statuses with their code", async () => {
    mockFetchOnce(503, { detail: "down" });
    const client = new HindsightRestClient("http://localhost:8000", null);
    await expect(client.recall("b", "q")).rejects.toMatchObject({
      name: "HindsightHttpError",
      status: 503,
    });
  });
});

describe("HindsightRestClient.reflect", () => {
  it("POSTs the question to the reflect path with a low budget by default", async () => {
    const fetchFn = mockFetchOnce(200, { text: "You prefer tabs." });
    const client = new HindsightRestClient("http://localhost:8000", null);
    const res = await client.reflect("b", "what indentation do I use?");
    expect(res.text).toBe("You prefer tabs.");
    const [url, init] = fetchFn.mock.calls[0];
    expect(url).toBe("http://localhost:8000/v1/default/banks/b/reflect");
    expect(JSON.parse(init.body)).toEqual({ query: "what indentation do I use?", budget: "low" });
  });
});

describe("HindsightRestClient error handling", () => {
  it("throws on a non-2xx response", async () => {
    mockFetchOnce(401, { detail: "unauthorized" });
    const client = new HindsightRestClient("https://api.hindsight.vectorize.io", "bad");
    await expect(client.recall("b", "q")).rejects.toThrow(/HTTP 401/);
  });

  it("requires a non-empty base URL", () => {
    expect(() => new HindsightRestClient("   ")).toThrow(/API URL is required/);
  });

  it("passes the caller's abort signal to fetch and lets its abort through", async () => {
    const fetchFn = vi.fn(async (_url: string, init: { signal: AbortSignal }) => {
      if (init.signal.aborted) throw new DOMException("aborted", "AbortError");
      return { ok: true, status: 200, json: async () => ({ results: [] }), text: async () => "" };
    });
    vi.stubGlobal("fetch", fetchFn);
    const controller = new AbortController();
    controller.abort();
    const client = new HindsightRestClient("http://localhost:8000", null);
    await expect(client.recall("b", "q", { signal: controller.signal })).rejects.toThrow(/aborted/);
  });

  it("reports a timeout as an ordinary error naming the path", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        (_url: string, init: { signal: AbortSignal }) =>
          new Promise((_resolve, reject) => {
            init.signal.addEventListener("abort", () => reject(init.signal.reason));
          })
      )
    );
    const client = new HindsightRestClient("http://localhost:8000", null, 5);
    await expect(client.recall("b", "q")).rejects.toThrow(/timed out after 5ms/);
  });
});

describe("buildRecallMarkdown / stripSentinelBlocks", () => {
  it("returns empty string for no results", () => {
    expect(buildRecallMarkdown([])).toBe("");
  });

  it("wraps results in sentinel markers as a bulleted list", () => {
    const md = buildRecallMarkdown([
      { id: "1", text: "prefers Python" },
      { id: "2", text: "no comments" },
    ]);
    expect(md.startsWith(SENTINEL_OPEN)).toBe(true);
    expect(md.trimEnd().endsWith(SENTINEL_CLOSE)).toBe(true);
    expect(md).toContain("- prefers Python");
    expect(md).toContain("- no comments");
  });

  it("strips a fenced recalled-context block out of text", () => {
    const md = buildRecallMarkdown([{ id: "1", text: "secret" }]);
    const polluted = `Here is my answer.\n${md}\nDone.`;
    const cleaned = stripSentinelBlocks(polluted);
    expect(cleaned).not.toContain("secret");
    expect(cleaned).toContain("Here is my answer.");
    expect(cleaned).toContain("Done.");
  });
});
