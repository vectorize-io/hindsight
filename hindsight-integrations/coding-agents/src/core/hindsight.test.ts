import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_MAX_PARALLEL_RETAINS,
  DEFAULT_OBSERVATION_SCOPES,
  HindsightClient,
  retryAfterMs,
} from "./hindsight";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

function jsonResponse(status: number, body: unknown, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...(headers ?? {}) },
  });
}

describe("HindsightClient.maxParallelRetains", () => {
  it("defaults to 10 when not provided", () => {
    const c = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    expect(c.maxParallelRetains).toBe(DEFAULT_MAX_PARALLEL_RETAINS);
  });

  it("honours the configured cap", () => {
    const c = new HindsightClient({ apiUrl: "http://x", bank: "b", maxParallelRetains: 3 });
    expect(c.maxParallelRetains).toBe(3);
  });
});

describe("HindsightClient.drain", () => {
  it("polls at most maxParallelRetains ops concurrently", async () => {
    const cap = 2;
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b", maxParallelRetains: cap });
    let inFlight = 0;
    let maxInFlight = 0;
    const fetchMock = vi.fn(async () => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight--;
      return jsonResponse(200, { status: "completed" });
    });
    vi.stubGlobal("fetch", fetchMock);

    await client.drain(["1", "2", "3", "4", "5"], "test", 10_000);

    expect(maxInFlight).toBeLessThanOrEqual(cap);
    expect(maxInFlight).toBe(cap); // 5 ids against a 2-wide pool must actually hit the cap
    expect(fetchMock).toHaveBeenCalledTimes(5);
  });

  it("backs off by Retry-After when it exceeds the 10s floor", async () => {
    vi.useFakeTimers();
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    const fetchMock = vi.fn(async () => jsonResponse(429, {}, { "Retry-After": "30" }));
    vi.stubGlobal("fetch", fetchMock);

    const p = client.drain(["1"], "test", 60_000);
    await vi.advanceTimersByTimeAsync(0); // first cycle completes → sleep 30s
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(10_000); // floor elapsed, but the header says 30s
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(20_000); // 30s elapsed → next cycle
    expect(fetchMock).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(60_000); // run out the 60-min cap
    await p;
  });

  it("caps the backoff however long Retry-After asks for", async () => {
    // The header is a hint, not a budget we owe the server: an hour-long value would otherwise
    // park the drain — and the background seed behind it — for that hour.
    vi.useFakeTimers();
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    const fetchMock = vi.fn(async () => jsonResponse(429, {}, { "Retry-After": "3600" }));
    vi.stubGlobal("fetch", fetchMock);

    const p = client.drain(["1"], "test", 300_000);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(60_000); // capped at 60s, not 3600s
    expect(fetchMock).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(300_000);
    await p;
  });

  it("uses the 10s floor when Retry-After is shorter than it", async () => {
    vi.useFakeTimers();
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    const fetchMock = vi.fn(async () => jsonResponse(429, {}, { "Retry-After": "2" }));
    vi.stubGlobal("fetch", fetchMock);

    const p = client.drain(["1"], "test", 60_000);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2_000); // Retry-After=2s would have fired by now
    expect(fetchMock).toHaveBeenCalledTimes(1); // the 10s floor keeps us waiting
    await vi.advanceTimersByTimeAsync(8_000);
    expect(fetchMock).toHaveBeenCalledTimes(2); // floor elapsed → next cycle
    await vi.advanceTimersByTimeAsync(60_000);
    await p;
  });

  it("backs off 10s on a 429 that carries no Retry-After", async () => {
    vi.useFakeTimers();
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    const fetchMock = vi.fn(async () => jsonResponse(429, {}));
    vi.stubGlobal("fetch", fetchMock);

    const p = client.drain(["1"], "test", 60_000);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5_000); // the old 5s cycle would fire here
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(60_000);
    await p;
  });

  it("keeps the 5s cycle when ops stay pending without a 429", async () => {
    vi.useFakeTimers();
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    const fetchMock = vi.fn(async () => jsonResponse(200, { status: "running" }));
    vi.stubGlobal("fetch", fetchMock);

    const p = client.drain(["1"], "test", 60_000);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(fetchMock).toHaveBeenCalledTimes(2); // 5s cycle preserved
    await vi.advanceTimersByTimeAsync(60_000);
    await p;
  });

  it("marks non-completed terminal ops as failed and drops them from polling", async () => {
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    const fetchMock = vi.fn(async (url: string | URL | Request) => {
      const id = String(url).split("/").pop();
      return id === "1"
        ? jsonResponse(200, { status: "failed" })
        : jsonResponse(200, { status: "completed" });
    });
    vi.stubGlobal("fetch", fetchMock);
    const log = vi.fn();
    const c = new HindsightClient({ apiUrl: "http://x", bank: "b", log });

    await c.drain(["1", "2"], "test", 10_000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(log).toHaveBeenCalledWith("[wait] test drained — 2 done, 1 failed");
  });
});

describe("retryAfterMs", () => {
  it("parses delta-seconds", () => {
    expect(retryAfterMs("30")).toBe(30_000);
    expect(retryAfterMs(" 2 ")).toBe(2_000);
  });

  it("parses an HTTP-date into a bounded delay", () => {
    const future = new Date(Date.now() + 60_000).toUTCString();
    const ms = retryAfterMs(future);
    expect(ms).toBeGreaterThan(50_000);
    expect(ms).toBeLessThanOrEqual(60_000);
  });

  it("returns 0 for absent or unparseable values", () => {
    expect(retryAfterMs(null)).toBe(0);
    expect(retryAfterMs(undefined)).toBe(0);
    expect(retryAfterMs("")).toBe(0);
    expect(retryAfterMs("soon")).toBe(0);
  });
});

describe("HindsightClient.retain — observation scoping", () => {
  async function retainItem(client: HindsightClient): Promise<Record<string, unknown>> {
    let sent: string | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        sent = String(init.body);
        return jsonResponse(200, { operation_id: "op-1" });
      })
    );
    await client.retain(
      "c",
      "ctx",
      "doc-1",
      ["source:chat", "harness:claude-code"],
      "conversation"
    );
    const body = JSON.parse(String(sent)) as { items: Record<string, unknown>[] };
    return body.items[0];
  }

  it("defaults every retain to the single global scope, so two agents on one repo build ONE set of observations (#3564)", async () => {
    expect(DEFAULT_OBSERVATION_SCOPES).toBe("shared");
    const item = await retainItem(new HindsightClient({ apiUrl: "http://x", bank: "b" }));
    expect(item.observation_scopes).toBe("shared");
    // The harness tag still travels: it is what the documents list filters and draws its logo from.
    expect(item.tags).toEqual(["source:chat", "harness:claude-code"]);
  });

  it("sends a configured scoping instead, including the server's own default", async () => {
    const combined = await retainItem(
      new HindsightClient({ apiUrl: "http://x", bank: "b", observationScopes: "combined" })
    );
    expect(combined.observation_scopes).toBe("combined");
    const explicit = await retainItem(
      new HindsightClient({ apiUrl: "http://x", bank: "b", observationScopes: [["project:demo"]] })
    );
    expect(explicit.observation_scopes).toEqual([["project:demo"]]);
  });
});

/**
 * The scoping default lives in the client, so an entrypoint that forgets to forward the config
 * fails SOFTLY — it keeps writing correct memories and just ignores the user's `observationScopes`.
 * Nothing would notice, and the next harness added would copy the site that forgot. So assert it
 * over the whole family instead of per entrypoint, the way daemon.test.ts guards `ensureDaemon`.
 */
describe("every client-building entrypoint forwards observationScopes", () => {
  const SRC = fileURLToPath(new URL("..", import.meta.url));

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  it("has no module that builds a client without passing cfg.observationScopes", () => {
    const dropped = sourceFiles(SRC).filter((rel) => {
      const src = readFileSync(join(SRC, rel), "utf8");
      // `makeClient({` is the hook/session-start seam: the ClientOpts are built there even though
      // the constructor call itself is the injected default further up the file.
      const buildsClient = src.includes("new HindsightClient({") || src.includes("makeClient({");
      return buildsClient && !src.includes("observationScopes:");
    });
    expect(dropped).toEqual([]);
  });
});
