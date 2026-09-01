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

describe("HindsightClient document-list safety", () => {
  it("uses strict strategy-tag matching on every page", async () => {
    const client = new HindsightClient({ apiUrl: "http://x", bank: "shared-bank" });
    const firstPage = Array.from({ length: 500 }, (_, i) => ({ id: `git:${i}` }));
    const fetchMock = vi.fn(async (_url: string | URL | Request) => {
      const offset = String(_url).includes("offset=500") ? 500 : 0;
      return jsonResponse(200, {
        items: offset === 0 ? firstPage : [{ id: "git:500" }],
        total: 501,
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const ids = await client.listDocumentIds("source:git", "all_strict");

    expect(ids.size).toBe(501);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[0][0])).toBe(
      "http://x/v1/default/banks/shared-bank/documents?tags=source%3Agit&tags_match=all_strict&limit=500&offset=0"
    );
    expect(String(fetchMock.mock.calls[1][0])).toBe(
      "http://x/v1/default/banks/shared-bank/documents?tags=source%3Agit&tags_match=all_strict&limit=500&offset=500"
    );
  });

  it("preserves the inclusive all mode for existing callers that do not opt into strict matching", async () => {
    const client = new HindsightClient({ apiUrl: "http://x", bank: "shared-bank" });
    const fetchMock = vi.fn(async (_url: string | URL | Request) =>
      jsonResponse(200, { items: [], total: 0 })
    );
    vi.stubGlobal("fetch", fetchMock);

    await client.listDocumentIds("custom:scope");

    expect(String(fetchMock.mock.calls[0][0])).toContain(
      "tags=custom%3Ascope&tags_match=all&limit=500&offset=0"
    );
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
  async function retainItem(
    client: HindsightClient,
    tags: string[] = ["source:chat", "harness:claude-code"]
  ): Promise<Record<string, unknown>> {
    let sent: string | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        sent = String(init.body);
        return jsonResponse(200, { operation_id: "op-1" });
      })
    );
    await client.retain("c", "ctx", "doc-1", tags, "conversation");
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
 * `per_source` is the one scoping a static config cannot express. The server treats an explicit
 * scope list as UNCONDITIONAL — `_resolve_obs_tags_list` returns it verbatim without filtering
 * against the memory's own tags — so configuring `[[], ["source:git"], ["source:chat"]]` writes
 * every document into all three, and the `source:git` scope fills with beliefs built from chat
 * transcripts. Deriving the scope per document from its own `source:` tag is the only way to get
 * "what the commits say" apart from "what was discussed" while keeping the merged global set.
 *
 * It reads ONLY `source:`, so volatile provenance tags (a session id in `retainTags`) can never
 * become a scope — the failure mode `per_tag` would reintroduce.
 */
describe("HindsightClient.retain — per_source scoping", () => {
  async function retainItem(
    client: HindsightClient,
    tags: string[]
  ): Promise<Record<string, unknown>> {
    let sent: string | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        sent = String(init.body);
        return jsonResponse(200, { operation_id: "op-1" });
      })
    );
    await client.retain("c", "ctx", "doc-1", tags, "conversation");
    const body = JSON.parse(String(sent)) as { items: Record<string, unknown>[] };
    return body.items[0];
  }

  const perSource = () =>
    new HindsightClient({ apiUrl: "http://x", bank: "b", observationScopes: "per_source" });

  it("keeps the global scope and adds the document's own source scope", async () => {
    const item = await retainItem(perSource(), ["source:chat", "harness:claude-code"]);
    expect(item.observation_scopes).toEqual([[], ["source:chat"]]);
  });

  it("scopes a git document apart from a chat one", async () => {
    const item = await retainItem(perSource(), ["source:git", "harness:claude-code"]);
    expect(item.observation_scopes).toEqual([[], ["source:git"]]);
  });

  it("falls back to the global scope alone when a document carries no source tag", async () => {
    const item = await retainItem(perSource(), ["knowledge:convention"]);
    expect(item.observation_scopes).toEqual([[]]);
  });

  // The commit-message seed carries `source:git` AND `source:git-log` (git.ts keeps
  // both so the cold-repo check can find it), so it writes to both scopes. That is
  // not duplication: `source:git-log` is fed only by the seed — what the commit
  // MESSAGES say — while `source:git` also collects every per-commit diff under
  // gitIngest: "full". Two questions, two answers, each deduplicated within itself.
  // A fact belonging to more than one axis is the design working, not a leak.
  it("gives a document carrying two source tags a scope for each", async () => {
    const item = await retainItem(perSource(), ["source:git", "source:git-log", "gitlog-head:abc"]);
    expect(item.observation_scopes).toEqual([[], ["source:git"], ["source:git-log"]]);
  });

  it("orders the scopes independently of the order the tags arrive in", async () => {
    const item = await retainItem(perSource(), ["source:git-log", "source:git"]);
    expect(item.observation_scopes).toEqual([[], ["source:git"], ["source:git-log"]]);
  });

  it("never lets a volatile provenance tag become a scope", async () => {
    const item = await retainItem(perSource(), [
      "source:chat",
      "hermes-session:20260829_101500_abc",
    ]);
    expect(item.observation_scopes).toEqual([[], ["source:chat"]]);
  });
});

/**
 * The scoping default lives in the client, so an entrypoint that forgets to forward the config
 * fails SOFTLY — it keeps writing correct memories and just ignores the user's `observationScopes`.
 * Nothing would notice, and the next harness added would copy the site that forgot. So assert it
 * over the whole family instead of per entrypoint, the way daemon.test.ts guards `ensureDaemon`.
 */
/**
 * #3600: the client captured `apiToken` at construction, so a long-lived host (dsh, Cline, Kilo,
 * Prime Agent, the MCP server) kept signing with a credential the operator had already replaced —
 * every call 401'd until the whole host restarted, while `hindsight_diagnose` read the file and
 * called it healthy.
 */
describe("HindsightClient credential refresh", () => {
  /** Stub fetch that only accepts one bearer token, and records what it was asked. */
  function server(accepted: () => string | undefined) {
    const calls: { auth: string | null; body: string | null }[] = [];
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const auth = new Headers(init.headers).get("Authorization");
      calls.push({ auth, body: (init.body as string) ?? null });
      const want = accepted();
      if (auth !== (want ? `Bearer ${want}` : null))
        return jsonResponse(401, { detail: "Authentication failed: Invalid API key" });
      return jsonResponse(200, { ok: true });
    });
    vi.stubGlobal("fetch", fetchMock);
    return { calls, fetchMock };
  }

  it("recovers from a rotated credential without restarting the host", async () => {
    const { calls } = server(() => "new-key");
    let onDisk = "old-key";
    const client = new HindsightClient({
      apiUrl: "http://x",
      bank: "b",
      apiToken: onDisk,
      tokenProvider: () => onDisk,
    });

    onDisk = "new-key"; // the operator rotates the key while the host keeps running
    await client.req("GET", "http://x/thing");

    expect(calls.map((c) => c.auth)).toEqual(["Bearer old-key", "Bearer new-key"]);
    expect(client.apiToken).toBe("new-key");
  });

  it("does not retry when the credential is unchanged — a wrong key stays one 401, not two", async () => {
    const { calls } = server(() => "right-key");
    const client = new HindsightClient({
      apiUrl: "http://x",
      bank: "b",
      apiToken: "wrong-key",
      tokenProvider: () => "wrong-key",
    });

    await expect(client.req("GET", "http://x/thing")).rejects.toThrow(/401/);
    expect(calls).toHaveLength(1);
  });

  it("keeps the last credential that worked when the provider throws", async () => {
    const { calls } = server(() => "any");
    const client = new HindsightClient({
      apiUrl: "http://x",
      bank: "b",
      apiToken: "good-key",
      // A half-written config file must not leave the client with no credential at all.
      tokenProvider: () => {
        throw new Error("unparseable config");
      },
    });

    await expect(client.req("GET", "http://x/thing")).rejects.toThrow(/401/);
    expect(calls).toHaveLength(1);
    expect(client.apiToken).toBe("good-key");
  });

  it("leaves a provider-less client exactly as it was", async () => {
    const { calls } = server(() => "any");
    const client = new HindsightClient({ apiUrl: "http://x", bank: "b", apiToken: "stale" });

    await expect(client.req("GET", "http://x/thing")).rejects.toThrow(/401/);
    expect(calls).toHaveLength(1);
  });

  it("replays the body of a retried POST", async () => {
    const { calls } = server(() => "new-key");
    let onDisk = "old-key";
    const client = new HindsightClient({
      apiUrl: "http://x",
      bank: "b",
      apiToken: onDisk,
      tokenProvider: () => onDisk,
    });

    onDisk = "new-key";
    await client.req("POST", "http://x/thing", { hello: "world" });

    expect(calls.map((c) => c.body)).toEqual([
      JSON.stringify({ hello: "world" }),
      JSON.stringify({ hello: "world" }),
    ]);
  });

  // reflect() and the drain poll used to fetch directly, so a recovery wired only into req() would
  // have left them failing forever — which is how the bug read: hooks worked, tools did not.
  it("recovers on the reflect path too, not just the generic request path", async () => {
    const { calls } = server(() => "new-key");
    let onDisk = "old-key";
    const client = new HindsightClient({
      apiUrl: "http://x",
      bank: "b",
      apiToken: onDisk,
      tokenProvider: () => onDisk,
    });

    onDisk = "new-key";
    await client.reflect("why?", { timeoutMs: 5_000 });
    expect(calls.map((c) => c.auth)).toEqual(["Bearer old-key", "Bearer new-key"]);
  });

  it("says whether a credential was even sent, which the server's 401 cannot", async () => {
    server(() => "needed");
    const none = new HindsightClient({ apiUrl: "http://x", bank: "b" });
    await expect(none.req("GET", "http://x/thing")).rejects.toThrow(/no apiToken is configured/);

    const wrong = new HindsightClient({ apiUrl: "http://x", bank: "b", apiToken: "nope" });
    await expect(wrong.req("GET", "http://x/thing")).rejects.toThrow(/was rejected/);
  });
});

/**
 * The other half of the same invariant: forwarding the config is worthless if a write path skips
 * the client method that SENDS it. One bank must consolidate into exactly one observation scope
 * (#3564), and `retain()` is the only place that puts `observation_scopes` on the wire — so a
 * second `/memories` POST anywhere would quietly consolidate under the server's `combined`
 * default, splitting the repo's beliefs per tag combination again. No unit test would fail: the
 * new path writes perfectly good memories. Hence a check over the whole source tree.
 */
describe("every memory write goes through the one call site that scopes it", () => {
  const SRC = fileURLToPath(new URL("..", import.meta.url));

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  it("has no module addressing the memories endpoint except the client", () => {
    const writers = sourceFiles(SRC).filter((rel) =>
      readFileSync(join(SRC, rel), "utf8").includes('"/memories')
    );
    expect(writers).toEqual(["core/hindsight.ts"]);
  });

  it("keeps that call site inside retain(), with the scoping on the item it posts", () => {
    const src = readFileSync(join(SRC, "core/hindsight.ts"), "utf8");
    expect(src.match(/bankUrl\("\/memories"\)/g)).toHaveLength(1);
    // Everything between retain()'s signature and the POST is the body it builds; the scoping
    // has to be set in there, not left to whatever the server defaults to.
    const body = src.slice(src.indexOf("async retain("), src.indexOf('bankUrl("/memories")'));
    // The scoping may be derived per document (see `per_source`), but it must still be set on the
    // item here and still come from the configured value — not from a server default.
    expect(body).toMatch(/observation_scopes: .*this\.observationScopes/);
  });
});

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
