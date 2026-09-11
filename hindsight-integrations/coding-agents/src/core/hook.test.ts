import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveConfig } from "./config";
import { buildHookOutput, runHook } from "./hook";
import { diagFilePath } from "./diag";
import { ReflectError } from "./hindsight";
import { buildReflectQuery } from "./inject";

let root: string;
let cacheFile: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-hook-"));
  cacheFile = join(root, "cache", "session.json");
});

afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(root, { recursive: true, force: true });
});

/** One page with a preamble and two headed sections. Pages are fetched/cached for the
 *  hindsight_search_knowledge_pages tool but are NEVER auto-injected into context. */
const PAGE_CONTENT =
  "Preamble prose about this page.\n\n" +
  "## Retry backoff\n" +
  "Uploads retry with exponential backoff and a 200ms jitter window.\n\n" +
  "## Auth tokens\n" +
  "Tokens rotate daily via the auth service.\n";

/** The header formatPageInjection uses — must NEVER appear in hook context anymore. */
const PAGES_HEADER = "Project knowledge from Hindsight, this repository's long-term memory";

/** A prompt overlapping the "Retry backoff" section (pages still must not inject). */
const MATCHING_PROMPT = "why does the upload retry backoff fail?";
/** A prompt matching nothing in the page. */
const UNRELATED_PROMPT = "completely unrelated banana smoothie question";

function makeClient(
  overrides: Partial<{
    reflect: (query: string, opts: { budget?: string; timeoutMs: number }) => Promise<string>;
    listPages: () => Promise<unknown>;
    getPage: (pageId: string) => Promise<unknown>;
    searchKnowledgePages: (
      query: string,
      limit: number,
      timeoutMs?: number
    ) => Promise<{ id: string; name: string; snippet: string }[]>;
    recallObservations: (
      query: string,
      opts: { maxTokens: number; timeoutMs: number }
    ) => Promise<string[]>;
  }> = {}
) {
  return {
    reflect: vi.fn(async () => "REFLECT_ANSWER"),
    listPages: vi.fn(async () => ({ items: [{ id: "p1", name: "Uploader guide" }] })),
    getPage: vi.fn(async () => ({ content: PAGE_CONTENT })),
    searchKnowledgePages: vi.fn(async () => [] as { id: string; name: string; snippet: string }[]),
    recallObservations: vi.fn(async () => [] as string[]),
    ...overrides,
  };
}

describe("buildHookOutput", () => {
  it("turn 1: injects the reflect answer wrapped in the system-injection preamble", async () => {
    const cfg = resolveConfig({});
    const client = makeClient();
    const result = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(result.context).toContain("Automatically retrieved by Hindsight");
    expect(result.context).toContain("REFLECT_ANSWER");
    expect(existsSync(cacheFile)).toBe(true);
    const cached = JSON.parse(readFileSync(cacheFile, "utf8"));
    expect(cached.turns).toBe(1);
    expect(cached.reflectAnswer).toBe("REFLECT_ANSWER");
  });

  it("reflect runs ONCE per session: cached on turn 1, not called again on turn 2", async () => {
    const cfg = resolveConfig({});
    const client = makeClient();
    const t1 = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    const t2 = await buildHookOutput({
      harness: "claude-code",
      prompt: "another prompt entirely",
      cfg,
      client,
      cacheFile,
    });
    expect(client.reflect).toHaveBeenCalledTimes(1);
    // Injected ONCE (hook context persists in the transcript — stacking it every turn would
    // duplicate the same block); turn 2 carries no repeat.
    expect(t1.context).toContain("REFLECT_ANSWER");
    expect(t2.context ?? "").not.toContain("REFLECT_ANSWER");
    expect(JSON.parse(readFileSync(cacheFile, "utf8")).turns).toBe(2);
  });

  it("defers the first automatic reflect for a new bank, then reflects on prompt two", async () => {
    mkdirSync(join(root, "cache"), { recursive: true });
    writeFileSync(cacheFile, JSON.stringify({ deferInitialReflect: true }));
    const cfg = resolveConfig({});
    const client = makeClient();

    const first = await buildHookOutput({
      harness: "claude-code",
      prompt: "first task while the bank is seeding",
      cfg,
      client,
      cacheFile,
    });
    expect(client.reflect).not.toHaveBeenCalled();
    expect(first.context).toBeUndefined();
    expect(JSON.parse(readFileSync(cacheFile, "utf8")).deferInitialReflect).toBeUndefined();

    const second = await buildHookOutput({
      harness: "claude-code",
      prompt: "second task after the bank has had time to populate",
      cfg,
      client,
      cacheFile,
    });
    expect(client.reflect).toHaveBeenCalledTimes(1);
    expect(second.context).toContain("REFLECT_ANSWER");
  });

  it("reflect rejection: caches '' (no retry next turn), no throw, no context at all", async () => {
    const cfg = resolveConfig({});
    const client = makeClient({
      reflect: vi.fn(async () => {
        throw new Error("reflect boom");
      }),
    });
    const t1 = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    // Reflect failed -> no reflect block; pages are never auto-injected -> nothing to inject.
    expect(t1.context).toBeUndefined();
    // ...but the turn is NOT silent: one line pointing at the diag trail (#3443).
    expect(t1.notice).toContain("no memory this turn");
    expect(t1.notice).toContain(diagFilePath());
    expect(JSON.parse(readFileSync(cacheFile, "utf8")).reflectAnswer).toBe("");

    await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    // The failure is cached as "" — reflect is NOT retried on the next turn.
    expect(client.reflect).toHaveBeenCalledTimes(1);
  });

  it("reflect_failed records the bank, the deadline, and the server's full error body", async () => {
    const diagFile = join(root, "diag.log");
    vi.stubEnv("HINDSIGHT_DIAG_FILE", diagFile);
    const body = `{"detail":"${"x".repeat(400)} upstream LLM rejected the turn"}`;
    const client = {
      ...makeClient({
        reflect: vi.fn(async () => {
          throw new Error(`reflect 500 ${body}`);
        }),
      }),
      bank: "coding-agent::demo",
    };
    await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg: resolveConfig({}),
      client,
      cacheFile,
    });

    const failed = readFileSync(diagFile, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line))
      .find((entry) => entry.event === "reflect_failed");
    expect(failed.bank).toBe("coding-agent::demo");
    expect(failed.timeoutMs).toBe(20_000);
    // Past describeError's default 200-char cut, where the root cause usually sits.
    expect(failed.error).toContain("upstream LLM rejected the turn");
  });

  it("the notice fires ONCE — the turn reflect failed, not on later turns", async () => {
    const cfg = resolveConfig({});
    const client = makeClient({
      reflect: vi.fn(async () => {
        throw new Error("reflect boom");
      }),
    });
    const args = { harness: "claude-code", prompt: UNRELATED_PROMPT, cfg, client, cacheFile };
    expect((await buildHookOutput(args)).notice).toContain("no memory this turn");
    // Turn 2 does not re-run reflect, so re-announcing a failure it did not observe would nag.
    expect((await buildHookOutput(args)).notice).toBeUndefined();
  });

  it("an EMPTY answer is not a failure: no notice (reflect simply had nothing to say)", async () => {
    const cfg = resolveConfig({});
    // The real client returns (data.text || "").trim() — a 200 with no text yields "".
    const client = makeClient({ reflect: vi.fn(async () => "") });
    const result = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(result.notice).toBeUndefined();
    expect(result.context).toBeUndefined();
  });

  it("uses a bounded low-budget reflect and caps its timeout at 20000ms", async () => {
    const cfg = resolveConfig({}); // reflectTimeoutMs default 120000
    const client = makeClient();
    await buildHookOutput({
      harness: "claude-code",
      prompt: "the prompt",
      cfg,
      client,
      cacheFile,
    });
    expect(client.reflect).toHaveBeenCalledWith(buildReflectQuery("the prompt"), {
      budget: "low",
      timeoutMs: 20000,
    });
  });

  it("uses the configured reflect timeout when it is below the 20s cap", async () => {
    const cfg = resolveConfig({ reflectTimeoutMs: 5000 });
    const client = makeClient();
    await buildHookOutput({
      harness: "claude-code",
      prompt: "the prompt",
      cfg,
      client,
      cacheFile,
    });
    expect(client.reflect).toHaveBeenCalledWith(buildReflectQuery("the prompt"), {
      budget: "low",
      timeoutMs: 5000,
    });
  });

  describe("reflect fallback (timeout / 5xx)", () => {
    const timedOut = () => new ReflectError("reflect timed out after 20000ms", undefined, true);
    const serverError = () => new ReflectError("reflect 502 bad gateway", 502, false);
    const HIT = { id: "kp-1", name: "Upload retries", snippet: "200ms  jitter\nwindow" };

    it("timeout -> injects matching knowledge pages, skips recall, caches it (no retry)", async () => {
      const client = makeClient({
        reflect: vi.fn(async () => {
          throw timedOut();
        }),
        searchKnowledgePages: vi.fn(async () => [HIT]),
      });
      const args = {
        harness: "claude-code",
        prompt: MATCHING_PROMPT,
        cfg: resolveConfig({}),
        client,
        cacheFile,
      };

      const t1 = await buildHookOutput(args);

      expect(client.searchKnowledgePages).toHaveBeenCalledWith(
        MATCHING_PROMPT,
        3,
        expect.any(Number)
      );
      expect(client.recallObservations).not.toHaveBeenCalled();
      expect(t1.context).toContain("<hindsight_memory>");
      expect(t1.context).toContain("- Upload retries (kp-1): 200ms jitter window");
      expect(t1.context).toContain("hindsight_read_knowledge_page");
      expect(t1.notice).toContain("reflect unavailable — fell back to 1 knowledge page ");
      expect(t1.notice).not.toContain("no memory this turn");

      const t2 = await buildHookOutput(args);
      expect(client.reflect).toHaveBeenCalledTimes(1);
      expect(client.searchKnowledgePages).toHaveBeenCalledTimes(1);
      // Injected once, on the turn it ran — like a reflect answer.
      expect(t2.context ?? "").not.toContain("Upload retries");
    });

    it("5xx with no matching page -> raw recall of observations is injected", async () => {
      const client = makeClient({
        reflect: vi.fn(async () => {
          throw serverError();
        }),
        recallObservations: vi.fn(async () => [
          "Retries back off exponentially.",
          "Tokens rotate daily.",
        ]),
      });

      const out = await buildHookOutput({
        harness: "claude-code",
        prompt: MATCHING_PROMPT,
        cfg: resolveConfig({}),
        client,
        cacheFile,
      });

      expect(client.searchKnowledgePages).toHaveBeenCalledTimes(1);
      expect(client.recallObservations).toHaveBeenCalledWith(MATCHING_PROMPT, {
        maxTokens: 2000,
        timeoutMs: expect.any(Number),
      });
      expect(out.context).toContain("consolidated observations");
      expect(out.context).toContain("- Retries back off exponentially.\n- Tokens rotate daily.");
      expect(out.notice).toContain("fell back to 2 observations");
    });

    it("a failing page search still falls through to observations", async () => {
      const client = makeClient({
        reflect: vi.fn(async () => {
          throw timedOut();
        }),
        searchKnowledgePages: vi.fn(async () => {
          throw new Error("knowledge pages unavailable");
        }),
        recallObservations: vi.fn(async () => ["An observation."]),
      });

      const out = await buildHookOutput({
        harness: "claude-code",
        prompt: MATCHING_PROMPT,
        cfg: resolveConfig({}),
        client,
        cacheFile,
      });

      expect(out.context).toContain("- An observation.");
    });

    it("both empty -> no context, the usual no-memory notice, failure cached", async () => {
      const client = makeClient({
        reflect: vi.fn(async () => {
          throw serverError();
        }),
      });

      const out = await buildHookOutput({
        harness: "claude-code",
        prompt: MATCHING_PROMPT,
        cfg: resolveConfig({}),
        client,
        cacheFile,
      });

      expect(client.recallObservations).toHaveBeenCalledTimes(1);
      expect(out.context).toBeUndefined();
      expect(out.notice).toContain("no memory this turn");
      expect(JSON.parse(readFileSync(cacheFile, "utf8")).reflectAnswer).toBe("");
    });

    it("4xx or a transport error does NOT fall back: every endpoint would fail the same way", async () => {
      for (const err of [new ReflectError("reflect 401", 401, false), new Error("fetch failed")]) {
        rmSync(cacheFile, { force: true });
        const client = makeClient({
          reflect: vi.fn(async () => {
            throw err;
          }),
        });
        const out = await buildHookOutput({
          harness: "claude-code",
          prompt: MATCHING_PROMPT,
          cfg: resolveConfig({}),
          client,
          cacheFile,
        });
        expect(client.searchKnowledgePages).not.toHaveBeenCalled();
        expect(client.recallObservations).not.toHaveBeenCalled();
        expect(out.notice).toContain("no memory this turn");
      }
    });

    it("records each fallback step in the diag trail", async () => {
      const diagFile = join(root, "diag.log");
      vi.stubEnv("HINDSIGHT_DIAG_FILE", diagFile);
      const client = makeClient({
        reflect: vi.fn(async () => {
          throw timedOut();
        }),
        recallObservations: vi.fn(async () => ["An observation."]),
      });

      await buildHookOutput({
        harness: "claude-code",
        prompt: MATCHING_PROMPT,
        cfg: resolveConfig({}),
        client,
        cacheFile,
      });

      const events = readFileSync(diagFile, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
      expect(events.find((e) => e.event === "reflect_fallback_pages")?.count).toBe(0);
      expect(events.find((e) => e.event === "reflect_fallback_observations")?.count).toBe(1);
    });
  });

  it("autoReflect false: never calls reflect, injects no memory block", async () => {
    const cfg = resolveConfig({ autoReflect: false });
    const client = makeClient();
    const out = await buildHookOutput({
      harness: "claude-code",
      prompt: "the prompt",
      cfg,
      client,
      cacheFile,
    });
    expect(client.reflect).not.toHaveBeenCalled();
    expect(out.context ?? "").not.toContain("<hindsight_memory>");
    // Tool-only mode's pull trigger: the roster refresh must carry the pages-first rule.
    const cfg2 = resolveConfig({ autoReflect: false, pageRefreshEveryTurns: 1 });
    const out2 = await buildHookOutput({
      harness: "claude-code",
      prompt: "next",
      cfg: cfg2,
      client,
      cacheFile,
    });
    expect(out2.context ?? "").toContain("NEW task or goal");
  });

  it("fetches the page ROSTER (ids + titles, no content) on the first turn; nothing injected from it", async () => {
    const cfg = resolveConfig({});
    const client = makeClient();
    const result = await buildHookOutput({
      harness: "claude-code",
      prompt: MATCHING_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    // The roster is fetched into the session cache (ids + titles only — content lives behind
    // the server-side search tool now, so getPage is never called here)…
    expect(client.listPages).toHaveBeenCalledTimes(1);
    expect(client.getPage).not.toHaveBeenCalled();
    // …but NO page content appears in context: the context is the reflect block only.
    expect(result.context).not.toContain(PAGES_HEADER);
    expect(result.context).not.toContain("Preamble prose about this page.");
    expect(result.context).not.toContain("200ms jitter window");
    expect(result.context).not.toContain("Auth tokens");
    expect(result.context).not.toContain("hindsight_read_knowledge_page p1");
    expect(result.context).toContain("Automatically retrieved by Hindsight");
    expect(result.context).toContain("REFLECT_ANSWER");
    // The roster is cached for the cadence-based refresh.
    const cached = JSON.parse(readFileSync(cacheFile, "utf8"));
    expect(cached.pages.atTurn).toBe(1);
    expect(cached.pages.list).toEqual([{ id: "p1", title: "Uploader guide" }]);
  });

  it("pages with content are NEVER auto-injected, even for an unrelated prompt", async () => {
    const cfg = resolveConfig({});
    const client = makeClient();
    const result = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(result.context).toContain("REFLECT_ANSWER"); // reflect still injected
    expect(result.context).not.toContain(PAGES_HEADER);
    expect(result.context).not.toContain("hindsight_read_knowledge_page p1");
  });

  it("empty page list -> no pages block", async () => {
    const cfg = resolveConfig({});
    const client = makeClient({ listPages: vi.fn(async () => ({ items: [] })) });
    const result = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(result.context).toContain("REFLECT_ANSWER"); // reflect still injected
    expect(result.context).not.toContain(PAGES_HEADER);
    expect(result.context).not.toContain("hindsight_read_knowledge_page");
  });

  it("injects the page-roster refresh only on cadence turns", async () => {
    const cfg = resolveConfig({ pageRefreshEveryTurns: 2 });
    const client = makeClient();
    // turn 1: not a multiple of 2 -> no refresh
    const t1 = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(t1.context).not.toContain("hindsight_knowledge_refresh");
    // turn 2: multiple of 2 -> refresh injected, listing the page roster
    const t2 = await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(t2.context).toContain("<hindsight_knowledge_refresh>");
    expect(t2.context).toContain("Uploader guide (p1)");
    // reflect block is NOT re-injected on cadence turns (injected once, on the reflect turn)
    expect(t2.context).not.toContain("REFLECT_ANSWER");
  });

  it("listPages rejection: no throw, reflect block still returned, turn still counted", async () => {
    const cfg = resolveConfig({});
    const client = makeClient({
      listPages: vi.fn(async () => {
        throw new Error("listPages boom");
      }),
    });
    const result = await buildHookOutput({
      harness: "claude-code",
      prompt: MATCHING_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(result.context).toContain("REFLECT_ANSWER");
    expect(result.context).not.toContain(PAGES_HEADER);
    expect(JSON.parse(readFileSync(cacheFile, "utf8")).turns).toBe(1);
  });

  it("persists and increments the turn counter each call", async () => {
    const cfg = resolveConfig({ pageRefreshEveryTurns: 3 });
    const client = makeClient();
    for (let n = 1; n <= 4; n++) {
      await buildHookOutput({
        harness: "claude-code",
        prompt: `turn ${n}`,
        cfg,
        client,
        cacheFile,
      });
      const cached = JSON.parse(readFileSync(cacheFile, "utf8")) as { turns?: number };
      expect(cached.turns).toBe(n);
    }
  });

  it("re-fetches pages every pageRefreshEveryTurns turns, serving the cache in between", async () => {
    const cfg = resolveConfig({ pageRefreshEveryTurns: 2 });
    const client = makeClient();
    // Seed the cache so this session already has pages from turn 1.
    mkdirSync(join(root, "cache"), { recursive: true });
    writeFileSync(
      cacheFile,
      JSON.stringify({
        turns: 1,
        reflectAnswer: "CACHED",
        pages: { atTurn: 1, list: [{ id: "p1", title: "Uploader guide", content: PAGE_CONTENT }] },
      })
    );
    // turn 2: 2 - 1 < 2 -> cache is fresh, no fetch
    await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(client.listPages).not.toHaveBeenCalled();
    // turn 3: 3 - 1 >= 2 -> stale, re-fetched
    await buildHookOutput({
      harness: "claude-code",
      prompt: UNRELATED_PROMPT,
      cfg,
      client,
      cacheFile,
    });
    expect(client.listPages).toHaveBeenCalledTimes(1);
  });
});

describe("runHook anti-recursion guard", () => {
  const ORIGINAL = process.env.HINDSIGHT_DISABLE_HOOKS;

  afterEach(() => {
    if (ORIGINAL === undefined) delete process.env.HINDSIGHT_DISABLE_HOOKS;
    else process.env.HINDSIGHT_DISABLE_HOOKS = ORIGINAL;
  });

  it("HINDSIGHT_DISABLE_HOOKS set -> returns immediately, never reads stdin or builds a client", async () => {
    process.env.HINDSIGHT_DISABLE_HOOKS = "1";
    const makeClient = vi.fn();
    // No stdin is provided/mocked here — if the guard didn't return before `readFileSync(0, ...)`,
    // this call would attempt to read the real process stdin. The fact this resolves at all (let
    // alone without calling makeClient) proves the guard fired first.
    await runHook({ harness: "claude-code", parse: () => ({}), emit: (c) => ({ c }) }, makeClient);
    expect(makeClient).not.toHaveBeenCalled();
  });
});
