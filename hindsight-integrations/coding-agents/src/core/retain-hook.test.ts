import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { deriveBankId } from "./bank";
import { type RawConfig, resolveConfig } from "./config";
import type { HindsightClient } from "./hindsight";
import { buildRetain, runRetainHook } from "./retain-hook";
import { memoryCursorStore, type RetainCursorStore } from "./retain-cursor";
import { dcodeAssistantText } from "./transcript-dcode";

/** The Stop event `runRetainHook` reads from fd 0; every other read stays real. */
let stdin = "";
vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return {
    ...actual,
    readFileSync: (target: unknown, ...rest: unknown[]) =>
      target === 0 ? stdin : (actual.readFileSync as (...a: unknown[]) => unknown)(target, ...rest),
  };
});

/** What the hook's `loadConfig` returns — the real resolver over a per-test raw config, so bank
 *  overrides and defaults behave exactly as they do against a real config file. */
let rawConfig: RawConfig = {};
vi.mock("./config", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./config")>();
  return { ...actual, loadConfig: () => actual.resolveConfig(rawConfig) };
});

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-retain-hook-"));
  file = join(root, "session.jsonl");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("buildRetain", () => {
  /** Turns retained by one buildRetain call, in order. */
  async function retainedTurns(
    args: Parameters<typeof buildRetain>[0] & { retainSpy: ReturnType<typeof vi.fn> }
  ): Promise<Array<{ role: string; content: string }>> {
    const { retainSpy, ...rest } = args;
    await buildRetain({ ...rest, client: { retain: retainSpy } as unknown as HindsightClient });
    const [content] = retainSpy.mock.calls[0];
    return (content as string).split("\n").map((line) => JSON.parse(line));
  }

  it("recovers Dcode's final assistant message when the materialized transcript lags", async () => {
    writeFileSync(file, JSON.stringify({ role: "user", content: "make the change" }));
    const parsed = await retainedTurns({
      harness: "dcode",
      sessionId: "sess-dcode",
      transcriptPath: file,
      client: undefined as never,
      retainSpy: vi.fn().mockResolvedValue(undefined),
      readTranscript: () => [{ role: "user", content: "make the change" }],
      lastAssistantMessage: "done <hindsight_memories>injected</hindsight_memories>",
    });
    expect(parsed.at(-1)).toMatchObject({ role: "assistant", content: "done" });
  });

  it("does not duplicate the final reply when Dcode's transcript already flushed it", async () => {
    // The branch the lag recovery has to not break: on runs where the store flushed in time the
    // reply is in BOTH the transcript and the Stop event, and must be retained exactly once.
    writeFileSync(file, JSON.stringify({ role: "user", content: "make the change" }));
    const parsed = await retainedTurns({
      harness: "dcode",
      sessionId: "sess-dcode",
      transcriptPath: file,
      client: undefined as never,
      retainSpy: vi.fn().mockResolvedValue(undefined),
      readTranscript: () => [
        { role: "user", content: "make the change" },
        { role: "assistant", content: "done" },
      ],
      lastAssistantMessage: "done",
    });
    expect(parsed.filter((t) => t.role === "assistant")).toHaveLength(1);
  });

  it("dedupes a flushed reply the harness serialized as content blocks", async () => {
    // Regression: Dcode sends `str(content)`, so without readLastMessage the Stop event's copy can
    // never compare equal to the transcript's clean text and a duplicate is appended every turn.
    writeFileSync(file, JSON.stringify({ role: "user", content: "make the change" }));
    const parsed = await retainedTurns({
      harness: "dcode",
      sessionId: "sess-dcode",
      transcriptPath: file,
      client: undefined as never,
      retainSpy: vi.fn().mockResolvedValue(undefined),
      readTranscript: () => [
        { role: "user", content: "make the change" },
        { role: "assistant", content: "done" },
      ],
      lastAssistantMessage: "[{'type': 'text', 'text': 'done'}]",
      readLastMessage: dcodeAssistantText,
    });
    expect(parsed.filter((t) => t.role === "assistant")).toHaveLength(1);
  });

  it("retains the recovered reply as text, not as a serialized block list", async () => {
    writeFileSync(file, JSON.stringify({ role: "user", content: "make the change" }));
    const parsed = await retainedTurns({
      harness: "dcode",
      sessionId: "sess-dcode",
      transcriptPath: file,
      client: undefined as never,
      retainSpy: vi.fn().mockResolvedValue(undefined),
      readTranscript: () => [{ role: "user", content: "make the change" }],
      lastAssistantMessage:
        "[{'type': 'reasoning', 'encrypted_content': 'gAAAAAsecret'}, " +
        "{'type': 'text', 'text': 'done'}]",
      readLastMessage: dcodeAssistantText,
    });
    expect(parsed.at(-1)).toMatchObject({ role: "assistant", content: "done" });
  });

  it("retains parsed turns", async () => {
    const lines = [
      JSON.stringify({
        type: "user",
        timestamp: "2026-01-01T00:00:00Z",
        message: { role: "user", content: "we use zod for validation" },
      }),
      JSON.stringify({
        type: "assistant",
        timestamp: "2026-01-01T00:00:01Z",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "noted, zod it is" }],
        },
      }),
    ];
    writeFileSync(file, lines.join("\n"));

    const retainSpy = vi.fn().mockResolvedValue(undefined);
    const client = { retain: retainSpy } as unknown as HindsightClient;

    await buildRetain({
      harness: "claude-code",
      sessionId: "sess-1",
      transcriptPath: file,
      client,
    });

    expect(retainSpy).toHaveBeenCalledTimes(1);
    const [content, , documentId, tags, strategy] = retainSpy.mock.calls[0];
    expect(documentId).toBe("conversation:sess-1");
    // A JSONL transcript (renderSessionJsonl): one {role, content, timestamp} object per line,
    // led by the REF-ID system turn.
    const parsed = (content as string)
      .split("\n")
      .map((line) => JSON.parse(line) as { role: string; content: string });
    expect(parsed[0]).toMatchObject({ role: "system", content: "REF-ID: conversation:sess-1" });
    expect(parsed[1]).toMatchObject({ role: "user", content: "we use zod for validation" });
    expect(parsed[2]).toMatchObject({ role: "assistant", content: "noted, zod it is" });
    // Verbose `session` extraction, not the ≤2-fact `chat` extractor.
    expect(strategy).toBe("conversation");
    expect(tags).toEqual(["source:chat", "harness:claude-code"]);
  });

  it("empty transcript -> no retain", async () => {
    const lines = [
      // isMeta line: dropped
      JSON.stringify({
        type: "user",
        isMeta: true,
        message: { role: "user", content: "<system-injected>" },
      }),
      // non-message summary line: dropped
      JSON.stringify({ type: "summary", summary: "…" }),
    ];
    writeFileSync(file, lines.join("\n"));

    const retainSpy = vi.fn().mockResolvedValue(undefined);
    const client = { retain: retainSpy } as unknown as HindsightClient;

    await buildRetain({
      harness: "claude-code",
      sessionId: "sess-2",
      transcriptPath: file,
      client,
    });

    expect(retainSpy).not.toHaveBeenCalled();
  });

  it("fails open on retain error", async () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        timestamp: "2026-01-01T00:00:00Z",
        message: { role: "user", content: "hello" },
      })
    );

    const retainSpy = vi.fn().mockRejectedValue(new Error("boom"));
    const client = { retain: retainSpy } as unknown as HindsightClient;

    await expect(
      buildRetain({
        harness: "claude-code",
        sessionId: "sess-3",
        transcriptPath: file,
        client,
      })
    ).resolves.toBeUndefined();
  });
});

describe("runRetainHook anti-recursion guard", () => {
  const ORIGINAL = process.env.HINDSIGHT_DISABLE_HOOKS;

  afterEach(() => {
    if (ORIGINAL === undefined) delete process.env.HINDSIGHT_DISABLE_HOOKS;
    else process.env.HINDSIGHT_DISABLE_HOOKS = ORIGINAL;
  });

  it("HINDSIGHT_DISABLE_HOOKS set -> returns immediately, never reads stdin or builds a client", async () => {
    process.env.HINDSIGHT_DISABLE_HOOKS = "1";
    const makeClient = vi.fn();
    // No stdin is provided/mocked here — if the guard didn't return before `readFileSync(0, ...)`,
    // this call would attempt to read the real process stdin. Resolving without calling makeClient
    // proves the guard fired first.
    await runRetainHook(
      { harness: "claude-code", hostTimeoutSec: 60, parse: () => ({}) },
      makeClient
    );
    expect(makeClient).not.toHaveBeenCalled();
  });
});

describe("buildRetain — incremental write-back across Stop hooks", () => {
  const line = (i: number) =>
    JSON.stringify({
      type: "user",
      timestamp: `2026-01-01T00:00:0${i}Z`,
      message: { role: "user", content: `turn ${i}` },
    });

  /** One Stop-hook invocation over the transcript as it stands. Each hook run is a fresh process in
   *  production, so only the cursor store carries state between these calls. */
  const stop = async (client: HindsightClient, cursors: RetainCursorStore) =>
    buildRetain({
      harness: "codex",
      sessionId: "sess-append",
      transcriptPath: file,
      client,
      cursors,
    });

  const stubClient = () => {
    const retain = vi.fn().mockResolvedValue(undefined);
    return {
      retain,
      client: {
        retain,
        bank: "coding-agent::repo",
        supportsIdempotentRetain: async () => true,
      } as unknown as HindsightClient,
    };
  };

  it("sends the whole session once, then only the turns the session grew by", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();

    writeFileSync(file, [line(0), line(1)].join("\n"));
    await stop(client, cursors);

    writeFileSync(file, [line(0), line(1), line(2)].join("\n"));
    await stop(client, cursors);

    expect(retain).toHaveBeenCalledTimes(2);
    // First write carries the REF-ID header plus both turns; the second carries turn 2 alone.
    expect((retain.mock.calls[0][0] as string).split("\n")).toHaveLength(3);
    expect(retain.mock.calls[0][5].updateMode).toBeUndefined();
    const appended = (retain.mock.calls[1][0] as string).split("\n");
    expect(appended).toHaveLength(1);
    expect(JSON.parse(appended[0])).toMatchObject({ role: "user", content: "turn 2" });
    expect(retain.mock.calls[1][5].updateMode).toBe("append");
  });

  it("does not write again when the session ended without new turns", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();
    writeFileSync(file, [line(0), line(1)].join("\n"));
    await stop(client, cursors);
    await stop(client, cursors);
    expect(retain).toHaveBeenCalledTimes(1);
  });

  it("replaces the document when the transcript was rewritten rather than extended", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();
    writeFileSync(file, [line(0), line(1), line(2)].join("\n"));
    await stop(client, cursors);

    // Compaction: earlier turns replaced by a summary, then the session continues.
    writeFileSync(
      file,
      [
        JSON.stringify({
          type: "user",
          timestamp: "2026-01-01T00:00:09Z",
          message: { role: "user", content: "summary of the work so far" },
        }),
        line(3),
      ].join("\n")
    );
    await stop(client, cursors);

    expect(retain.mock.calls[1][5].updateMode).toBeUndefined();
    const rewritten = (retain.mock.calls[1][0] as string).split("\n");
    expect(rewritten).toHaveLength(3); // REF-ID + the two turns that now exist
    expect(JSON.parse(rewritten[1])).toMatchObject({ content: "summary of the work so far" });
  });

  it("a failed write is not silently skipped by the next one — it replaces", async () => {
    const { retain, client } = stubClient();
    const cursors = memoryCursorStore();
    writeFileSync(file, [line(0), line(1)].join("\n"));
    await stop(client, cursors);

    retain.mockRejectedValueOnce(new Error("server unreachable"));
    writeFileSync(file, [line(0), line(1), line(2)].join("\n"));
    await stop(client, cursors); // buildRetain swallows the failure by design

    writeFileSync(file, [line(0), line(1), line(2), line(3)].join("\n"));
    await stop(client, cursors);

    expect(retain).toHaveBeenCalledTimes(3);
    expect(retain.mock.calls[2][5].updateMode).toBeUndefined();
    // Everything the failed append would have carried is back in the replaced document.
    expect((retain.mock.calls[2][0] as string).split("\n")).toHaveLength(5);
  });
});

/**
 * The `retainSessions: false` opt-out (#3596): the flag was parsed and env-mapped but no Stop-hook
 * path ever read it, so hook harnesses wrote every transcript back regardless. These drive the
 * real `runRetainHook` — stdin event in, config resolved through the real loader — because the bug
 * was precisely a missing wire between the two, which a test of either half alone cannot see.
 */
describe("runRetainHook honors retainSessions", () => {
  const event = () => ({
    // A fresh session id per test: `sessionRootDir` caches a session's starting directory in /tmp
    // for the life of the session, so reusing one id would pin every test to the first test's temp
    // dir — and so to the wrong bank.
    session_id: `sess-gate-${basename(root)}`,
    transcript_path: file,
    cwd: root,
  });

  const spec = {
    harness: "claude-code",
    hostTimeoutSec: 60,
    parse: (ev: Record<string, unknown>) => ({
      sessionId: ev.session_id as string,
      transcriptPath: ev.transcript_path as string,
      cwd: ev.cwd as string,
    }),
  };

  beforeEach(() => {
    vi.stubEnv("HINDSIGHT_DIAG_FILE", join(root, "diag.log"));
    rawConfig = {};
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        timestamp: "2026-01-01T00:00:00Z",
        message: { role: "user", content: "we use zod for validation" },
      })
    );
    stdin = JSON.stringify(event());
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  const stubClient = () => {
    const retain = vi.fn().mockResolvedValue(undefined);
    const makeClient = vi.fn(() => ({
      retain,
      supportsIdempotentRetain: async () => false,
    })) as unknown as Parameters<typeof runRetainHook>[1];
    return { retain, makeClient };
  };

  it("writes the transcript back by default", async () => {
    const { retain, makeClient } = stubClient();
    await runRetainHook(spec, makeClient);
    expect(retain).toHaveBeenCalledTimes(1);
  });

  it("retainSessions: false -> no write-back, and no client is even built", async () => {
    rawConfig = { retainSessions: false };
    const { retain, makeClient } = stubClient();
    await runRetainHook(spec, makeClient);
    expect(retain).not.toHaveBeenCalled();
    expect(makeClient).not.toHaveBeenCalled();
  });

  it("a banks.<id> override opts one repo out while the global default still writes", async () => {
    const bankId = deriveBankId(resolveConfig(), root, spec.harness, root);
    rawConfig = { banks: { [bankId]: { retainSessions: false } } };
    const optedOut = stubClient();
    await runRetainHook(spec, optedOut.makeClient);
    expect(optedOut.retain).not.toHaveBeenCalled();

    rawConfig = { banks: { "some-other-bank": { retainSessions: false } } };
    const untouched = stubClient();
    await runRetainHook(spec, untouched.makeClient);
    expect(untouched.retain).toHaveBeenCalledTimes(1);
  });

  it("a banks.<id> override turns write-back back on under a global opt-out", async () => {
    const bankId = deriveBankId(resolveConfig(), root, spec.harness, root);
    rawConfig = { retainSessions: false, banks: { [bankId]: { retainSessions: true } } };
    const { retain, makeClient } = stubClient();
    await runRetainHook(spec, makeClient);
    expect(retain).toHaveBeenCalledTimes(1);
  });
});

/**
 * Family-wide guard, in the shape of `daemon.test.ts`'s "every harness entrypoint reaches a
 * daemon". #3596 was not a broken line of code but a MISSING one: the persistent-plugin path
 * honored `retainSessions` and the hook path silently didn't, and no test failed because the path
 * that forgot is by definition the one nobody wrote a test for. So assert over the whole family:
 * every module that puts a conversation in the bank must consult the flag — the live write-back
 * (`retainLiveSession`) and deepen's history import (`ingestChats`) alike, since either one alone
 * would leave the opt-out cosmetic.
 */
describe("every session write-back path honors retainSessions", () => {
  const SRC = fileURLToPath(new URL("..", import.meta.url));
  const WRITERS = ["retainLiveSession(", "ingestChats("];
  /** The modules that DEFINE the writers — the callers are what must gate. */
  const DEFINITIONS = ["core/chat.ts"];

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  it("has no conversation-writing module that ignores the flag", () => {
    const ungated = sourceFiles(SRC).filter((rel) => {
      if (DEFINITIONS.includes(rel)) return false;
      const src = readFileSync(join(SRC, rel), "utf8");
      if (!WRITERS.some((w) => src.includes(w))) return false;
      // `writeBackEnabled` is RuntimeCore's own reading of the same flag, shared by every
      // persistent-plugin host.
      return !src.includes("cfg.retainSessions") && !src.includes("writeBackEnabled");
    });
    expect(ungated).toEqual([]);
  });
});
