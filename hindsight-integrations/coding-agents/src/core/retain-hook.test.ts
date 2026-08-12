import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import { buildRetain, runRetainHook } from "./retain-hook";
import { memoryCursorStore, type RetainCursorStore } from "./retain-cursor";

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
    await runRetainHook({ harness: "claude-code", parse: () => ({}) }, makeClient);
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
