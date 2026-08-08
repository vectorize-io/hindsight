import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import { buildRetain, runRetainHook } from "./retain-hook";

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

  it("merges configured attribution into the session document", async () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        timestamp: "2026-01-01T00:00:00Z",
        message: { role: "user", content: "hello" },
      })
    );
    const retainSpy = vi.fn().mockResolvedValue(undefined);

    await buildRetain({
      harness: "codex",
      sessionId: "sess-tags",
      transcriptPath: file,
      client: { retain: retainSpy } as unknown as HindsightClient,
      attribution: {
        tags: ["project:potcodev", "auto:codex-transcript"],
        metadata: { project: "potcodev", cwd: "/work/potcodev" },
      },
    });

    const [, , , tags, , opts] = retainSpy.mock.calls[0];
    expect(tags).toEqual([
      "source:chat",
      "harness:codex",
      "project:potcodev",
      "auto:codex-transcript",
    ]);
    expect(opts.metadata).toMatchObject({
      source: "chat",
      harness: "codex",
      project: "potcodev",
      cwd: "/work/potcodev",
    });
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
