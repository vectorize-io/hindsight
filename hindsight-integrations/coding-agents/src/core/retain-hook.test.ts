import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import { buildRetain, runRetainHook } from "./retain-hook";

let root: string;
let file: string;
const ORIGINAL_DIAG_FILE = process.env.HINDSIGHT_DIAG_FILE;
const ORIGINAL_LOG_FILE = process.env.HINDSIGHT_LOG_FILE;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-retain-hook-"));
  file = join(root, "session.jsonl");
  process.env.HINDSIGHT_DIAG_FILE = join(root, "diag.jsonl");
  process.env.HINDSIGHT_LOG_FILE = join(root, "plugin.log");
});

afterEach(() => {
  if (ORIGINAL_DIAG_FILE === undefined) delete process.env.HINDSIGHT_DIAG_FILE;
  else process.env.HINDSIGHT_DIAG_FILE = ORIGINAL_DIAG_FILE;
  if (ORIGINAL_LOG_FILE === undefined) delete process.env.HINDSIGHT_LOG_FILE;
  else process.env.HINDSIGHT_LOG_FILE = ORIGINAL_LOG_FILE;
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

  it("fails open and records a diagnostic when the transcript reader throws", async () => {
    const retainSpy = vi.fn().mockResolvedValue(undefined);
    const client = { retain: retainSpy } as unknown as HindsightClient;
    const readTranscript = vi.fn(() => {
      throw new Error("spawnSync sqlite3 ENOENT");
    });

    await expect(
      buildRetain({
        harness: "devin-cli",
        sessionId: "session-unreadable",
        transcriptPath: "session-unreadable",
        client,
        readTranscript,
      })
    ).resolves.toBeUndefined();

    expect(retainSpy).not.toHaveBeenCalled();
    expect(JSON.parse(readFileSync(process.env.HINDSIGHT_DIAG_FILE!, "utf8"))).toMatchObject({
      harness: "devin-cli",
      event: "retain_transcript_failed",
      error: "spawnSync sqlite3 ENOENT",
      session: "session-unreadable",
    });
    expect(readFileSync(process.env.HINDSIGHT_LOG_FILE!, "utf8")).toContain(
      "session transcript read failed"
    );
  });

  it("keeps sqlite stderr when a long child-process message would hide it", async () => {
    const client = { retain: vi.fn() } as unknown as HindsightClient;
    const readTranscript = vi.fn(() => {
      throw Object.assign(new Error(`Command failed: ${"x".repeat(400)}`), {
        stderr: Buffer.from(
          "\u001b[31mError: in prepare, database disk image is malformed\u001b[0m\nnext line"
        ),
      });
    });

    await buildRetain({
      harness: "devin-cli",
      sessionId: "session-malformed",
      transcriptPath: "session-malformed",
      client,
      readTranscript,
    });

    const diagnostic = JSON.parse(readFileSync(process.env.HINDSIGHT_DIAG_FILE!, "utf8"));
    expect(diagnostic).toMatchObject({ event: "retain_transcript_failed" });
    expect(diagnostic.error).toContain(
      "stderr: Error: in prepare, database disk image is malformed next line"
    );
    expect(diagnostic.error).toContain("message: Command failed:");
    expect(diagnostic.error).not.toContain("\u001b");
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
