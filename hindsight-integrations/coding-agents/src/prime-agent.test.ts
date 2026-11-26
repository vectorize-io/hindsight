import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TransportTurn } from "./core/chat";
import { primeSession, startPrimeWriteBack } from "./prime-agent";

const dirs: string[] = [];
afterEach(() => {
  vi.useRealTimers();
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

describe("primeSession", () => {
  it("finds the transcript under the forwarded agent dir", () => {
    expect(
      primeSession({ HINDSIGHT_PRIME_SESSION_ID: "abc", HINDSIGHT_PRIME_AGENT_DIR: "/a" }, "/h")
    ).toEqual({ id: "abc", file: "/a/sessions/abc.jsonl" });
    expect(primeSession({ HINDSIGHT_PRIME_SESSION_ID: "abc" }, "/h")?.file).toBe(
      "/h/.prime/agent/sessions/abc.jsonl"
    );
  });

  it("is undefined for an unsaved session, whose id the kernel blanks", () => {
    expect(primeSession({ HINDSIGHT_PRIME_SESSION_ID: "" })).toBeUndefined();
    expect(primeSession({})).toBeUndefined();
  });
});

const line = (role: string, text: string, ts: string) =>
  JSON.stringify({
    type: "message",
    timestamp: ts,
    message: { role, content: [{ type: "text", text }] },
  });

describe("startPrimeWriteBack", () => {
  function setup() {
    const dir = mkdtempSync(join(tmpdir(), "hs-prime-"));
    dirs.push(dir);
    mkdirSync(join(dir, "sessions"));
    const file = join(dir, "sessions", "s1.jsonl");
    const header = JSON.stringify({ type: "session", id: "s1", cwd: "/repo" });
    let source: (() => Promise<TransportTurn[]>) | undefined;
    const seen: TransportTurn[][] = [];
    const core = {
      setTranscriptSource: (f: () => Promise<TransportTurn[]>) => (source = f),
      onSessionIdle: vi.fn(async () => {
        seen.push(await source!());
      }),
      flush: vi.fn(async () => {}),
    };
    let t = 1_000;
    const write = (...lines: string[]) => {
      writeFileSync(file, [header, ...lines].join("\n") + "\n");
      utimesSync(file, ++t, ++t); // a distinct mtime per write, whatever the FS resolution
    };
    return { file, core, seen, write };
  }

  it("writes back once the reply is in, skips unchanged files, and does a final write on stop", async () => {
    vi.useFakeTimers();
    const { file, core, seen, write } = setup();
    const stop = startPrimeWriteBack(core as never, { id: "s1", file }, 100);

    write(line("user", "fix the bug", "2026-11-26T10:00:00Z"));
    await vi.advanceTimersByTimeAsync(100);
    expect(core.onSessionIdle).not.toHaveBeenCalled(); // mid-turn: no reply yet

    write(
      line("user", "fix the bug", "2026-11-26T10:00:00Z"),
      line("assistant", "fixed", "2026-11-26T10:00:05Z")
    );
    await vi.advanceTimersByTimeAsync(100);
    await vi.advanceTimersByTimeAsync(100); // same mtime: nothing new
    expect(core.onSessionIdle).toHaveBeenCalledTimes(1);
    expect(seen[0].map((x) => x.role)).toEqual(["user", "assistant"]);

    await stop();
    expect(core.onSessionIdle).toHaveBeenCalledTimes(2);
    expect(core.flush).toHaveBeenCalled();
  });

  it("does nothing before Prime has written the session", async () => {
    const { file, core } = setup();
    const stop = startPrimeWriteBack(core as never, { id: "s1", file }, 100);
    await stop();
    expect(core.onSessionIdle).not.toHaveBeenCalled();
  });
});
