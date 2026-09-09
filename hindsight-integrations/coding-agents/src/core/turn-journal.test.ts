import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { appendJournalTurn, journalPath, readJournalTranscript } from "./turn-journal";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-journal-"));
  file = join(root, "session.journal.jsonl");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("journalPath", () => {
  it("keeps each session in its own file, beside the session cache", () => {
    expect(journalPath("zcode", "sess-1")).toBe(
      join(tmpdir(), "hindsight-zcode", "sess-1.journal.jsonl")
    );
    expect(journalPath("zcode", "sess-2")).not.toBe(journalPath("zcode", "sess-1"));
  });

  it("falls back to a fixed name when the host sent no session id", () => {
    expect(journalPath("zcode", undefined)).toBe(
      join(tmpdir(), "hindsight-zcode", "no-session.journal.jsonl")
    );
  });
});

describe("appendJournalTurn", () => {
  it("builds the conversation in order across separate hook processes", () => {
    appendJournalTurn(file, { role: "user", content: "we use zod for validation" });
    appendJournalTurn(file, { role: "assistant", content: "noted" });
    appendJournalTurn(file, { role: "user", content: "and pytest-xdist" });

    expect(readJournalTranscript(file).map((t) => [t.role, t.content])).toEqual([
      ["user", "we use zod for validation"],
      ["assistant", "noted"],
      ["user", "and pytest-xdist"],
    ]);
  });

  it("stamps a timestamp when the caller supplies none, and keeps one that is supplied", () => {
    appendJournalTurn(file, { role: "user", content: "a", timestamp: "2026-01-01T00:00:00Z" });
    appendJournalTurn(file, { role: "assistant", content: "b" });
    const [first, second] = readJournalTranscript(file);
    expect(first.timestamp).toBe("2026-01-01T00:00:00Z");
    expect(Date.parse(second.timestamp!)).not.toBeNaN();
  });

  /** A host can deliver Stop twice for one reply; the same text twice is two turns that never
   *  happened, and the retain cursor would then append the duplicate to the bank. */
  it("ignores a turn that repeats the one already at the end", () => {
    appendJournalTurn(file, { role: "assistant", content: "done" });
    appendJournalTurn(file, { role: "assistant", content: "done" });
    expect(readJournalTranscript(file)).toHaveLength(1);
  });

  it("still records the same text when the other speaker says it, or when it recurs later", () => {
    appendJournalTurn(file, { role: "user", content: "ship it" });
    appendJournalTurn(file, { role: "assistant", content: "ship it" });
    appendJournalTurn(file, { role: "user", content: "ship it" });
    expect(readJournalTranscript(file)).toHaveLength(3);
  });

  it("drops empty and whitespace-only turns rather than writing a blank line", () => {
    appendJournalTurn(file, { role: "assistant", content: "" });
    appendJournalTurn(file, { role: "assistant", content: "   \n " });
    expect(readJournalTranscript(file)).toEqual([]);
  });

  /** Without this the recall injected on turn N would be retained as the user's own words on turn
   *  N, and recalled again on turn N+1 — the feedback loop stripInjectedMemory exists to break. */
  it("strips injected memory before it can be written back", () => {
    appendJournalTurn(file, {
      role: "assistant",
      content: "done <hindsight_memories>recalled fact</hindsight_memories>",
    });
    expect(readJournalTranscript(file)).toEqual([
      expect.objectContaining({ role: "assistant", content: "done" }),
    ]);
  });

  it("creates the harness directory on the first turn of a session", () => {
    const nested = journalPath("zcode-test-harness", `sess-${Date.now()}`);
    try {
      appendJournalTurn(nested, { role: "user", content: "hello" });
      expect(readJournalTranscript(nested)).toHaveLength(1);
    } finally {
      rmSync(nested, { force: true });
    }
  });
});

describe("readJournalTranscript", () => {
  it("returns nothing for a journal that does not exist", () => {
    expect(readJournalTranscript(join(root, "absent.jsonl"))).toEqual([]);
  });

  /** A process killed mid-append leaves a torn final line. That costs its own turn, never the
   *  session — the whole conversation before it is still retainable. */
  it("keeps every complete turn before a torn final line", () => {
    writeFileSync(
      file,
      `${JSON.stringify({ role: "user", content: "first" })}\n{"role":"assistant","cont`
    );
    expect(readJournalTranscript(file)).toEqual([
      expect.objectContaining({ role: "user", content: "first" }),
    ]);
  });

  it("skips lines that parse to something other than a turn", () => {
    writeFileSync(
      file,
      ["null", "42", '["user","hi"]', '{"role":"user"}', '{"content":"no role"}', ""].join("\n") +
        "\n" +
        JSON.stringify({ role: "user", content: "real" })
    );
    expect(readJournalTranscript(file)).toEqual([
      expect.objectContaining({ role: "user", content: "real" }),
    ]);
  });

  it("round-trips content with newlines and quotes", () => {
    const content = 'line one\nline "two"\ttabbed';
    appendJournalTurn(file, { role: "user", content });
    expect(readFileSync(file, "utf8").trimEnd().split("\n")).toHaveLength(1);
    expect(readJournalTranscript(file)[0].content).toBe(content);
  });
});
