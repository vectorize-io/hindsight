/**
 * End-to-end write-back for the harnesses whose host keeps no durable transcript (ZCode, TraeCode),
 * driven through the REAL lifecycle declaration. The ZCode suites cover its fallbacks; the
 * family-wide suite at the bottom runs the shared wire for every journal-backed harness.
 *
 * ZCode was the first such host: its `Stop` payload carries the
 * assistant reply plus an ephemeral, assistant-only file it deletes as soon as the hook returns,
 * and no user prompt at all. The plugin therefore journals the conversation itself
 * (core/turn-journal.ts). That makes the retain path a WIRE between two separate hook processes —
 * exactly the kind of seam a test of either half alone cannot see — so these drive the shared
 * runtime with real events and assert on what would actually be stored.
 */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { appendJournalTurn, journalPath, readJournalTranscript } from "./core/turn-journal";
import { runRetainHook } from "./core/retain-hook";
import { runHook } from "./core/hook";
import type { RawConfig } from "./core/config";
import { HOOK_HARNESSES } from "./harness/hook-lifecycle";

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

let rawConfig: RawConfig = {};
vi.mock("./core/config", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./core/config")>();
  return { ...actual, loadConfig: () => actual.resolveConfig(rawConfig) };
});

const zcode = HOOK_HARNESSES.zcode;

let root: string;
let sessionId: string;

/** Retained turns, parsed out of the JSONL content the client was handed. The leading `system`
 *  REF-ID line every write-back carries (core/chat.ts) is dropped — it is not conversation. */
const retainedTurns = (retain: ReturnType<typeof vi.fn>, call = 0) =>
  (retain.mock.calls[call][0] as string)
    .split("\n")
    .map((line) => JSON.parse(line) as { role: string; content: string })
    .filter((turn) => turn.role !== "system");

const stubClient = () => {
  const retain = vi.fn().mockResolvedValue(undefined);
  const makeClient = vi.fn(() => ({
    retain,
    supportsAppendRetain: async () => false,
  })) as unknown as Parameters<typeof runRetainHook>[1];
  return { retain, makeClient };
};

/** What the prompt hook does for this harness: record the user's turn in the session journal. */
const promptTurn = (content: string) =>
  appendJournalTurn(journalPath("zcode", sessionId), { role: "user", content });

const stopEvent = (extra: Record<string, unknown>) => ({
  sessionId,
  session_id: sessionId,
  cwd: root,
  hookEventName: "Stop",
  ...extra,
});

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-zcode-"));
  // A fresh session id per test: `sessionRootDir` caches a session's starting directory in /tmp for
  // the life of the session, so a reused id would pin later tests to the first one's temp dir.
  sessionId = `sess-zcode-${basename(root)}`;
  rawConfig = {};
  vi.stubEnv("HINDSIGHT_DIAG_FILE", join(root, "diag.log"));
});

afterEach(() => {
  rmSync(journalPath("zcode", sessionId), { force: true });
  rmSync(root, { recursive: true, force: true });
  vi.unstubAllEnvs();
});

describe("ZCode write-back", () => {
  it("retains the whole turn, pairing the journaled prompt with the Stop reply", async () => {
    promptTurn("we use zod for validation");
    stdin = JSON.stringify(stopEvent({ responseText: "noted — zod it is" }));

    const { retain, makeClient } = stubClient();
    await runRetainHook(zcode.retain, makeClient);

    expect(retainedTurns(retain).map((t) => [t.role, t.content])).toEqual([
      ["user", "we use zod for validation"],
      ["assistant", "noted — zod it is"],
    ]);
  });

  /** The reason the journal exists: `retainLiveSession` re-reads the FULL conversation each time
   *  and lets the retain cursor send only what is new. A host that exposed just the latest reply
   *  would shrink and reorder the turn list between Stops, replacing the document every time. */
  it("keeps the whole conversation across turns, in order", async () => {
    promptTurn("we use zod for validation");
    stdin = JSON.stringify(stopEvent({ responseText: "noted" }));
    const first = stubClient();
    await runRetainHook(zcode.retain, first.makeClient);

    promptTurn("and pytest-xdist for the suite");
    stdin = JSON.stringify(stopEvent({ responseText: "got it" }));
    const second = stubClient();
    await runRetainHook(zcode.retain, second.makeClient);

    expect(retainedTurns(second.retain).map((t) => [t.role, t.content])).toEqual([
      ["user", "we use zod for validation"],
      ["assistant", "noted"],
      ["user", "and pytest-xdist for the suite"],
      ["assistant", "got it"],
    ]);
  });

  it("falls back to the ephemeral transcript when the event carries no responseText", async () => {
    const ephemeral = join(root, "ephemeral.jsonl");
    writeFileSync(
      ephemeral,
      JSON.stringify({
        message: { role: "assistant", content: [{ type: "text", text: "from the transcript" }] },
      })
    );
    promptTurn("what did you do?");
    stdin = JSON.stringify(
      stopEvent({ responseText: "", responsePreview: "trunc…", transcript_path: ephemeral })
    );

    const { retain, makeClient } = stubClient();
    await runRetainHook(zcode.retain, makeClient);
    expect(retainedTurns(retain).at(-1)).toMatchObject({
      role: "assistant",
      content: "from the transcript",
    });
  });

  it("uses the truncated preview only when nothing better survives", async () => {
    promptTurn("what did you do?");
    stdin = JSON.stringify(stopEvent({ responseText: "", responsePreview: "truncated reply" }));

    const { retain, makeClient } = stubClient();
    await runRetainHook(zcode.retain, makeClient);
    expect(retainedTurns(retain).at(-1)).toMatchObject({
      role: "assistant",
      content: "truncated reply",
    });
  });

  it("does not store the reply twice when Stop is delivered twice for one turn", async () => {
    promptTurn("ship it");
    const event = JSON.stringify(stopEvent({ responseText: "shipped" }));

    stdin = event;
    await runRetainHook(zcode.retain, stubClient().makeClient);
    stdin = event;
    const second = stubClient();
    await runRetainHook(zcode.retain, second.makeClient);

    expect(retainedTurns(second.retain).map((t) => [t.role, t.content])).toEqual([
      ["user", "ship it"],
      ["assistant", "shipped"],
    ]);
  });

  it("stores nothing when the session journal is empty and the reply is too", async () => {
    stdin = JSON.stringify(stopEvent({ responseText: "" }));
    const { retain, makeClient } = stubClient();
    await runRetainHook(zcode.retain, makeClient);
    expect(retain).not.toHaveBeenCalled();
  });

  it("honors retainSessions: false without even building a client", async () => {
    rawConfig = { retainSessions: false };
    promptTurn("we use zod for validation");
    stdin = JSON.stringify(stopEvent({ responseText: "noted" }));

    const { retain, makeClient } = stubClient();
    await runRetainHook(zcode.retain, makeClient);
    expect(retain).not.toHaveBeenCalled();
    expect(makeClient).not.toHaveBeenCalled();
  });
});

/**
 * The prompt half of the wire, driven through the REAL prompt hook rather than by calling the
 * journal directly: the journal write lives in core/hook.ts, and a harness that declares
 * `journalPrompt` but never reaches that line retains assistant-only sessions with no error.
 */
describe("ZCode prompt hook", () => {
  const promptEvent = (prompt: string) =>
    JSON.stringify({
      prompt,
      cwd: root,
      session_id: sessionId,
      hook_event_name: "UserPromptSubmit",
    });

  const runPrompt = async (prompt: string) => {
    stdin = promptEvent(prompt);
    const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    try {
      await runHook(zcode.prompt, () => ({
        reflect: async () => "",
        listPages: async () => ({ items: [] }),
        searchKnowledgePages: async () => [],
        recallObservations: async () => [],
        knowledgePagesSupported: false,
      }));
    } finally {
      write.mockRestore();
    }
  };

  beforeEach(() => {
    // Seeding spawns a background process; this suite is about the journal write, not ingestion.
    rawConfig = { autoSeed: false };
  });

  it("records the user's prompt for the Stop hook to pair with the reply", async () => {
    await runPrompt("we use zod for validation");
    expect(readJournalTranscript(journalPath("zcode", sessionId))).toEqual([
      expect.objectContaining({ role: "user", content: "we use zod for validation" }),
    ]);
  });

  it("writes nothing when the kill switch is set", async () => {
    rawConfig = { autoSeed: false, disabled: true };
    await runPrompt("we use zod for validation");
    expect(readJournalTranscript(journalPath("zcode", sessionId))).toEqual([]);
  });
});

describe("ZCode event parsing", () => {
  it("accepts either spelling of the session id — UserPromptSubmit sends one, Stop both", () => {
    expect(zcode.prompt.parse({ prompt: "hi", cwd: "/repo", session_id: "s1" })).toEqual({
      prompt: "hi",
      cwd: "/repo",
      sessionId: "s1",
    });
    expect(zcode.retain.parse({ sessionId: "s1", cwd: "/repo" })).toEqual({
      sessionId: "s1",
      cwd: "/repo",
    });
  });

  it("emits Claude Code's injection schema, which ZCode's embedded runtime reads", () => {
    expect(zcode.prompt.emit("context", "visible")).toEqual({
      systemMessage: "visible",
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: "context",
      },
    });
    expect(zcode.sessionStart.emit({ additionalContext: "context" })).toEqual({
      hookSpecificOutput: { hookEventName: "SessionStart", additionalContext: "context" },
    });
  });
});

/**
 * The same wire for EVERY harness that journals its own conversation (`journalPrompt`), driven
 * through both real hooks. Only the Stop field carrying the reply differs per host, so a new
 * journal-backed harness must name that field here — the guard below fails until it does.
 */
const JOURNAL_REPLY_FIELD: Record<string, string> = {
  zcode: "responseText",
  traecode: "last_assistant_message",
};

it("covers every journal-backed harness in the wire tests below", () => {
  const journaled = Object.entries(HOOK_HARNESSES)
    .filter(([, spec]) => spec.prompt.journalPrompt)
    .map(([name]) => name);
  expect(journaled.sort()).toEqual(Object.keys(JOURNAL_REPLY_FIELD).sort());
});

describe.each(Object.keys(JOURNAL_REPLY_FIELD))("%s: prompt hook -> Stop hook", (name) => {
  const spec = HOOK_HARNESSES[name as keyof typeof HOOK_HARNESSES];

  const prompt = async (text: string) => {
    stdin = JSON.stringify({ prompt: text, cwd: root, session_id: sessionId });
    const write = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    try {
      await runHook(spec.prompt, () => ({
        reflect: async () => "",
        listPages: async () => ({ items: [] }),
        searchKnowledgePages: async () => [],
        recallObservations: async () => [],
        knowledgePagesSupported: false,
      }));
    } finally {
      write.mockRestore();
    }
  };

  const stop = async (reply: string) => {
    stdin = JSON.stringify({
      session_id: sessionId,
      sessionId,
      cwd: root,
      [JOURNAL_REPLY_FIELD[name]]: reply,
    });
    const client = stubClient();
    await runRetainHook(spec.retain, client.makeClient);
    return client.retain;
  };

  beforeEach(() => {
    rawConfig = { autoSeed: false };
  });

  afterEach(() => {
    rmSync(journalPath(name, sessionId), { force: true });
  });

  it("retains the whole conversation across turns, in order", async () => {
    await prompt("we use zod for validation");
    await stop("noted");
    await prompt("and pytest-xdist for the suite");
    const retain = await stop("got it");

    expect(retainedTurns(retain).map((t) => [t.role, t.content])).toEqual([
      ["user", "we use zod for validation"],
      ["assistant", "noted"],
      ["user", "and pytest-xdist for the suite"],
      ["assistant", "got it"],
    ]);
  });

  it("does not store the reply twice when Stop is delivered twice for one turn", async () => {
    await prompt("ship it");
    await stop("shipped");
    const retain = await stop("shipped");

    expect(retainedTurns(retain).map((t) => [t.role, t.content])).toEqual([
      ["user", "ship it"],
      ["assistant", "shipped"],
    ]);
  });
});
