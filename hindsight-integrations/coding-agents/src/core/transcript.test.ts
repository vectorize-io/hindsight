import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readClaudeTranscript } from "./transcript";
import { buildSystemInjection } from "./inject";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-transcript-"));
  file = join(root, "session.jsonl");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("readClaudeTranscript", () => {
  it("captures text + compact action turns, dropping tool_result/non-message/isMeta/isSidechain/thinking/empty lines and tolerating malformed or non-object JSON lines", () => {
    const lines = [
      // non-message line: dropped
      JSON.stringify({ type: "last-prompt", leafUuid: "x" }),
      // kept: string content
      JSON.stringify({
        type: "user",
        timestamp: "2026-01-01T00:00:00Z",
        message: { role: "user", content: "how do we validate input?" },
      }),
      // kept: only the text block survives (thinking dropped)
      JSON.stringify({
        type: "assistant",
        timestamp: "2026-01-01T00:00:01Z",
        message: {
          role: "assistant",
          content: [
            { type: "thinking", thinking: "hmm" },
            { type: "text", text: "We use zod." },
          ],
        },
      }),
      // kept: a tool_use-only assistant line becomes a compact role:"action" turn (name + target)
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "tool_use", name: "Bash", input: { command: "npm test" } }],
        },
      }),
      // dropped: a tool_result-only user line — outputs are mechanical noise for extraction
      JSON.stringify({
        type: "user",
        message: { role: "user", content: [{ type: "tool_result", content: "12 passed" }] },
      }),
      // dropped: isMeta
      JSON.stringify({
        type: "user",
        isMeta: true,
        message: { role: "user", content: "<system-injected>" },
      }),
      // dropped: isSidechain (subagent/Task turn, not the main conversation)
      JSON.stringify({
        type: "assistant",
        isSidechain: true,
        message: { role: "assistant", content: "subagent output" },
      }),
      // malformed line: must not throw
      "{ not json",
      // JSON.parse succeeds but yields a non-object value: must not throw
      "null",
      "42",
      "[]",
      // blank line: must be skipped
      "",
    ];
    writeFileSync(file, lines.join("\n"));

    const result = readClaudeTranscript(file);

    expect(result).toEqual([
      { role: "user", content: "how do we validate input?", timestamp: "2026-01-01T00:00:00Z" },
      { role: "assistant", content: "We use zod.", timestamp: "2026-01-01T00:00:01Z" },
      { role: "action", content: "Bash npm test" },
    ]);
  });

  it("splits a mixed text + tool_use message into a prose turn plus an action turn; drops the tool_result", () => {
    const lines = [
      JSON.stringify({
        type: "assistant",
        timestamp: "2026-01-01T00:00:02Z",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "Editing the uploader." },
            {
              type: "tool_use",
              name: "Edit",
              input: { file_path: "uploader.ts", old_string: "a" },
            },
          ],
        },
      }),
      JSON.stringify({
        type: "user",
        message: { role: "user", content: [{ type: "tool_result", content: "x".repeat(5000) }] },
      }),
    ];
    writeFileSync(file, lines.join("\n"));

    const result = readClaudeTranscript(file);

    expect(result).toEqual([
      { role: "assistant", content: "Editing the uploader.", timestamp: "2026-01-01T00:00:02Z" },
      // Action line: tool name + primary target only — no arguments, no output.
      { role: "action", content: "Edit uploader.ts", timestamp: "2026-01-01T00:00:02Z" },
    ]);
  });

  it("caps a very long action target at 100 chars, and falls back to the bare tool name when the input has no target key", () => {
    const longCmd = "echo " + "y".repeat(200);
    const lines = [
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "tool_use", name: "Bash", input: { command: longCmd } }],
        },
      }),
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [{ type: "tool_use", name: "TodoWrite", input: { todos: [] } }],
        },
      }),
    ];
    writeFileSync(file, lines.join("\n"));

    const result = readClaudeTranscript(file);

    expect(result[0].role).toBe("action");
    expect(result[0].content).toBe(`Bash ${longCmd.slice(0, 100)}…`);
    expect(result[1]).toEqual({ role: "action", content: "TodoWrite" });
  });

  it("strips injected recall context so retained turns can't feed memory back into the bank", () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        message: {
          role: "user",
          content:
            "<hindsight_memories>\nsecret prior fact\n</hindsight_memories>\nWhy does upload retry?",
        },
      })
    );

    const result = readClaudeTranscript(file);

    expect(result).toHaveLength(1);
    expect(result[0].content).toContain("Why does upload retry?");
    expect(result[0].content).not.toContain("secret prior fact");
    expect(result[0].content).not.toContain("hindsight_memories");
  });

  it("drops the compaction summary — Claude Code's recap, not the user's words", () => {
    // Written when the context window fills, as a plain type:"user" record with no isMeta flag.
    // Measured at 29 records / 474,016 chars across local transcripts, averaging 16KB each, every
    // one retained as if the user had typed it — and each one summarising turns already retained,
    // since compaction APPENDS to the transcript rather than rewriting it (#3379).
    writeFileSync(
      file,
      [
        JSON.stringify({
          type: "user",
          isCompactSummary: true,
          message: {
            role: "user",
            content:
              "This session is being continued from a previous conversation that ran out of " +
              "context. The summary below covers <analysis>…</analysis>",
          },
        }),
        JSON.stringify({
          type: "user",
          message: { role: "user", content: "now add the retry backoff" },
        }),
      ].join("\n")
    );

    const result = readClaudeTranscript(file);
    expect(result).toHaveLength(1);
    expect(result[0].content).toBe("now add the retry backoff");
  });

  it("drops a <task-notification>: the harness's background-task plumbing, not the user's words", () => {
    // Claude Code delivers these as an ordinary type:"user" message with a string body and no
    // isMeta flag, so nothing else filters them and extraction saw task ids and status lines as
    // things the user said. Measured at 39 across 400 local transcripts, each the whole message
    // (#3023 — which named skill bodies and <system-reminder>; both are already handled, this is
    // the case that actually survived).
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        message: {
          role: "user",
          content:
            "<task-notification>\n<task-id>b46ca19nr</task-id>\n" +
            "<tool-use-id>toolu_0127NeaVZbdiAsbacNvasB78</tool-use-id>\n" +
            "<status>stopped</status>\n<summary>No completion record</summary>\n</task-notification>",
        },
      })
    );

    expect(readClaudeTranscript(file)).toEqual([]); // renders empty -> no turn at all
  });

  it("keeps what the user wrote around a harness wrapper", () => {
    // The stripper removes the BLOCK, never the message: replacing the whole turn with the tag's
    // contents is what made the old plugin's strip_channel_envelope discard real user text (#3124).
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        message: {
          role: "user",
          content: "before <task-notification><status>done</status></task-notification> after",
        },
      })
    );

    const result = readClaudeTranscript(file);
    expect(result).toHaveLength(1);
    expect(result[0].content).toBe("before  after");
    expect(result[0].content).not.toContain("status");
  });

  it("drops a <system-reminder> block if the harness ever delivers one as user text", () => {
    // Today these ride inside tool_result blocks, which this reader already drops entirely; the
    // rule is tag-structural so it holds if that placement changes.
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        message: {
          role: "user",
          content: "<system-reminder>plan mode is active</system-reminder>\nship the fix",
        },
      })
    );

    const result = readClaudeTranscript(file);
    expect(result).toHaveLength(1);
    expect(result[0].content).toBe("ship the fix");
  });

  it("strips the reflect hook's <hindsight_memory> injection block (buildSystemInjection output)", () => {
    // The exact block the UserPromptSubmit hook injects — wrapper tags, preamble, attribution
    // text and the surfaced memory itself must ALL be gone from retained text, or the session
    // write-back would re-ingest the injected synthesis (a retain→reflect feedback loop).
    writeFileSync(
      file,
      JSON.stringify({
        type: "user",
        message: {
          role: "user",
          content: `${buildSystemInjection("SECRET")}\nWhy does upload retry?`,
        },
      })
    );

    const result = readClaudeTranscript(file);

    expect(result).toHaveLength(1);
    expect(result[0].content).toContain("Why does upload retry?");
    expect(result[0].content).not.toContain("SECRET");
    expect(result[0].content).not.toContain("Relevant project memory");
    expect(result[0].content).not.toContain("hindsight_memory");
  });

  it("fails open (returns []) when the file cannot be read", () => {
    expect(readClaudeTranscript(join(root, "does-not-exist.jsonl"))).toEqual([]);
  });

  it("joins multiple text blocks in a single message with newlines", () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "First paragraph." },
            { type: "thinking", thinking: "irrelevant" },
            { type: "text", text: "Second paragraph." },
          ],
        },
      })
    );

    const result = readClaudeTranscript(file);

    expect(result).toEqual([{ role: "assistant", content: "First paragraph.\nSecond paragraph." }]);
  });
});
