import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { readCodebuddyIdeTranscript, readCodebuddyTranscript } from "./transcript-codebuddy-ide";

interface Rec {
  role: string;
  content: unknown[];
  createdAt?: string;
}

/**
 * Write a CodeBuddy IDE conversation DIRECTORY — the real layout, measured on disk: one file per
 * message carrying the model-facing record as an ESCAPED JSON string, plus an `index.json` that
 * holds only the order. Returns the `index.json` path the Stop hook is handed.
 */
function session(records: Rec[]): string {
  const conv = join(
    mkdtempSync(join(tmpdir(), "hs-codebuddy-ide-")),
    "history",
    "12ed3c88751a19b6375609f74a9e4f3a",
    "e8587082afc54e929a96132018ba7304"
  );
  const messagesDir = join(conv, "messages");
  mkdirSync(messagesDir, { recursive: true });
  const index = records.map((rec, i) => {
    const id = `msg-${i}`;
    writeFileSync(
      join(messagesDir, `${id}.json`),
      JSON.stringify({
        role: rec.role,
        id,
        extra: { modelId: "deepseek-v4.1-flash" },
        createdAt: rec.createdAt,
        message: JSON.stringify({ role: rec.role, content: rec.content }),
      })
    );
    return { id, type: "text", role: rec.role, isComplete: true };
  });
  const indexPath = join(conv, "index.json");
  writeFileSync(indexPath, JSON.stringify({ messages: index }));
  return indexPath;
}

/** A conversation's first prompt — the real shape: the IDE's context blocks, then the wrapped query. */
const FIRST_PROMPT =
  "<user_info>\nOS Version: darwin\nShell: Zsh\nWorkspace Folder: /w/repo\n</user_info>\n\n" +
  "<git_status>\nThis is the git status at the start of the conversation.\n</git_status>\n\n" +
  "<additional_data>\ncurrent_time: Tuesday, September 15, 2026\n</additional_data>\n\n" +
  '<system_reminder data-role="user-context">\n<hindsight_memory>old</hindsight_memory>\n</system_reminder>\n\n' +
  "<user_query>\nfix the flake\n</user_query>";

describe("readCodebuddyIdeTranscript", () => {
  it("renders a prompt, its tool call and the reply in index order", () => {
    const path = session([
      {
        role: "user",
        createdAt: "2026-09-15T07:32:20.000Z",
        content: [{ type: "text", text: FIRST_PROMPT }],
      },
      {
        role: "assistant",
        createdAt: "2026-09-15T07:32:21.000Z",
        content: [
          { type: "reasoning", text: "thinking about it" },
          {
            type: "tool-call",
            toolCallId: "c1",
            toolName: "execute_command",
            args: { command: "ls -la" },
          },
        ],
      },
      {
        role: "tool",
        createdAt: "2026-09-15T07:32:21.500Z",
        content: [
          {
            type: "tool-result",
            toolCallId: "c1",
            toolName: "execute_command",
            result: { status: "success", success: true },
            isError: false,
          },
        ],
      },
      {
        role: "assistant",
        createdAt: "2026-09-15T07:32:22.000Z",
        content: [
          { type: "reasoning", text: "done" },
          { type: "text", text: "Patched the retry loop." },
        ],
      },
    ]);

    expect(readCodebuddyIdeTranscript(path)).toEqual([
      { role: "user", content: "fix the flake", timestamp: "2026-09-15T07:32:20.000Z" },
      { role: "action", content: "execute_command ls -la", timestamp: "2026-09-15T07:32:21.000Z" },
      {
        role: "assistant",
        content: "Patched the retry loop.",
        timestamp: "2026-09-15T07:32:22.000Z",
      },
    ]);
  });

  it("strips the IDE's context blocks, which sit OUTSIDE the wrapper WorkBuddy nests them in", () => {
    // WorkBuddy nests <user_info> inside <system-reminder>, so the shared stripper removed it as a
    // side effect and no reader ever had to know the tag exists. The IDE emits its blocks OUTSIDE
    // that wrapper — and spells the wrapper with an underscore — so without this reader's own pass
    // the retained prompt would carry OS/shell/workspace/git/time scaffolding as the human's words.
    const path = session([
      {
        role: "user",
        createdAt: "2026-09-15T07:32:20.000Z",
        content: [
          {
            type: "text",
            text:
              "<user_info>\nOS Version: darwin\nShell: Zsh\n</user_info>\n\n" +
              "<git_status>\nOn branch main\n</git_status>\n\n" +
              "<additional_data>\ncurrent_time: Tuesday, September 15, 2026\n</additional_data>\n\n" +
              "<user_query>carry on</user_query>",
          },
        ],
      },
      // The hyphen spelling with a nested <user_info> (WorkBuddy's shape) stays the shared
      // stripper's job — this reader must not double-handle it, only leave it clean.
      {
        role: "user",
        createdAt: "2026-09-15T07:32:21.000Z",
        content: [
          {
            type: "text",
            text:
              '<system-reminder data-role="user-context">\n<user_info>…</user_info>\n</system-reminder>\n' +
              "<user_query>and again</user_query>",
          },
        ],
      },
    ]);

    expect(readCodebuddyIdeTranscript(path)).toEqual([
      { role: "user", content: "carry on", timestamp: "2026-09-15T07:32:20.000Z" },
      { role: "user", content: "and again", timestamp: "2026-09-15T07:32:21.000Z" },
    ]);
  });

  it("drops the host's machine-written user records, tool output, reasoning and images", () => {
    const path = session([
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "<cb_summary>\nSummary of the conversation so far: …\n</cb_summary>",
          },
        ],
      },
      { role: "assistant", content: [{ type: "reasoning", text: "thinking" }] },
      {
        role: "tool",
        content: [
          { type: "tool-result", toolCallId: "c", toolName: "Read", result: { text: "…" } },
        ],
      },
      { role: "assistant", content: [{ type: "image", image: "…" }] },
    ]);

    expect(readCodebuddyIdeTranscript(path)).toEqual([]);
  });

  it("leaves text that merely mentions the wrapper alone — anchored, never a substring replace", () => {
    // A real prompt in a repo whose subject IS this integration can quote the wrapper verbatim.
    const path = session([
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "<user_info>\nOS\n</user_info>\nnote: the host writes <user_query>…</user_query> around prompts",
          },
        ],
      },
    ]);

    expect(readCodebuddyIdeTranscript(path)[0].content).toBe(
      "note: the host writes <user_query>…</user_query> around prompts"
    );
  });

  it("joins a record's text blocks into one turn and keeps a tool call with malformed args", () => {
    const path = session([
      {
        role: "assistant",
        createdAt: "2026-09-15T07:32:22.000Z",
        content: [
          { type: "text", text: "line one" },
          { type: "image", image: "…" },
          { type: "text", text: "line two" },
        ],
      },
      {
        role: "assistant",
        createdAt: "2026-09-15T07:32:23.000Z",
        content: [{ type: "tool-call", toolCallId: "c", toolName: "Bash", args: "{oops" }],
      },
    ]);

    expect(readCodebuddyIdeTranscript(path)).toEqual([
      { role: "assistant", content: "line one\nline two", timestamp: "2026-09-15T07:32:22.000Z" },
      { role: "action", content: "Bash {oops", timestamp: "2026-09-15T07:32:23.000Z" },
    ]);
  });

  it("fails open: a missing index, a missing message file and a malformed inner record", () => {
    expect(
      readCodebuddyIdeTranscript(join(tmpdir(), "hs-codebuddy-ide-missing/index.json"))
    ).toEqual([]);

    const dir = mkdtempSync(join(tmpdir(), "hs-codebuddy-ide-broken-"));
    mkdirSync(join(dir, "messages"), { recursive: true });
    writeFileSync(
      join(dir, "messages", "bad.json"),
      JSON.stringify({ role: "user", message: "{ not json" })
    );
    writeFileSync(
      join(dir, "index.json"),
      JSON.stringify({ messages: [{ id: "gone" }, { id: "bad" }] })
    );

    expect(readCodebuddyIdeTranscript(join(dir, "index.json"))).toEqual([]);
  });
});

describe("readCodebuddyTranscript", () => {
  it("dispatches the IDE's index.json to the IDE reader and a .jsonl to the CLI reader", () => {
    const ide = session([
      {
        role: "assistant",
        createdAt: "2026-09-15T07:32:22.000Z",
        content: [{ type: "text", text: "hello from the IDE" }],
      },
    ]);
    expect(readCodebuddyTranscript(ide)).toEqual([
      { role: "assistant", content: "hello from the IDE", timestamp: "2026-09-15T07:32:22.000Z" },
    ]);

    // CodeBuddy Code (the CLI) keeps writing the WorkBuddy JSONL, which must keep working unchanged.
    const dir = mkdtempSync(join(tmpdir(), "hs-codebuddy-cli-"));
    const jsonl = join(dir, "session.jsonl");
    writeFileSync(
      jsonl,
      JSON.stringify({
        type: "message",
        role: "assistant",
        timestamp: Date.parse("2026-09-15T07:32:22Z"),
        content: [{ type: "output_text", text: "hello from the CLI" }],
      }) + "\n"
    );
    expect(readCodebuddyTranscript(jsonl)).toEqual([
      { role: "assistant", content: "hello from the CLI", timestamp: "2026-09-15T07:32:22.000Z" },
    ]);
  });
});
