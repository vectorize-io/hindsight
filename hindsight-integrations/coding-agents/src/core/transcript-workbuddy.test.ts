import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { readWorkbuddyTranscript } from "./transcript-workbuddy";

const at = (iso: string) => Date.parse(iso);

/** Write raw lines to a temp JSONL file and read it back — the reader is a file reader. */
function read(lines: unknown[]): ReturnType<typeof readWorkbuddyTranscript> {
  const dir = mkdtempSync(join(tmpdir(), "hs-workbuddy-"));
  const path = join(dir, "session.jsonl");
  writeFileSync(
    path,
    lines.map((l) => (typeof l === "string" ? l : JSON.stringify(l))).join("\n") + "\n"
  );
  return readWorkbuddyTranscript(path);
}

/** A `type:"message"` user record — the real shape: host context, then the prompt in <user_query>. */
const userMsg = (text: string, ts: number) => ({
  type: "message",
  role: "user",
  timestamp: ts,
  content: [{ type: "input_text", text }],
});

const asstMsg = (text: string, ts: number) => ({
  type: "message",
  role: "assistant",
  timestamp: ts,
  content: [{ providerData: { annotations: [] }, type: "output_text", text }],
});

describe("readWorkbuddyTranscript", () => {
  it("renders a prompt, its tool call and the reply in log order", () => {
    const prompt =
      '<system-reminder data-role="user-context">\n<user_info>…</user_info>\n</system-reminder>\n' +
      "<user_query>fix the flake</user_query>";
    const turns = read([
      userMsg(prompt, at("2026-08-14T10:00:01Z")),
      {
        type: "function_call",
        name: "Edit",
        arguments: '{"file_path":"src/app.ts"}',
        timestamp: at("2026-08-14T10:00:02Z"),
      },
      // tool OUTPUT — dropped, like every other reader keeps raw results out of the bank
      {
        type: "function_call_result",
        name: "Edit",
        output: { type: "text", text: "ok" },
        timestamp: at("2026-08-14T10:00:02Z"),
      },
      asstMsg("Patched the retry loop.", at("2026-08-14T10:00:03Z")),
    ]);

    expect(turns).toEqual([
      { role: "user", content: "fix the flake", timestamp: "2026-08-14T10:00:01.000Z" },
      { role: "action", content: "Edit src/app.ts", timestamp: "2026-08-14T10:00:02.000Z" },
      {
        role: "assistant",
        content: "Patched the retry loop.",
        timestamp: "2026-08-14T10:00:03.000Z",
      },
    ]);
  });

  it("drops the host's own user records: compaction summaries and background-task notices", () => {
    const turns = read([
      userMsg(
        "<cb_summary>\nSummary of the conversation so far: …\n</cb_summary>",
        at("2026-08-14T10:00:00Z")
      ),
      userMsg(
        "<conversation_history_summary>\nAll tasks … completed.\n</conversation_history_summary>",
        at("2026-08-14T10:00:01Z")
      ),
      userMsg(
        'Use the TaskOutput tool with task_id="abc" to retrieve the full output',
        at("2026-08-14T10:00:02Z")
      ),
      userMsg(
        "<system-reminder>x</system-reminder>\n<user_query>carry on</user_query>",
        at("2026-08-14T10:00:03Z")
      ),
    ]);

    expect(turns).toEqual([
      { role: "user", content: "carry on", timestamp: "2026-08-14T10:00:03.000Z" },
    ]);
  });

  it("unwraps a prompt that carries an attached image's local path", () => {
    const text =
      "<user_query>@image#1:shot.png what is this?</user_query>" +
      "<image_local_path>/Users/u/.workbuddy/clipboard-images/x.png</image_local_path>";
    expect(read([userMsg(text, at("2026-08-14T10:00:00Z"))])).toEqual([
      {
        role: "user",
        content: "@image#1:shot.png what is this?",
        timestamp: "2026-08-14T10:00:00.000Z",
      },
    ]);
  });

  it("leaves text that merely mentions the tag alone — anchored, never a substring replace", () => {
    // A real prompt in a repo whose subject IS this integration can quote the wrapper verbatim;
    // only a whole-text match may be unwrapped (see transcript-qwen.ts's warning).
    const text =
      "<system-reminder>x</system-reminder>\nnote: the host writes <user_query>…</user_query> around prompts";
    expect(read([userMsg(text, at("2026-08-14T10:00:00Z"))])[0].content).toBe(
      "note: the host writes <user_query>…</user_query> around prompts"
    );
  });

  it("ignores non-conversation records, empty renders and malformed lines", () => {
    const lines = [
      {
        type: "reasoning",
        timestamp: at("2026-08-14T10:00:00Z"),
        content: [],
        rawContent: [{ type: "reasoning_text", text: "thinking" }],
      },
      { type: "file-history-snapshot", timestamp: at("2026-08-14T10:00:01Z"), snapshot: {} },
      { type: "ai-title", timestamp: at("2026-08-14T10:00:02Z"), aiTitle: "t" },
      { type: "summary", timestamp: at("2026-08-14T10:00:03Z") },
      { type: "resend-fork-notice", timestamp: at("2026-08-14T10:00:04Z") },
      // a user record that renders empty once the host context is stripped
      userMsg(
        "<system-reminder>only harness scaffolding</system-reminder>",
        at("2026-08-14T10:00:05Z")
      ),
      null,
      42,
      "{ not valid json",
    ];

    expect(read(lines)).toEqual([]);
  });

  it("joins every text block of one record, ignoring non-text blocks", () => {
    const turns = read([
      {
        type: "message",
        role: "assistant",
        timestamp: at("2026-08-14T10:00:00Z"),
        content: [
          { providerData: {}, type: "output_text", text: "line one" },
          { type: "image_blob_ref", ref: "abc" },
          { type: "output_text", text: "line two" },
        ],
      },
    ]);
    expect(turns).toEqual([
      { role: "assistant", content: "line one\nline two", timestamp: "2026-08-14T10:00:00.000Z" },
    ]);
  });

  it("keeps a tool call whose arguments the model produced malformed", () => {
    const turns = read([
      {
        type: "function_call",
        name: "Bash",
        arguments: "{oops",
        timestamp: at("2026-08-14T10:00:00Z"),
      },
    ]);
    expect(turns).toEqual([
      { role: "action", content: "Bash {oops", timestamp: "2026-08-14T10:00:00.000Z" },
    ]);
  });

  it("survives a missing file", () => {
    expect(readWorkbuddyTranscript(join(tmpdir(), "hs-workbuddy-missing-xyz.jsonl"))).toEqual([]);
  });
});
