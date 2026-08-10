import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readCodexTranscript } from "./transcript-codex";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-codex-transcript-"));
  file = join(root, "rollout.jsonl");
});
afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const item = (payload: unknown, timestamp?: string) =>
  JSON.stringify({ type: "response_item", ...(timestamp ? { timestamp } : {}), payload });

describe("readCodexTranscript", () => {
  it("keeps user/assistant text + compact action turns; drops developer/synthetic/reasoning/outputs/injected", () => {
    const lines = [
      // non-response_item line: dropped
      JSON.stringify({ type: "session_meta", payload: { cwd: "/repo" } }),
      // developer message (Codex system prompt + our injected context): dropped entirely
      item({
        type: "message",
        role: "developer",
        content: [{ type: "input_text", text: "<permissions instructions>…</permissions>" }],
      }),
      item({
        type: "message",
        role: "developer",
        content: [
          {
            type: "input_text",
            text: "<hindsight_memories>\nsecret recalled fact\n</hindsight_memories>",
          },
        ],
      }),
      // synthetic startup user message: dropped
      item({
        type: "message",
        role: "user",
        content: [
          {
            type: "input_text",
            text: "# AGENTS.md instructions for /repo\n<INSTRUCTIONS>x</INSTRUCTIONS>",
          },
          {
            type: "input_text",
            text: "<environment_context>\n<cwd>/repo</cwd>\n</environment_context>",
          },
        ],
      }),
      // real user prompt: kept
      item({
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "add retry backoff to the uploader" }],
      }),
      // reasoning: dropped
      item({ type: "reasoning", id: "rs_1", encrypted_content: "…" }),
      // assistant commentary: kept
      item({
        type: "message",
        role: "assistant",
        phase: "commentary",
        content: [{ type: "output_text", text: "I'll add exponential backoff." }],
      }),
      // tool call: kept as a compact role:"action" turn (name + primary target, no args)
      item({
        type: "function_call",
        name: "exec_command",
        arguments: '{"command":"npm test"}',
        call_id: "call_1",
      }),
      // tool result: dropped — outputs are mechanical noise for extraction
      item({ type: "function_call_output", call_id: "call_1", output: "12 passed" }),
      // assistant final answer: kept
      item({
        type: "message",
        role: "assistant",
        phase: "final_answer",
        content: [{ type: "output_text", text: "Done — backoff added, tests pass." }],
      }),
      // malformed line + non-object: tolerated
      "{ not json",
      "null",
    ];
    writeFileSync(file, lines.join("\n"));

    const turns = readCodexTranscript(file);

    expect(turns).toEqual([
      { role: "user", content: "add retry backoff to the uploader" },
      { role: "assistant", content: "I'll add exponential backoff." },
      { role: "action", content: "exec_command npm test" },
      { role: "assistant", content: "Done — backoff added, tests pass." },
    ]);
  });

  it("strips injected memory that leaks into a kept (user/assistant) message", () => {
    writeFileSync(
      file,
      item({
        type: "message",
        role: "user",
        content: [
          {
            type: "input_text",
            text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?",
          },
        ],
      })
    );
    const turns = readCodexTranscript(file);
    expect(turns).toEqual([{ role: "user", content: "Why retry?" }]);
  });

  it("preserves stable rollout timestamps on retained message and action turns", () => {
    writeFileSync(
      file,
      [
        item(
          {
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: "retry this retain" }],
          },
          "2026-08-10T10:00:00.000Z"
        ),
        item(
          { type: "function_call", name: "shell", arguments: '{"command":"npm test"}' },
          "2026-08-10T10:00:01.000Z"
        ),
      ].join("\n")
    );

    expect(readCodexTranscript(file)).toEqual([
      {
        role: "user",
        content: "retry this retain",
        timestamp: "2026-08-10T10:00:00.000Z",
      },
      {
        role: "action",
        content: "shell npm test",
        timestamp: "2026-08-10T10:00:01.000Z",
      },
    ]);
  });

  it("strips <hook_prompt> transport wrappers from user messages: a pure hook_prompt yields no turn; mixed content keeps only the real text", () => {
    // A user message that is ONLY a hook_prompt block (codex surfaces hook stdout/errors this
    // way — transport noise, not the user's work): stripped fully → no turn at all.
    writeFileSync(
      file,
      item({
        type: "message",
        role: "user",
        content: [
          {
            type: "input_text",
            text: "<hook_prompt hook_run_id=\"stop:4:abc\">python3: can't open file '/tmp/check.py': [Errno 2] No such file or directory</hook_prompt>",
          },
        ],
      })
    );
    expect(readCodexTranscript(file)).toEqual([]);

    // A hook_prompt block followed by real user text: only the real text survives.
    writeFileSync(
      file,
      item({
        type: "message",
        role: "user",
        content: [
          {
            type: "input_text",
            text: '<hook_prompt hook_run_id="stop:5:def">noise</hook_prompt>\nplease fix the uploader',
          },
        ],
      })
    );
    expect(readCodexTranscript(file)).toEqual([
      { role: "user", content: "please fix the uploader" },
    ]);
  });

  it("a function_call with unparsable arguments still yields the bare tool name", () => {
    writeFileSync(file, item({ type: "function_call", name: "shell", arguments: "not json" }));
    expect(readCodexTranscript(file)).toEqual([{ role: "action", content: "shell" }]);
  });

  it("drops function_call_output entirely — even a huge one produces no turn", () => {
    writeFileSync(
      file,
      item({ type: "function_call_output", call_id: "c", output: "x".repeat(5000) })
    );
    expect(readCodexTranscript(file)).toEqual([]);
  });

  it("fails open (returns []) when the file cannot be read", () => {
    expect(readCodexTranscript(join(root, "nope.jsonl"))).toEqual([]);
  });
});
