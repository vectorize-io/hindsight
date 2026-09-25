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

const item = (payload: unknown) => JSON.stringify({ type: "response_item", payload });

const userEvent = (...content: unknown[]) =>
  JSON.stringify({
    type: "event_msg",
    payload: { type: "item_completed", item: { type: "UserMessage", content } },
  });
const userItem = (text: string) =>
  item({ type: "message", role: "user", content: [{ type: "input_text", text }] });
const text = (t: string) => ({ type: "text", text: t });

const startup =
  "<recommended_plugins>\nUse available tools.\n</recommended_plugins>\n" +
  "<environment_context>\n<cwd>/example</cwd>\n</environment_context>";

describe("readCodexTranscript", () => {
  it("takes user turns from UserMessage events: injected role:user items never become turns", () => {
    writeFileSync(
      file,
      [
        userItem(startup), // Desktop startup: no UserMessage event
        userItem("What is 2 + 2?"),
        userEvent(text("What is 2 + 2?")),
        item({
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "4" }],
        }),
        userItem("<turn_aborted>\nThe user interrupted.\n</turn_aborted>"),
        userItem("The following is the Codex agent history…"), // compaction summary
        userItem("And 3 + 3?"),
        userEvent(text("And 3 + 3?")),
      ].join("\n")
    );
    expect(readCodexTranscript(file)).toEqual([
      { role: "user", content: "What is 2 + 2?" },
      { role: "assistant", content: "4" },
      { role: "user", content: "And 3 + 3?" },
    ]);
  });

  it("keeps genuine user text even when it is identical to startup markup", () => {
    writeFileSync(file, [userItem(startup), userEvent(text(startup))].join("\n"));
    expect(readCodexTranscript(file)).toEqual([{ role: "user", content: startup }]);
  });

  it("keeps only the text of a UserMessage with images, minus injected memory", () => {
    writeFileSync(
      file,
      userEvent(
        { type: "local_image", path: "/tmp/x.png" },
        text("Explain the image. <hindsight_memories>leak</hindsight_memories>")
      )
    );
    expect(readCodexTranscript(file)).toEqual([{ role: "user", content: "Explain the image." }]);
  });

  it("without UserMessage events (older Codex): keeps user/assistant text + compact action turns; drops developer/synthetic/reasoning/outputs/injected", () => {
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

  it("preserves event timestamps on user, assistant, and function-call turns", () => {
    writeFileSync(
      file,
      [
        JSON.stringify({
          type: "event_msg",
          timestamp: "2026-01-02T10:00:00Z",
          payload: {
            type: "item_completed",
            item: { type: "UserMessage", content: [text("inspect the service")] },
          },
        }),
        JSON.stringify({
          type: "response_item",
          timestamp: "2026-01-02T10:00:02Z",
          payload: {
            type: "function_call",
            name: "exec",
            arguments: JSON.stringify({ command: "status" }),
          },
        }),
        JSON.stringify({
          type: "response_item",
          timestamp: "2026-01-02T10:00:04Z",
          payload: {
            type: "message",
            role: "assistant",
            content: [text("The service is healthy.")],
          },
        }),
      ].join("\n")
    );
    expect(readCodexTranscript(file)).toEqual([
      { role: "user", content: "inspect the service", timestamp: "2026-01-02T10:00:00Z" },
      { role: "action", content: "exec status", timestamp: "2026-01-02T10:00:02Z" },
      { role: "assistant", content: "The service is healthy.", timestamp: "2026-01-02T10:00:04Z" },
    ]);
  });

  it("omits a non-string timestamp rather than passing it through", () => {
    // A rollout line is unvalidated JSON. An envelope time that is not a string must not
    // reach the turn: chat.ts writes whatever is there verbatim into the retained JSONL.
    writeFileSync(
      file,
      JSON.stringify({
        type: "response_item",
        timestamp: 1767348000000,
        payload: { type: "message", role: "assistant", content: [text("done")] },
      })
    );
    expect(readCodexTranscript(file)).toEqual([{ role: "assistant", content: "done" }]);
  });

  it("normalizes current custom_tool_call records as compact action turns", () => {
    writeFileSync(
      file,
      [
        item({
          type: "custom_tool_call",
          name: "exec",
          input: JSON.stringify({ command: "systemctl status service" }),
          call_id: "call_1",
        }),
        item({
          type: "custom_tool_call_output",
          call_id: "call_1",
          output: [{ type: "text", text: "active" }],
        }),
      ].join("\n")
    );
    expect(readCodexTranscript(file)).toEqual([
      { role: "action", content: "exec systemctl status service" },
    ]);
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
