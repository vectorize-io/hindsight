import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readCodexTranscript } from "./transcript-codex";
import { memoryUsageCursorStore, recordUsage, summarizeTurns } from "./usage";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-codex-transcript-"));
  file = join(root, "rollout.jsonl");
});
afterEach(() => {
  vi.unstubAllEnvs();
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

  describe("code mode (exec wrapper + McpToolCall/CommandExecution items)", () => {
    const at = (timestamp: string, line: string) =>
      JSON.stringify({ ...JSON.parse(line), timestamp });
    const toolItem = (i: Record<string, unknown>) =>
      JSON.stringify({ type: "event_msg", payload: { type: "item_completed", item: i } });
    const execCall = (callId: string, input: string) =>
      item({ type: "custom_tool_call", name: "exec", call_id: callId, input });

    const rollout = () =>
      [
        at("2026-01-02T10:00:00Z", userEvent(text("how do we retry uploads?"))),
        at(
          "2026-01-02T10:00:01Z",
          execCall(
            "call_1",
            'text(await tools.mcp__hindsight__hindsight_search_knowledge_pages({query:"retry policy"}));\n' +
              'text(await tools.exec_command({cmd:"git status --short"}));\n'
          )
        ),
        at(
          "2026-01-02T10:00:02Z",
          toolItem({
            type: "McpToolCall",
            server: "hindsight",
            tool: "hindsight_search_knowledge_pages",
            arguments: { query: "retry policy" },
            result: { content: [{ type: "text", text: "RESULT BODY" }] },
            status: "completed",
          })
        ),
        at(
          "2026-01-02T10:00:03Z",
          toolItem({
            type: "CommandExecution",
            command: ["/bin/zsh", "-lc", "git status --short\necho second line"],
            cwd: "file:///repo",
            aggregated_output: "M src/upload.ts",
            stdout: "M src/upload.ts",
            exit_code: 0,
            status: "completed",
          })
        ),
        item({ type: "custom_tool_call_output", call_id: "call_1", output: "M src/upload.ts" }),
        item({ type: "function_call", name: "sleep", arguments: '{"id":"timer"}' }),
        at(
          "2026-01-02T10:00:04Z",
          item({
            type: "message",
            role: "assistant",
            content: [text("From Hindsight memory: retries use backoff.")],
          })
        ),
        at("2026-01-02T10:00:05Z", execCall("call_2", "text(1 + 1);\n")),
        at(
          "2026-01-02T10:00:06Z",
          toolItem({
            type: "CommandExecution",
            command: ["npm", "test"],
            status: "failed",
            stderr: "1 failed",
          })
        ),
      ].join("\n");

    it("derives action turns from the items, in rollout order, without outputs or exec wrappers", () => {
      writeFileSync(file, rollout());
      expect(readCodexTranscript(file)).toEqual([
        {
          role: "user",
          content: "how do we retry uploads?",
          timestamp: "2026-01-02T10:00:00Z",
        },
        {
          role: "action",
          content: "mcp__hindsight__hindsight_search_knowledge_pages retry policy",
          timestamp: "2026-01-02T10:00:02Z",
        },
        {
          role: "action",
          content: "exec_command git status --short",
          timestamp: "2026-01-02T10:00:03Z",
        },
        { role: "action", content: "sleep timer" },
        {
          role: "assistant",
          content: "From Hindsight memory: retries use backoff.",
          timestamp: "2026-01-02T10:00:04Z",
        },
        { role: "action", content: "exec_command npm test", timestamp: "2026-01-02T10:00:06Z" },
      ]);
    });

    it("counts the Codex hindsight_* MCP call in summarizeTurns and recordUsage", () => {
      writeFileSync(file, rollout());
      const turns = readCodexTranscript(file);
      expect(summarizeTurns(turns)).toEqual([
        { turn: 1, calls: ["hindsight_search_knowledge_pages"], credited: true },
      ]);

      const usageFile = join(root, "usage.jsonl");
      vi.stubEnv("HINDSIGHT_USAGE_FILE", usageFile);
      recordUsage({
        harness: "codex",
        sessionId: "s",
        bankId: "b",
        turns,
        cursors: memoryUsageCursorStore(),
        lastTurnComplete: true,
      });
      expect(JSON.parse(readFileSync(usageFile, "utf8"))).toMatchObject({
        harness: "codex",
        turn: 1,
        calls: ["hindsight_search_knowledge_pages"],
        credited: true,
      });
    });

    it("accepts JSON-string MCP arguments and a bare tool name without a server", () => {
      writeFileSync(
        file,
        [
          toolItem({
            type: "McpToolCall",
            tool: "hindsight_reflect",
            arguments: '{"query":"why backoff"}',
          }),
          toolItem({ type: "McpToolCall", server: "docs", tool: "fetch", arguments: "not json" }),
        ].join("\n")
      );
      expect(readCodexTranscript(file)).toEqual([
        { role: "action", content: "hindsight_reflect why backoff" },
        { role: "action", content: "mcp__docs__fetch" },
      ]);
    });

    it("keeps the call-record path for exec calls when the rollout has no tool items", () => {
      writeFileSync(
        file,
        [
          execCall("call_1", 'text(await tools.exec_command({cmd:"ls"}));'),
          execCall("call_2", JSON.stringify({ command: "systemctl status service" })),
        ].join("\n")
      );
      expect(readCodexTranscript(file)).toEqual([
        { role: "action", content: "exec" },
        { role: "action", content: "exec systemctl status service" },
      ]);
    });
  });
});
