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

const desktopStartup = [
  "<recommended_plugins>\nUse available tools when relevant.\n</recommended_plugins>",
  "# AGENTS.md instructions\n<INSTRUCTIONS>Follow project conventions.</INSTRUCTIONS>",
  "<environment_context>\n<cwd>/example</cwd>\n</environment_context>",
].join("\n\n");

describe("readCodexTranscript", () => {
  it.each<[string, number[]]>([
    ["missing AGENTS", [0, 2]],
    ["missing plugins", [1, 2]],
    ["plugins only", [0]],
  ])("filters metadata startup blocks: %s", (_label, indices) => {
    const texts = desktopStartup.split("\n\n");
    const kinds = [
      "plugins.recommendations",
      "agents_md.instructions",
      "environments.environment_context",
    ];
    writeFileSync(
      file,
      item({
        type: "message",
        role: "user",
        content: indices.map((index) => ({ type: "input_text", text: texts[index] })),
        internal_chat_message_metadata_passthrough: {
          content_item_kinds: indices.map((index) => kinds[index]),
        },
      })
    );
    expect(readCodexTranscript(file)).toEqual([]);
  });

  it.each([
    ["user.text", desktopStartup],
    ["user.text", "# AGENTS.md instructions for /example\nPlease explain these rules."],
    ["user.text", "<environment_context>Discuss this example.</environment_context>"],
    ["unknown.future_kind", desktopStartup],
    ["skills.selected_skill_instructions", desktopStartup],
    ["goal.internal_context", desktopStartup],
  ])("preserves %s even when its text resembles startup", (kind, text) => {
    writeFileSync(
      file,
      item({
        type: "message",
        role: "user",
        content: [{ type: "input_text", text }],
        internal_chat_message_metadata_passthrough: { content_item_kinds: [kind] },
      })
    );
    expect(readCodexTranscript(file)).toEqual([{ role: "user", content: text }]);
  });

  it("keeps metadata aligned around images and preserves genuine text in a mixed message", () => {
    writeFileSync(
      file,
      item({
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: "synthetic-image" },
          { type: "input_text", text: "injected plugin guidance" },
          {
            type: "input_text",
            text: "Explain the image. <hindsight_memories>injected</hindsight_memories>",
          },
          { type: "input_text", text: "injected environment" },
          { type: "input_text", text: "Keep this follow-up." },
        ],
        internal_chat_message_metadata_passthrough: {
          content_item_kinds: [
            "user.image",
            "plugins.recommendations",
            "user.text",
            "environments.environment_context",
            "user.text",
          ],
        },
      })
    );
    expect(readCodexTranscript(file)).toEqual([
      { role: "user", content: "Explain the image. \nKeep this follow-up." },
    ]);
  });

  it.each([
    null,
    {},
    { content_item_kinds: "user.text" },
    { content_item_kinds: [] },
    { content_item_kinds: ["user.text", "user.text"] },
    { content_item_kinds: [null] },
    { content_item_kinds: [""] },
  ])("falls back to legacy detection for invalid metadata: %j", (metadata) => {
    writeFileSync(
      file,
      [desktopStartup, "Please explain the rules."]
        .map((text) =>
          item({
            type: "message",
            role: "user",
            content: [{ type: "input_text", text }],
            internal_chat_message_metadata_passthrough: metadata,
          })
        )
        .join("\n")
    );
    expect(readCodexTranscript(file)).toEqual([
      { role: "user", content: "Please explain the rules." },
    ]);
  });

  it.each(["commentary", "final_answer", undefined])(
    "ignores startup origin labels on assistant %s",
    (phase) => {
      writeFileSync(
        file,
        item({
          type: "message",
          role: "assistant",
          phase,
          content: [{ type: "output_text", text: desktopStartup }],
          internal_chat_message_metadata_passthrough: {
            content_item_kinds: ["plugins.recommendations"],
          },
        })
      );
      expect(readCodexTranscript(file)).toEqual([{ role: "assistant", content: desktopStartup }]);
    }
  );

  it("excludes the complete Desktop startup envelope before the first genuine user message", () => {
    writeFileSync(
      file,
      [
        item({
          type: "message",
          role: "user",
          content: desktopStartup.split("\n\n").map((text) => ({ type: "input_text", text })),
        }),
        item({
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "What is 2 + 2?" }],
        }),
      ].join("\n")
    );
    expect(readCodexTranscript(file)).toEqual([{ role: "user", content: "What is 2 + 2?" }]);
  });

  it.each([
    "Explain how AGENTS.md, <recommended_plugins>, and <environment_context> are used.",
    `<recommended_plugins>\nExample plugin guidance.\n</recommended_plugins>\nWhat does this mean?`,
    `Please explain this example:\n\n\`\`\`text\n${desktopStartup}\n\`\`\``,
    `${desktopStartup}\nPlease review the instructions above.`,
    `${desktopStartup}\nPlease compare this context:\n<environment_context><cwd>/other</cwd></environment_context>`,
    "<recommended_plugins>\nAn unfinished example",
  ])("preserves genuine user prose and quoted startup examples: %s", (text) => {
    writeFileSync(
      file,
      item({ type: "message", role: "user", content: [{ type: "input_text", text }] })
    );
    expect(readCodexTranscript(file)).toEqual([{ role: "user", content: text }]);
  });

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
