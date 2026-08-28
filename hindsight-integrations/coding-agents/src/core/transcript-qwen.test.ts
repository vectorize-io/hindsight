import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { readQwenTranscript } from "./transcript-qwen";
import { stripInjectedMemory } from "./transcript-util";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-qwen-transcript-"));
  file = join(root, "0fda5621-cda9-46c2-89dd-b9e4188a7b54.jsonl");
});
afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

/** One record with Qwen's Claude-shaped envelope; `fields` carries what each case is about. */
const record = (fields: Record<string, unknown>) =>
  JSON.stringify({
    uuid: "1e65de67-abad-4509-8998-d2d7409d14d0",
    parentUuid: "59208e3b-d0b0-4472-9258-07bbedea4124",
    sessionId: "0fda5621-cda9-46c2-89dd-b9e4188a7b54",
    cwd: "/data/repos/qwen38-qua-moe",
    version: "0.22.0",
    gitBranch: "research/cpt-downcycling-phase6",
    ...fields,
  });

/** Exactly what Qwen writes back after a UserPromptSubmit hook returns additionalContext: its own
 *  wrapper around our block, with the INNER tags HTML-escaped by the host. */
const injectedEcho = [
  "<qwen:user-prompt-submit-context>",
  "&lt;hindsight_memories&gt;",
  "Relevant memories from past conversations (prioritize recent when conflicting).",
  "- the uploader retries with exponential backoff [experience] (2026-08-22T08:56:50Z)",
  "&lt;/hindsight_memories&gt;",
  "</qwen:user-prompt-submit-context>",
].join("\n");

describe("readQwenTranscript", () => {
  it("keeps real user prompts, assistant prose and compact action turns; drops system/tool_result/thought/synthetic-user records", () => {
    const lines = [
      // system telemetry — 54% of a real transcript, and never conversation: dropped
      record({
        type: "system",
        provenance: "system",
        subtype: "ui_telemetry",
        systemPayload: {
          uiEvent: {
            "event.name": "qwen-code.api_response",
            model: "qwen/qwen3.8-max",
            status_code: 200,
            duration_ms: 4193,
          },
        },
      }),
      // real user prompt: kept
      record({
        type: "user",
        provenance: "real_user",
        timestamp: "2026-08-24T06:07:26.836Z",
        message: { role: "user", parts: [{ text: "Reply with exactly: probe-ok" }] },
      }),
      // assistant: the `thought` part is reasoning (dropped), the plain text is the answer (kept).
      // Note message.role is Gemini's "model" — role comes from the record's `type`, not from here.
      record({
        type: "assistant",
        provenance: "assistant_output",
        timestamp: "2026-08-24T06:07:31.074Z",
        model: "qwen/qwen3.8-max",
        message: {
          role: "model",
          parts: [
            {
              text: 'The user wants me to reply with exactly "probe-ok". Let me do that.',
              thought: true,
            },
            { text: "probe-ok" },
          ],
        },
        usageMetadata: { promptTokenCount: 40485, candidatesTokenCount: 34 },
      }),
      // a functionCall-only assistant record becomes a compact role:"action" turn (name + target)
      record({
        type: "assistant",
        provenance: "assistant_output",
        message: {
          role: "model",
          parts: [
            {
              functionCall: {
                id: "call_16257430bada4eef815199fc",
                name: "list_directory",
                args: { path: "/data/repos" },
              },
            },
          ],
        },
      }),
      // tool_result is its OWN record type, with message.role "user": dropped whole
      record({
        type: "tool_result",
        provenance: "tool_result",
        message: {
          role: "user",
          parts: [
            {
              functionResponse: {
                id: "call_16257430bada4eef815199fc",
                name: "list_directory",
                response: { output: "Listed 181 item(s) in /data/repos:\n---\n[DIR] bdh" },
              },
            },
          ],
        },
      }),
      // `type:"user"` but MACHINE-authored: a background-task notification. Dropped on provenance —
      // these outnumbered genuine prompts 307 to 69 in the corpus this reader was measured on.
      record({
        type: "user",
        provenance: "system",
        subtype: "notification",
        message: {
          role: "user",
          parts: [
            {
              text:
                "<task-notification>\n<task-id>general-purpose-call_f1b126b417f3491da3edc4c3</task-id>\n" +
                '<status>completed</status>\n<summary>Agent "Survey KV-cache prior art" completed.</summary>\n' +
                "</task-notification>",
            },
          ],
        },
      }),
      // same class: a /loop cron re-entry, written as a user record
      record({
        type: "user",
        provenance: "system",
        subtype: "cron",
        message: {
          role: "user",
          parts: [
            {
              text: "Base directory for this skill: /qwen-code/lib/bundled/loop",
            },
          ],
        },
      }),
      // a human interjection mid-turn IS real: provenance says so, subtype does not disqualify it
      record({
        type: "user",
        provenance: "real_user",
        subtype: "mid_turn_user_message",
        timestamp: "2026-08-22T00:23:23.004Z",
        message: {
          role: "user",
          parts: [{ text: "Also, install whatever vast tooling you want." }],
        },
        systemPayload: { displayText: "Also, install whatever vast tooling you want." },
      }),
      // malformed line + non-object JSON + blank line: tolerated
      "{ not json",
      "null",
      "",
    ];
    writeFileSync(file, lines.join("\n"));

    const turns = readQwenTranscript(file);

    expect(turns).toEqual([
      {
        role: "user",
        content: "Reply with exactly: probe-ok",
        timestamp: "2026-08-24T06:07:26.836Z",
      },
      { role: "assistant", content: "probe-ok", timestamp: "2026-08-24T06:07:31.074Z" },
      { role: "action", content: "list_directory /data/repos" },
      {
        role: "user",
        content: "Also, install whatever vast tooling you want.",
        timestamp: "2026-08-22T00:23:23.004Z",
      },
    ]);
  });

  it("splits a mixed text + functionCall record into a prose turn plus one action turn each", () => {
    writeFileSync(
      file,
      record({
        type: "assistant",
        provenance: "assistant_output",
        timestamp: "2026-08-22T00:24:02.117Z",
        message: {
          role: "model",
          parts: [
            { text: "Checking the trainer, then running it.", thought: true },
            { text: "Editing the stream, then running the probe." },
            {
              functionCall: {
                id: "call_a1",
                name: "edit",
                args: {
                  file_path: "/data/repos/bdh-attention/bdh_kv/stream.py",
                  old_string: "steps.append(step)",
                  new_string: "steps.append(step)  # noqa",
                },
              },
            },
            {
              functionCall: {
                id: "call_a2",
                name: "run_shell_command",
                args: { command: "python -m pytest -q", description: "Run the probe" },
              },
            },
          ],
        },
      })
    );

    // Action lines carry the tool name + primary target only — no arguments, no output.
    expect(readQwenTranscript(file)).toEqual([
      {
        role: "assistant",
        content: "Editing the stream, then running the probe.",
        timestamp: "2026-08-22T00:24:02.117Z",
      },
      {
        role: "action",
        content: "edit /data/repos/bdh-attention/bdh_kv/stream.py",
        timestamp: "2026-08-22T00:24:02.117Z",
      },
      {
        role: "action",
        content: "run_shell_command python -m pytest -q",
        timestamp: "2026-08-22T00:24:02.117Z",
      },
    ]);
  });

  it("strips Qwen's echo of our own injection, which the shared tag stripper cannot see because the host HTML-escapes the inner tags", () => {
    // The premise of the harness-specific strip, asserted rather than assumed: MEMORY_TAG_RE
    // matches RAW tags, and every injection Qwen writes back is escaped.
    expect(stripInjectedMemory(injectedEcho)).toBe(injectedEcho);

    // The real shape, and the only one with pairing evidence: the user's prompt part, then the
    // echo part appended by the host as the FINAL part. Only the prompt survives — otherwise
    // every Stop re-ingests the memories that turn's recall injected.
    writeFileSync(
      file,
      record({
        type: "user",
        provenance: "real_user",
        timestamp: "2026-08-24T06:07:26.836Z",
        message: {
          role: "user",
          parts: [{ text: "Reply with exactly: probe-ok" }, { text: injectedEcho }],
        },
      })
    );
    expect(readQwenTranscript(file)).toEqual([
      {
        role: "user",
        content: "Reply with exactly: probe-ok",
        timestamp: "2026-08-24T06:07:26.836Z",
      },
    ]);
  });

  it("prefers systemPayload.displayText, the host's own pre-hook projection, over parsing parts", () => {
    // The primary evidence in Qwen's contract: when `hookContext` is a string, `displayText` is
    // the pre-hook prompt and "never includes the hook context". No tag matching is involved, so
    // this path is immune to whatever the user happened to type.
    writeFileSync(
      file,
      record({
        type: "user",
        provenance: "real_user",
        timestamp: "2026-08-24T06:07:26.836Z",
        systemPayload: { displayText: "what did we decide about vchord?", hookContext: "…" },
        message: {
          role: "user",
          parts: [{ text: "expanded pre-hook prompt" }, { text: injectedEcho }],
        },
      })
    );
    expect(readQwenTranscript(file)).toEqual([
      {
        role: "user",
        content: "what did we decide about vchord?",
        timestamp: "2026-08-24T06:07:26.836Z",
      },
    ]);
  });

  it("does NOT treat tag-like text the user actually wrote as hook provenance", () => {
    // Qwen's contract is explicit that the tag is "a provenance marker, not ... a general trust
    // boundary" and that consumers "must not infer that arbitrary tag-like user text is hook
    // provenance". An unanchored global regex over every part deletes real prose — and in a repo
    // whose subject IS this integration, quoting the tag is a prompt someone will really write.
    const quoting =
      "why does <qwen:user-prompt-submit-context>foo</qwen:user-prompt-submit-context> show up twice?";
    writeFileSync(
      file,
      record({
        type: "user",
        provenance: "real_user",
        message: { role: "user", parts: [{ text: quoting }] },
      })
    );
    expect(readQwenTranscript(file)).toEqual([{ role: "user", content: quoting }]);

    // Same text mid-record, with a genuine echo appended after it: the echo goes, the prose stays.
    writeFileSync(
      file,
      record({
        type: "user",
        provenance: "real_user",
        message: { role: "user", parts: [{ text: quoting }, { text: injectedEcho }] },
      })
    );
    expect(readQwenTranscript(file)).toEqual([{ role: "user", content: quoting }]);
  });

  it("strips injected memory that arrives unescaped inside a kept turn", () => {
    writeFileSync(
      file,
      record({
        type: "user",
        provenance: "real_user",
        message: {
          role: "user",
          parts: [{ text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?" }],
        },
      })
    );
    expect(readQwenTranscript(file)).toEqual([{ role: "user", content: "Why retry?" }]);
  });

  it("a functionCall whose args name no primary target still yields the bare tool name", () => {
    writeFileSync(
      file,
      record({
        type: "assistant",
        provenance: "assistant_output",
        message: {
          role: "model",
          parts: [
            {
              functionCall: {
                id: "call_053a278ceaee48b99111b615",
                name: "todo_write",
                args: { todos: [{ id: "1", content: "Survey prior art", status: "in_progress" }] },
              },
            },
          ],
        },
      })
    );
    expect(readQwenTranscript(file)).toEqual([{ role: "action", content: "todo_write" }]);
  });

  it("drops tool_result records entirely — even a huge one produces no turn", () => {
    writeFileSync(
      file,
      record({
        type: "tool_result",
        provenance: "tool_result",
        message: {
          role: "user",
          parts: [
            {
              functionResponse: {
                id: "call_1",
                name: "run_shell_command",
                response: { output: "x".repeat(5000) },
              },
            },
          ],
        },
      })
    );
    expect(readQwenTranscript(file)).toEqual([]);
  });

  it("requires a positive subagent marker — a bare record with no provenance is NOT a user turn", () => {
    // The previous version of this test passed unchanged with agentId/isSidechain removed, so it
    // never tested the boundary it documented. Absence of `subtype` is not evidence of a human:
    // a field-stripped or corrupt record has neither. Only the subagent envelope earns the
    // fallback, and a malformed `provenance` must never reach it at all.
    const bare = JSON.stringify({
      type: "user",
      message: { role: "user", parts: [{ text: "no provenance, no envelope" }] },
    });
    writeFileSync(file, bare + "\n");
    expect(readQwenTranscript(file)).toEqual([]);

    for (const bad of [null, 42, { kind: "real_user" }, ["real_user"]]) {
      writeFileSync(
        file,
        JSON.stringify({
          type: "user",
          provenance: bad,
          agentId: "general-purpose-call_x",
          isSidechain: true,
          message: { role: "user", parts: [{ text: "malformed provenance" }] },
        }) + "\n"
      );
      expect(readQwenTranscript(file), `provenance=${JSON.stringify(bad)}`).toEqual([]);
    }
  });

  it("reads a subagent transcript, whose envelope carries no provenance at all", () => {
    // ~/.qwen/projects/<slug>/subagents/<parentSessionId>/agent-<name>-<callId>.jsonl uses a THIRD
    // envelope: agentId/agentName/isSidechain, no `provenance` and no `subtype`. A strict
    // provenance === "real_user" test would read every one of these as synthetic and retain
    // nothing, so the reader falls back the way Qwen's own legacySafeProvenance does.
    const subagent = (fields: Record<string, unknown>) =>
      JSON.stringify({
        uuid: "d656e8f0-97ea-4d06-908a-51cb169b73bd",
        parentUuid: null,
        sessionId: "b3cbbf83-fea5-4308-8571-24d6f9c6bf47",
        cwd: "/data/repos/bdh-attention",
        version: "0.21.12",
        agentId: "general-purpose-call_f1b126b417f3491da3edc4c3",
        agentName: "general-purpose",
        isSidechain: true,
        ...fields,
      });
    writeFileSync(
      file,
      [
        subagent({
          type: "user",
          timestamp: "2026-08-14T22:05:21.828Z",
          message: { role: "user", parts: [{ text: "Survey the KV-cache prior art." }] },
        }),
        subagent({
          type: "assistant",
          timestamp: "2026-08-14T22:05:33.294Z",
          message: {
            role: "model",
            parts: [
              { text: "Planning the fetches.", thought: true },
              { text: "Fetching the 7 sources." },
              {
                functionCall: {
                  id: "call_c18036a2c14d489e81968305",
                  name: "web_fetch",
                  args: {
                    url: "https://proceedings.mlr.press/v235/nawrot24a.html",
                    prompt: "summarize",
                  },
                },
              },
            ],
          },
        }),
      ].join("\n")
    );

    expect(readQwenTranscript(file)).toEqual([
      {
        role: "user",
        content: "Survey the KV-cache prior art.",
        timestamp: "2026-08-14T22:05:21.828Z",
      },
      {
        role: "assistant",
        content: "Fetching the 7 sources.",
        timestamp: "2026-08-14T22:05:33.294Z",
      },
      {
        role: "action",
        content: "web_fetch https://proceedings.mlr.press/v235/nawrot24a.html",
        timestamp: "2026-08-14T22:05:33.294Z",
      },
    ]);
  });

  it("fails open (returns []) when the file cannot be read", () => {
    expect(readQwenTranscript(join(root, "nope.jsonl"))).toEqual([]);
  });

  it("fails open when transcript_path is a DIRECTORY — the fault is on the LAZY read", () => {
    // Not the same case as a missing file, and the one that actually happens: on Linux a directory
    // passes statSync AND openSync, then throws EISDIR on the first readSync — i.e. after the
    // reader has already returned its generator. runRetainHook calls the reader outside
    // buildRetain's catch, so an uncaught throw there rejects the whole Stop hook.
    expect(() => readQwenTranscript(root)).not.toThrow();
    expect(readQwenTranscript(root)).toEqual([]);
  });
});
