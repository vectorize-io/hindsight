import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { kimiSessionDir, readKimiTranscript, readKimiWire } from "./transcript-kimi";
import type { KimiWireEvent } from "./transcript-kimi";

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-kimi-transcript-"));
});
afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const at = (iso: string) => Date.parse(iso);

/** Every conversational record is nested inside this envelope; only the inner `event` differs. */
const loop = (event: Record<string, unknown>, time: number, agentId = "main") => ({
  type: "context.append_loop_event",
  agentId,
  event,
  time,
});

/** Kimi's loop groups an assistant message's parts and calls by `stepUuid` — real ids from a
 *  captured session, so the grouping under test is the grouping the CLI actually writes. */
const STEP_1 = "7561ceca-514a-4d03-8a56-deee632af302";
const STEP_2 = "74c82ed8-5dae-4724-959b-d7137e2ae963";

const writeWire = (agentId: string, records: unknown[]): void => {
  const dir = join(root, "agents", agentId);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "wire.jsonl"), records.map((r) => JSON.stringify(r)).join("\n"));
};

describe("readKimiWire", () => {
  it("keeps the human prompt, one assistant turn per step and compact action turns; drops reasoning, tool output and telemetry", () => {
    const events: KimiWireEvent[] = [
      // header + binding/telemetry records: dropped
      { type: "metadata", time: at("2026-08-21T09:18:32.558Z") } as KimiWireEvent,
      { type: "runtime.set_binding", time: at("2026-08-21T09:18:32.560Z") } as KimiWireEvent,
      // the human prompt: kept
      {
        type: "turn.prompt",
        input: [{ type: "text", text: "Adrian Kosowski is the first target author." }],
        origin: { kind: "user" },
        time: at("2026-08-21T09:18:32.595Z"),
      },
      // loop bookkeeping: dropped
      loop(
        { type: "step.begin", uuid: STEP_1, turnId: "0", step: 1 },
        at("2026-08-21T09:18:32.602Z")
      ),
      // reasoning: dropped, like Claude `thinking` and Codex `reasoning`
      loop(
        {
          type: "content.part",
          turnId: "0",
          step: 1,
          stepUuid: STEP_1,
          part: { type: "think", think: "Let me start by reading SYSTEM.md." },
        },
        at("2026-08-21T09:18:42.770Z")
      ),
      // the visible answer for step 1: kept
      loop(
        {
          type: "content.part",
          turnId: "0",
          step: 1,
          stepUuid: STEP_1,
          part: { type: "text", text: "I'll start by reading the project files." },
        },
        at("2026-08-21T09:18:42.771Z")
      ),
      // two tool calls in the same step: one compact action turn each, `args` already an object
      loop(
        {
          type: "tool.call",
          turnId: "0",
          step: 1,
          stepUuid: STEP_1,
          toolCallId: "tool_YLfojO2Sivr4si0SbsvnlwAW",
          name: "Read",
          args: { path: "/repos/example-project/SYSTEM.md" },
        },
        at("2026-08-21T09:18:42.798Z")
      ),
      loop(
        {
          type: "tool.call",
          turnId: "0",
          step: 1,
          stepUuid: STEP_1,
          toolCallId: "tool_u6rVhKjfW1wZWuVTREdMBqqH",
          name: "Bash",
          args: { command: "ls -la /repos/example-project" },
        },
        at("2026-08-21T09:18:42.798Z")
      ),
      // tool output: dropped — mechanical noise for extraction
      loop(
        {
          type: "tool.result",
          parentUuid: "f99b1eb4-c4cf-4e07-a49c-2bf8a2f35ed6",
          toolCallId: "tool_YLfojO2Sivr4si0SbsvnlwAW",
          result: { output: "1\tYou are the research-ingestion agent." },
        },
        at("2026-08-21T09:18:42.810Z")
      ),
      loop(
        { type: "step.end", uuid: STEP_1, turnId: "0", step: 1, finishReason: "tool_use" },
        at("2026-08-21T09:18:42.823Z")
      ),
      // step 2's answer: its own assistant turn
      loop(
        {
          type: "content.part",
          turnId: "0",
          step: 2,
          stepUuid: STEP_2,
          part: { type: "text", text: "DBLP gives 134 records. Cross-checking arXiv next." },
        },
        at("2026-08-21T09:19:19.113Z")
      ),
      // per-request telemetry: dropped
      { type: "usage.record", time: at("2026-08-21T09:19:19.200Z") } as KimiWireEvent,
      { type: "token_counting.measured", time: at("2026-08-21T09:19:19.201Z") } as KimiWireEvent,
      { type: "turn.ended", time: at("2026-08-21T09:19:20.000Z") } as KimiWireEvent,
    ];

    expect(readKimiWire(events)).toEqual([
      {
        role: "user",
        content: "Adrian Kosowski is the first target author.",
        timestamp: "2026-08-21T09:18:32.595Z",
      },
      {
        role: "assistant",
        content: "I'll start by reading the project files.",
        timestamp: "2026-08-21T09:18:42.771Z",
      },
      {
        role: "action",
        content: "Read /repos/example-project/SYSTEM.md",
        timestamp: "2026-08-21T09:18:42.798Z",
      },
      {
        role: "action",
        content: "Bash ls -la /repos/example-project",
        timestamp: "2026-08-21T09:18:42.798Z",
      },
      {
        role: "assistant",
        content: "DBLP gives 134 records. Cross-checking arXiv next.",
        timestamp: "2026-08-21T09:19:19.113Z",
      },
    ]);
  });

  it("joins one step's streamed text parts into a single assistant turn", () => {
    const events: KimiWireEvent[] = [
      loop(
        {
          type: "content.part",
          stepUuid: STEP_1,
          part: { type: "text", text: "Tools available and DBLP author found." },
        },
        at("2026-08-21T09:19:19.113Z")
      ),
      loop(
        {
          type: "content.part",
          stepUuid: STEP_1,
          part: { type: "text", text: "Now fetching his full DBLP record." },
        },
        at("2026-08-21T09:19:19.180Z")
      ),
    ];

    expect(readKimiWire(events)).toEqual([
      {
        role: "assistant",
        content: "Tools available and DBLP author found.\nNow fetching his full DBLP record.",
        timestamp: "2026-08-21T09:19:19.113Z",
      },
    ]);
  });

  it("drops every machine-authored prompt — only origin.kind 'user' is a human turn", () => {
    const events: KimiWireEvent[] = [
      // the swarm dispatcher briefing a subagent: Kimi talking to itself
      {
        type: "turn.prompt",
        input: [
          {
            type: "text",
            text: "<git-context>…</git-context>\nYou are one of several parallel research agents.",
          },
        ],
        origin: { kind: "system_trigger", name: "subagent" },
        time: at("2026-08-21T18:42:00.000Z"),
      } as KimiWireEvent,
      {
        type: "turn.prompt",
        input: [{ type: "text", text: "Continue the background task." }],
        origin: { kind: "task" },
        time: at("2026-08-21T18:42:01.000Z"),
      } as KimiWireEvent,
      {
        type: "turn.prompt",
        input: [{ type: "text", text: "Activating skill: research." }],
        origin: { kind: "skill_activation" },
        time: at("2026-08-21T18:42:02.000Z"),
      } as KimiWireEvent,
      {
        type: "turn.prompt",
        input: [{ type: "text", text: "what is the most common failure condition?" }],
        origin: { kind: "user" },
        time: at("2026-08-21T18:42:03.000Z"),
      },
    ];

    expect(readKimiWire(events)).toEqual([
      {
        role: "user",
        content: "what is the most common failure condition?",
        timestamp: "2026-08-21T18:42:03.000Z",
      },
    ]);
  });

  it("strips injected memory that leaks into a genuine prompt", () => {
    const events: KimiWireEvent[] = [
      {
        type: "turn.prompt",
        input: [
          { type: "text", text: "<hindsight_memories>\nleak\n</hindsight_memories>\nWhy retry?" },
        ],
        origin: { kind: "user" },
        time: at("2026-08-21T09:18:32.595Z"),
      },
    ];
    expect(readKimiWire(events)).toEqual([
      { role: "user", content: "Why retry?", timestamp: "2026-08-21T09:18:32.595Z" },
    ]);
  });

  it("a tool call whose args carry no recognised target still yields the bare tool name", () => {
    const events: KimiWireEvent[] = [
      loop(
        { type: "tool.call", stepUuid: STEP_1, name: "TodoList", args: { todos: [] } },
        at("2026-08-21T09:18:42.798Z")
      ),
    ];
    expect(readKimiWire(events)).toEqual([
      { role: "action", content: "TodoList", timestamp: "2026-08-21T09:18:42.798Z" },
    ]);
  });

  it("drops tool output entirely — even a huge one produces no turn", () => {
    const events: KimiWireEvent[] = [
      loop(
        { type: "tool.result", toolCallId: "tool_1", result: { output: "x".repeat(5000) } },
        at("2026-08-21T09:18:42.810Z")
      ),
    ];
    expect(readKimiWire(events)).toEqual([]);
  });

  it("ignores unknown types, unnested loop names and malformed entries", () => {
    const events = [
      null,
      "not an event",
      // the five conversational names are NESTED: at the top level they mean nothing
      { type: "content.part", part: { type: "text", text: "not where this lives" }, time: 1 },
      { type: "tool.call", name: "Bash", args: { command: "echo hi" }, time: 2 },
      { type: "context.append_loop_event", time: 3 },
      { type: "context.append_loop_event", event: null, time: 4 },
      { type: "turn.prompt", origin: { kind: "user" }, time: 5 },
      { type: "staleGuard.recorded", path: "/repo/AGENTS.md", time: 6 },
    ] as unknown as KimiWireEvent[];

    expect(readKimiWire(events)).toEqual([]);
  });

  it("survives an empty or absent log", () => {
    expect(readKimiWire([])).toEqual([]);
    expect(readKimiWire(undefined as unknown as KimiWireEvent[])).toEqual([]);
  });
});

describe("readKimiTranscript", () => {
  it("reads every agent's wire log in agent order — main first, then agent-<n> numerically", () => {
    writeWire("main", [
      {
        type: "turn.prompt",
        input: [{ type: "text", text: "research the AMD decode path" }],
        origin: { kind: "user" },
        time: at("2026-08-21T18:42:00.000Z"),
      },
      loop(
        {
          type: "tool.call",
          stepUuid: STEP_1,
          name: "Agent",
          args: { description: "Deep research: decode kernel fusion", subagent_type: "research" },
        },
        at("2026-08-21T18:42:10.000Z")
      ),
    ]);
    // `agent-10` must not sort before `agent-2`, which is what a lexicographic sort does.
    writeWire("agent-2", [
      loop(
        {
          type: "content.part",
          stepUuid: STEP_1,
          part: { type: "text", text: "agent 2 reporting" },
        },
        at("2026-08-21T18:43:00.000Z"),
        "agent-2"
      ),
    ]);
    writeWire("agent-10", [
      loop(
        {
          type: "content.part",
          stepUuid: STEP_1,
          part: { type: "text", text: "agent 10 reporting" },
        },
        at("2026-08-21T18:44:00.000Z"),
        "agent-10"
      ),
    ]);
    // a stray file next to the agent directories must not be mistaken for an agent
    writeFileSync(join(root, "agents", "index.json"), "{}");

    expect(readKimiTranscript(root)).toEqual([
      {
        role: "user",
        content: "research the AMD decode path",
        timestamp: "2026-08-21T18:42:00.000Z",
      },
      { role: "action", content: "Agent", timestamp: "2026-08-21T18:42:10.000Z" },
      { role: "assistant", content: "agent 2 reporting", timestamp: "2026-08-21T18:43:00.000Z" },
      { role: "assistant", content: "agent 10 reporting", timestamp: "2026-08-21T18:44:00.000Z" },
    ]);
  });

  it("ignores context.append_message, so a fork never replays its parent and an injection is never retained", () => {
    writeWire("main", [
      {
        type: "turn.prompt",
        input: [{ type: "text", text: "Adrian Kosowski is the first target author." }],
        origin: { kind: "user" },
        time: at("2026-08-21T09:18:32.595Z"),
      },
      // Kimi mirrors every prompt into the context channel; reading it would double the turn.
      {
        type: "context.append_message",
        message: {
          role: "user",
          content: [{ type: "text", text: "Adrian Kosowski is the first target author." }],
          toolCalls: [],
          origin: { kind: "user" },
        },
        time: at("2026-08-21T09:18:32.596Z"),
      },
      // our own recall injection comes back on the same channel as origin.kind "hook_result"
      {
        type: "context.append_message",
        message: {
          role: "user",
          content: [{ type: "text", text: "<hindsight_memory>recalled</hindsight_memory>" }],
          toolCalls: [],
          origin: { kind: "hook_result", event: "UserPromptSubmit" },
        },
        time: at("2026-08-21T09:18:32.597Z"),
      },
      loop(
        {
          type: "content.part",
          stepUuid: STEP_1,
          part: { type: "text", text: "I'll start by reading the project files." },
        },
        at("2026-08-21T09:18:42.771Z")
      ),
    ]);
    // A FORKED agent is seeded with a full replay of its parent's conversation as
    // context.append_message records, then runs its own turn as loop events.
    writeWire("agent-0", [
      {
        type: "context.append_message",
        agentId: "agent-0",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "I'll start by reading the project files." }],
          toolCalls: [
            {
              type: "function",
              id: "tool_YLfojO2Sivr4si0SbsvnlwAW",
              name: "Read",
              arguments: '{"path":"/repos/example-project/SYSTEM.md"}',
            },
          ],
        },
        time: at("2026-08-21T11:53:39.657Z"),
      },
      {
        type: "context.append_message",
        agentId: "agent-0",
        message: {
          role: "tool",
          content: [{ type: "text", text: "1\tYou are the research-ingestion agent." }],
        },
        time: at("2026-08-21T11:53:39.658Z"),
      },
      {
        type: "turn.prompt",
        agentId: "agent-0",
        input: [{ type: "text", text: "what tools would help the most here?" }],
        origin: { kind: "user" },
        time: at("2026-08-21T11:53:40.000Z"),
      },
      loop(
        {
          type: "content.part",
          stepUuid: STEP_2,
          part: { type: "text", text: "A DBLP mirror would remove the rate limit." },
        },
        at("2026-08-21T11:53:45.000Z"),
        "agent-0"
      ),
    ]);

    expect(readKimiTranscript(root)).toEqual([
      {
        role: "user",
        content: "Adrian Kosowski is the first target author.",
        timestamp: "2026-08-21T09:18:32.595Z",
      },
      {
        role: "assistant",
        content: "I'll start by reading the project files.",
        timestamp: "2026-08-21T09:18:42.771Z",
      },
      {
        role: "user",
        content: "what tools would help the most here?",
        timestamp: "2026-08-21T11:53:40.000Z",
      },
      {
        role: "assistant",
        content: "A DBLP mirror would remove the rate limit.",
        timestamp: "2026-08-21T11:53:45.000Z",
      },
    ]);
  });

  it("tolerates a malformed line without losing the rest of the log", () => {
    mkdirSync(join(root, "agents", "main"), { recursive: true });
    writeFileSync(
      join(root, "agents", "main", "wire.jsonl"),
      [
        "{ not json",
        "null",
        JSON.stringify({
          type: "turn.prompt",
          input: [{ type: "text", text: "carry on" }],
          origin: { kind: "user" },
          time: at("2026-08-21T09:18:32.595Z"),
        }),
        "",
      ].join("\n")
    );

    expect(readKimiTranscript(root)).toEqual([
      { role: "user", content: "carry on", timestamp: "2026-08-21T09:18:32.595Z" },
    ]);
  });

  it("fails open (returns []) when the session directory cannot be read", () => {
    expect(readKimiTranscript(join(root, "nope"))).toEqual([]);
    // a session directory that exists but holds no agents yet
    mkdirSync(join(root, "agents"), { recursive: true });
    expect(readKimiTranscript(root)).toEqual([]);
  });
});

describe("kimiSessionDir", () => {
  const SESSION = "4af6ea82-a66b-416e-930c-a7476ba44596";

  it("finds the session under its hashed workspace directory, with or without the session_ prefix", () => {
    // The workspace directory carries a hash that cwd cannot reproduce — two repos on one machine
    // share the basename `example-project` — so the session is found by scanning, not composing.
    const dir = join(root, "sessions", "wd_example-project_4b1c9e07a2f3", `session_${SESSION}`);
    mkdirSync(join(root, "sessions", "wd_example-project_9f27d1b6c845"), { recursive: true });
    mkdirSync(join(dir, "agents", "main"), { recursive: true });

    expect(kimiSessionDir(SESSION, root)).toBe(dir);
    // session_index.jsonl records the prefixed form; hook payloads carry the bare uuid.
    expect(kimiSessionDir(`session_${SESSION}`, root)).toBe(dir);
  });

  it("returns undefined when no workspace holds the session", () => {
    mkdirSync(join(root, "sessions", "wd_amd-research_18567076f83d"), { recursive: true });
    expect(kimiSessionDir(SESSION, root)).toBeUndefined();
    // no sessions directory at all (a fresh install)
    expect(kimiSessionDir(SESSION, join(root, "nope"))).toBeUndefined();
  });
});
