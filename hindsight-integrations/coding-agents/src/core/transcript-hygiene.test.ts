import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { TransportTurn } from "./chat";
import { readAntigravityTranscript } from "./transcript-antigravity";
import { readCodexTranscript } from "./transcript-codex";
import { readClaudeTranscript } from "./transcript";
import { applyTranscriptHygiene } from "./transcript-hygiene";

const CODEX_RESPONSE_ITEM_TYPE = "response_item";
const CODEX_MESSAGE_PAYLOAD_TYPE = "message";
const CODEX_FUNCTION_CALL_PAYLOAD_TYPE = "function_call";
const EMPTY_ITEM_COUNT = 0;
const ONE_ITEM_COUNT = 1;
const TWO_ITEM_COUNT = 2;
const THREE_ITEM_COUNT = 3;
const FIRST_ITEM_INDEX = 0;
const SECOND_ITEM_INDEX = 1;
const TWO_ACTIONS_GROUP_HEADER = `Action breadcrumbs (${TWO_ITEM_COUNT} grouped):`;

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-transcript-hygiene-"));
  file = join(root, "session.jsonl");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function codexItem(payload: unknown): string {
  return JSON.stringify({ type: CODEX_RESPONSE_ITEM_TYPE, payload });
}

function applyBeta(turns: TransportTurn[]) {
  return applyTranscriptHygiene("semantic-beta", turns);
}

describe("applyTranscriptHygiene", () => {
  it("is off by default and returns the same turns without rewriting user text", () => {
    const turns = [{ role: "user", content: "keep my exact request" }];
    const result = applyTranscriptHygiene("off", turns);
    expect(result.turns).toBe(turns);
    expect(result.receipt).toMatchObject({
      applied: false,
      inputTurns: turns.length,
      outputTurns: turns.length,
      userTurns: turns.length,
      actionGroups: EMPTY_ITEM_COUNT,
    });
  });

  it("groups consecutive action breadcrumbs while preserving surrounding prose order", () => {
    const turns = [
      { role: "user", content: "ship it" },
      { role: "action", content: "Read src/a.ts" },
      { role: "action", content: "Edit src/a.ts" },
      { role: "assistant", content: "done" },
      { role: "action", content: "Bash npm test" },
    ];

    const result = applyBeta(turns);

    expect(result.turns).toEqual([
      { role: "user", content: "ship it" },
      {
        role: "action",
        content: `${TWO_ACTIONS_GROUP_HEADER}\n- Read src/a.ts\n- Edit src/a.ts`,
      },
      { role: "assistant", content: "done" },
      { role: "action", content: "Bash npm test" },
    ]);
    expect(result.receipt).toMatchObject({
      applied: true,
      inputTurns: turns.length,
      outputTurns: result.turns.length,
      actionTurns: THREE_ITEM_COUNT,
      actionGroups: ONE_ITEM_COUNT,
      groupedActionTurns: TWO_ITEM_COUNT,
    });
  });

  it("supports Codex normalized turns", () => {
    writeFileSync(
      file,
      [
        codexItem({
          type: CODEX_MESSAGE_PAYLOAD_TYPE,
          role: "user",
          content: [{ type: "input_text", text: "compare both filters" }],
        }),
        codexItem({
          type: CODEX_FUNCTION_CALL_PAYLOAD_TYPE,
          name: "exec_command",
          arguments: '{"command":"npm test"}',
        }),
        codexItem({
          type: CODEX_FUNCTION_CALL_PAYLOAD_TYPE,
          name: "exec_command",
          arguments: '{"command":"npm run build"}',
        }),
      ].join("\n")
    );

    const result = applyBeta(readCodexTranscript(file));

    expect(result.turns).toHaveLength(TWO_ITEM_COUNT);
    expect(result.turns[SECOND_ITEM_INDEX].content).toContain(TWO_ACTIONS_GROUP_HEADER);
  });

  it("supports Claude normalized turns", () => {
    writeFileSync(
      file,
      JSON.stringify({
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            { type: "tool_use", name: "Bash", input: { command: "npm test" } },
            { type: "tool_use", name: "Edit", input: { file_path: "src/core/chat.ts" } },
          ],
        },
      })
    );

    const result = applyBeta(readClaudeTranscript(file));

    expect(result.turns).toHaveLength(ONE_ITEM_COUNT);
    expect(result.turns[FIRST_ITEM_INDEX].content).toContain("Edit src/core/chat.ts");
  });

  it("supports Antigravity normalized turns without inventing action breadcrumbs", () => {
    writeFileSync(
      file,
      [
        JSON.stringify({
          type: "USER_INPUT",
          content: "Retain this session.",
          timestamp: "2026-01-01T00:00:00Z",
        }),
        JSON.stringify({ role: "assistant", content: "Stored." }),
      ].join("\n")
    );

    const result = applyBeta(readAntigravityTranscript(file));

    expect(result.turns).toEqual([
      { role: "user", content: "Retain this session.", timestamp: "2026-01-01T00:00:00Z" },
      { role: "assistant", content: "Stored." },
    ]);
    expect(result.receipt.actionGroups).toBe(EMPTY_ITEM_COUNT);
  });
});
