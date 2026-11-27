import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { HOOK_HARNESSES } from "../harness/hook-lifecycle";
import { readAntigravityPromptTurn, readAntigravityTranscript } from "./transcript-antigravity";

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-antigravity-transcript-"));
  file = join(root, "transcript.jsonl");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

describe("readAntigravityTranscript", () => {
  it("reads supported user and assistant message shapes while ignoring metadata", () => {
    writeFileSync(
      file,
      [
        JSON.stringify({ type: "system", content: "metadata" }),
        JSON.stringify({
          role: "user",
          content: "Fix the retry bug",
          timestamp: "2026-01-01T00:00:00Z",
        }),
        JSON.stringify({ message: { role: "assistant", text: "I will inspect it." } }),
      ].join("\n")
    );
    expect(readAntigravityTranscript(file)).toEqual([
      { role: "user", content: "Fix the retry bug", timestamp: "2026-01-01T00:00:00Z" },
      { role: "assistant", content: "I will inspect it." },
    ]);
  });

  it("reads Antigravity CLI's native USER_INPUT records without agy's envelope (#5791)", () => {
    writeFileSync(
      file,
      [
        JSON.stringify({
          type: "USER_INPUT",
          content:
            "<USER_REQUEST>\nCreate a bank for this repo.\n</USER_REQUEST>\n" +
            "<ADDITIONAL_METADATA>\nThe current local time is: 2026-10-23T18:00:00\n</ADDITIONAL_METADATA>",
        }),
        JSON.stringify({
          type: "USER_INPUT",
          content:
            "<ADDITIONAL_METADATA>\nlocal time\n</ADDITIONAL_METADATA>\nNo request tag here.",
        }),
      ].join("\n")
    );
    expect(readAntigravityTranscript(file)).toEqual([
      { role: "user", content: "Create a bank for this repo." },
      { role: "user", content: "No request tag here." },
    ]);
  });

  it("strips injected memory and fails open for an unavailable transcript", () => {
    writeFileSync(
      file,
      JSON.stringify({
        role: "user",
        text: "<hindsight_memory>old</hindsight_memory>\nWhat changed?",
      })
    );
    expect(readAntigravityTranscript(file)).toEqual([{ role: "user", content: "What changed?" }]);
    expect(readAntigravityTranscript(join(root, "missing.jsonl"))).toEqual([]);
  });
});

const rec = (type: string, content = "") => JSON.stringify({ type, content });
const ask = (text: string) => rec("USER_INPUT", `<USER_REQUEST>\n${text}\n</USER_REQUEST>`);

describe("readAntigravityPromptTurn", () => {
  it("returns the newest unwrapped user message and its line, even with tool records after it", () => {
    writeFileSync(
      file,
      [
        ask("Audit the retry logic."),
        rec("CHECKPOINT"),
        rec("PLANNER_RESPONSE"),
        rec("GENERIC", "x"),
      ].join("\n")
    );
    expect(readAntigravityPromptTurn(file)).toEqual({ prompt: "Audit the retry logic.", line: 0 });

    writeFileSync(
      file,
      [ask("first"), rec("GENERIC", "x"), ask("second"), rec("GENERIC", "y")].join("\n")
    );
    expect(readAntigravityPromptTurn(file)).toEqual({ prompt: "second", line: 2 });
  });

  it("returns undefined without a user message", () => {
    writeFileSync(file, rec("SYSTEM_MESSAGE", "Task finished"));
    expect(readAntigravityPromptTurn(file)).toBeUndefined();
    expect(readAntigravityPromptTurn(undefined)).toBeUndefined();
  });
});

describe("antigravity PreInvocation prompt (#6944)", () => {
  // The record order agy 1.5.18 writes, read at each PreInvocation: the mid-turn follow-up lands
  // AFTER the hook that follows it, and tool records already sit behind it at the next one.
  it("recalls the opening prompt once and a mid-turn follow-up once, never a tool step", () => {
    const parse = HOOK_HARNESSES["antigravity-cli"].prompt.parse;
    const ev = {
      transcriptPath: file,
      conversationId: `test-6944-${process.pid}-${Date.now()}`,
      workspacePaths: [root],
    };
    const steps: string[][] = [
      [ask("Run sleep 40")],
      [
        ask("Run sleep 40"),
        rec("SYSTEM_MESSAGE", "s"),
        rec("PLANNER_RESPONSE"),
        rec("GENERIC", "running"),
      ],
      [
        ask("Run sleep 40"),
        rec("SYSTEM_MESSAGE", "s"),
        rec("PLANNER_RESPONSE"),
        rec("GENERIC", "running"),
        ask("Also: what is the team mascot?"),
        rec("PLANNER_RESPONSE"),
        rec("GENERIC", "done"),
      ],
      [
        ask("Run sleep 40"),
        rec("SYSTEM_MESSAGE", "s"),
        rec("PLANNER_RESPONSE"),
        rec("GENERIC", "running"),
        ask("Also: what is the team mascot?"),
        rec("PLANNER_RESPONSE"),
        rec("GENERIC", "done"),
        rec("PLANNER_RESPONSE"),
        rec("GENERIC", "more"),
      ],
    ];
    const prompts = steps.map((lines) => {
      writeFileSync(file, lines.join("\n"));
      return parse(ev).prompt;
    });
    rmSync(join(tmpdir(), "hindsight-antigravity-cli", `${ev.conversationId}.recalled`), {
      force: true,
    });
    expect(prompts).toEqual([
      "Run sleep 40",
      undefined,
      "Also: what is the team mascot?",
      undefined,
    ]);
  });
});
