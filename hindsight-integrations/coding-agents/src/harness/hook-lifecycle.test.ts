import { describe, expect, it } from "vitest";
import { HOOK_HARNESSES, type HookHarnessName } from "./hook-lifecycle";

const HOOK_HARNESS_NAMES: HookHarnessName[] = [
  "claude-code",
  "codex",
  "antigravity-cli",
  "cursor-cli",
  "copilot-cli",
  "grok-build",
  "dcode",
];

describe("HOOK_HARNESSES lifecycle contract", () => {
  it("declares every lifecycle once for every hook-based harness", () => {
    for (const harness of HOOK_HARNESS_NAMES) {
      expect(Object.keys(HOOK_HARNESSES[harness].install).sort()).toEqual([
        "prompt",
        "sessionStart",
        "stop",
      ]);
      expect(HOOK_HARNESSES[harness].sessionStart.harness).toBe(harness);
      expect(HOOK_HARNESSES[harness].prompt.harness).toBe(harness);
      expect(HOOK_HARNESSES[harness].retain.harness).toBe(harness);
    }
  });

  it("keeps the runtime schema and installed event names in the same host declaration", () => {
    const cursor = HOOK_HARNESSES["cursor-cli"];
    expect(cursor.install).toMatchObject({
      sessionStart: { event: "sessionStart", entry: "cursor-sessionstart-hook.js" },
      prompt: { event: "beforeSubmitPrompt", entry: "cursor-hook.js" },
      stop: { event: "stop", entry: "cursor-stop-hook.js" },
    });
    expect(
      cursor.sessionStart.emit({ systemMessage: "visible", additionalContext: "context" })
    ).toEqual({
      additional_context: "context",
    });
    expect(cursor.prompt.emit("context", "visible")).toEqual({
      continue: true,
      additional_context: "context",
    });

    const claude = HOOK_HARNESSES["claude-code"];
    expect(claude.install.prompt.event).toBe("UserPromptSubmit");
    expect(claude.sessionStart.emit({ additionalContext: "context" })).toEqual({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: "context",
      },
    });

    const antigravity = HOOK_HARNESSES["antigravity-cli"];
    expect(antigravity.prompt.requireCwd).toBe(true);
    expect(antigravity.install).toMatchObject({
      sessionStart: { event: "PreInvocation", entry: "antigravity-hook.js", timeout: 30 },
      prompt: { event: "PreInvocation", entry: "antigravity-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "antigravity-stop-hook.js", timeout: 30 },
    });
    expect(antigravity.prompt.emit("context")).toEqual({
      injectSteps: [{ ephemeralMessage: "context" }],
    });

    const copilot = HOOK_HARNESSES["copilot-cli"];
    expect(copilot.install).toMatchObject({
      sessionStart: { event: "sessionStart", entry: "copilot-sessionstart-hook.js" },
      prompt: { event: "userPromptTransformed", entry: "copilot-hook.js" },
      stop: { event: "agentStop", entry: "copilot-stop-hook.js" },
    });
    expect(copilot.prompt.emit("context", undefined, { transformedPrompt: "original" })).toEqual({
      modifiedTransformedPrompt: "original\n\ncontext",
    });

    const grok = HOOK_HARNESSES["grok-build"];
    expect(grok.install).toMatchObject({
      sessionStart: { event: "SessionStart", entry: "grok-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "grok-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "grok-stop-hook.js", timeout: 60 },
    });
    expect(grok.prompt.emit("context")).toEqual({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: "context",
      },
    });

    const dcode = HOOK_HARNESSES.dcode;
    expect(dcode.install).toMatchObject({
      sessionStart: { event: "SessionStart", entry: "dcode-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "dcode-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "dcode-stop-hook.js", timeout: 60 },
    });
    expect(dcode.prompt.parse({ prompt: "hello", cwd: "/repo", session_id: "s1" })).toEqual({
      prompt: "hello",
      cwd: "/repo",
      sessionId: "s1",
    });
    expect(dcode.prompt.emit("context")).toEqual({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext: "context",
      },
    });
    expect(
      dcode.retain.parse({
        session_id: "s1",
        transcript_path: "/tmp/s1.jsonl",
        cwd: "/repo",
        last_assistant_message: "done",
      })
    ).toEqual({
      sessionId: "s1",
      transcriptPath: "/tmp/s1.jsonl",
      cwd: "/repo",
      lastAssistantMessage: "done",
    });
  });
});
