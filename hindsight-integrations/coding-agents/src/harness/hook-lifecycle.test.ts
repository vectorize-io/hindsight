import { describe, expect, it } from "vitest";
import { HOOK_HARNESSES, type HookHarnessName } from "./hook-lifecycle";

const HOOK_HARNESS_NAMES: HookHarnessName[] = [
  "claude-code",
  "codex",
  "antigravity-cli",
  "cursor-cli",
  "copilot-cli",
  "grok-build",
  "qwen-code",
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
  });

  // The prompt hook must outlive the once-per-session reflect, or the FIRST prompt of every
  // session is killed mid-flight and recall silently degrades to nothing. Nothing coupled these
  // two numbers before: qwen-code's timeouts are MILLISECONDS while every other harness's are
  // SECONDS, so a bare `>= 25_000` would pass vacuously for the seven seconds-based harnesses and
  // a bare `>= 25` would pass vacuously for qwen. Normalising through the declared unit is what
  // makes this catch a mutation in EITHER direction.
  it("gives every prompt hook a budget above the once-per-session reflect cap", () => {
    const HOOK_REFLECT_CAP_MS = 25_000;
    for (const harness of HOOK_HARNESS_NAMES) {
      const spec = HOOK_HARNESSES[harness];
      const raw = spec.install.prompt.timeout;
      if (raw === undefined) continue; // cursor-cli deliberately omits it — the host default applies
      const ms = spec.timeoutUnit === "milliseconds" ? raw : raw * 1000;
      expect(
        ms,
        `${harness} prompt timeout (${raw} ${spec.timeoutUnit ?? "seconds"})`
      ).toBeGreaterThan(HOOK_REFLECT_CAP_MS);
    }
  });

  // hostTimeoutSec is SECONDS for every harness, including qwen-code where the installed values
  // are milliseconds. They describe the same budget, so they must agree once normalised.
  it("keeps the installed stop timeout consistent with hostTimeoutSec", () => {
    for (const harness of HOOK_HARNESS_NAMES) {
      const spec = HOOK_HARNESSES[harness];
      const raw = spec.install.stop.timeout;
      if (raw === undefined) continue;
      const ms = spec.timeoutUnit === "milliseconds" ? raw : raw * 1000;
      expect(ms, `${harness} stop timeout vs hostTimeoutSec`).toBe(
        spec.retain.hostTimeoutSec * 1000
      );
    }
  });
});
