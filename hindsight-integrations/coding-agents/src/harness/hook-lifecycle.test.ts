import { describe, expect, it } from "vitest";
import { HOOK_HARNESSES, type HookHarnessName } from "./hook-lifecycle";

// Derived, not hand-listed: a hand-written roster silently stops covering the newest harness, which
// is exactly the sibling it most needs to cover. (It had already fallen behind — devin-cli was
// missing.)
const HOOK_HARNESS_NAMES = Object.keys(HOOK_HARNESSES) as HookHarnessName[];

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

  /**
   * Family guard for the Stop-event reply recovery. A harness that reads `last_assistant_message`
   * off its Stop event MUST also say how to decode it, because the field is not always prose:
   * Dcode sends `str(content)`, a serialized content-block list. Retaining that undecoded stores
   * the provider's reasoning payload as if the assistant had said it, AND makes the
   * already-flushed compare fail so a duplicate turn is appended every turn.
   *
   * Asserted over the whole family rather than for Dcode alone: the next harness to surface this
   * field is by definition the one with no test of its own.
   */
  it("makes every harness that reads a Stop-event reply declare how to decode it", () => {
    const probe = {
      session_id: "s",
      transcript_path: "/t",
      cwd: "/c",
      last_assistant_message: "[{'type': 'text', 'text': 'hi'}]",
    };
    for (const harness of HOOK_HARNESS_NAMES) {
      const spec = HOOK_HARNESSES[harness].retain;
      if (spec.parse(probe).lastAssistantMessage === undefined) continue;
      expect(
        spec.readLastMessage,
        `${harness} reads last_assistant_message but never decodes it`
      ).toBeDefined();
      // And the decoder must actually reduce a block list to its text, not pass the repr through.
      expect(spec.readLastMessage!(probe.last_assistant_message)).toBe("hi");
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
