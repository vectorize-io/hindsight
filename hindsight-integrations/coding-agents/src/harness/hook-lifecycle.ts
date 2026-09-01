/**
 * The single lifecycle contract for every hook-based harness. Entry-point binaries, installer
 * wiring, payload parsing, host response encoding, and transcript readers all resolve through
 * this registry. Adding a harness means adding one complete declaration here; its lifecycle
 * cannot silently drift between the runtime and installer.
 */
import { runHook, type HookSpec } from "../core/hook";
import { runRetainHook, type RetainHookSpec } from "../core/retain-hook";
import { runSessionStartHook, type SessionStartHookSpec } from "../core/session-start";
import { readCodexTranscript } from "../core/transcript-codex";
import { readCursorTranscript } from "../core/transcript-cursor";
import { readAntigravityTranscript } from "../core/transcript-antigravity";
import { readCopilotTranscript } from "../core/transcript-copilot";
import { grokTranscriptPath, readGrokTranscript } from "../core/transcript-grok";
import { readDevinTranscript } from "../core/transcript-devin";
import { dcodeAssistantText, readDcodeTranscript } from "../core/transcript-dcode";
import { readQwenTranscript } from "../core/transcript-qwen";

export type HookHarnessName =
  | "claude-code"
  | "codex"
  | "antigravity-cli"
  | "cursor-cli"
  | "copilot-cli"
  | "devin-cli"
  | "grok-build"
  | "dcode"
  | "qwen-code";
export type HookLifecycle = "sessionStart" | "prompt" | "stop";
export type HookConfigStyle = "nested" | "flat";

export interface HookInstallSpec {
  event: string;
  entry: string;
  /** In `HookHarnessSpec.timeoutUnit` — NOT always seconds. See that field. */
  timeout?: number;
}

/**
 * The unit the HOST reads `HookInstallSpec.timeout` in. Every host but Qwen Code uses seconds;
 * Qwen passes the value straight to setTimeout, so 30 there means 30ms and the hook is dead before
 * a Node process starts. Declaring the unit makes that difference checkable instead of a comment:
 * the lifecycle test normalises through it, so changing Qwen's 30_000 to 30 without also changing
 * this field now fails a test rather than silently shipping dead hooks.
 */
export type HookTimeoutUnit = "seconds" | "milliseconds";

export interface HookHarnessSpec {
  configStyle: HookConfigStyle;
  /** Defaults to "seconds" when absent — the unit every host but qwen-code uses. */
  timeoutUnit?: HookTimeoutUnit;
  install: Record<HookLifecycle, HookInstallSpec>;
  sessionStart: SessionStartHookSpec;
  prompt: HookSpec;
  retain: RetainHookSpec;
}

const cursorCwd = (ev: Record<string, unknown>): string | undefined =>
  (ev.cwd as string | undefined) ??
  (ev.workspace_root as string | undefined) ??
  (Array.isArray(ev.workspace_roots) ? (ev.workspace_roots[0] as string | undefined) : undefined);

const claudePrompt: HookSpec = {
  harness: "claude-code",
  parse: (ev) => ({
    prompt: ev.prompt as string | undefined,
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
  emit: (context, notice) => ({
    ...(notice ? { systemMessage: notice } : {}),
    hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context },
  }),
};

const codexPrompt: HookSpec = {
  ...claudePrompt,
  harness: "codex",
  parse: (ev) => ({
    prompt: (ev.prompt as string | undefined) ?? (ev.user_prompt as string | undefined),
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
};

const dcodePrompt: HookSpec = {
  ...claudePrompt,
  harness: "dcode",
};

/**
 * Qwen Code speaks Claude Code's hook protocol field for field (session_id / transcript_path / cwd
 * in, hookSpecificOutput.additionalContext + systemMessage out), so only `parse` differs — and it
 * reads `submitted_prompt`, NOT `prompt`.
 *
 * `UserPromptSubmit` fires on tool-result continuations too, not just on submissions: Qwen's send
 * loop labels the first turn `userQuery` and every continuation `toolResult`, and the hook's fire
 * guard excludes only retry/steer/cron/notification/teammate/goal. On a continuation `prompt` holds
 * whatever text is currently model-bound (a tool result), so keying on it would recall ~20 times per
 * user turn against tool output. `submitted_prompt` is attached only when the turn is BOTH the first
 * one and a real `userQuery`, which makes it the exact genuine-submission marker — and runHook's
 * `if (!prompt) return` then suppresses every continuation with no core change.
 *
 * Cost, deliberately accepted: `submitted_prompt` is the interactive TUI's text projection, so
 * headless (`qwen -p`), serve/SDK and ACP sessions carry none and never recall. They still get the
 * SessionStart seed and the Stop write-back.
 */
const qwenPrompt: HookSpec = {
  ...claudePrompt,
  harness: "qwen-code",
  parse: (ev) => ({
    prompt: ev.submitted_prompt as string | undefined,
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
};

const antigravityCwd = (ev: Record<string, unknown>): string | undefined =>
  Array.isArray(ev.workspacePaths) ? (ev.workspacePaths[0] as string | undefined) : undefined;

const antigravityPrompt: HookSpec = {
  harness: "antigravity-cli",
  requireCwd: true,
  parse: (ev) => ({
    // Antigravity's PreInvocation payload deliberately omits the prompt. Its transcript is already
    // persisted at that point, so recover the latest real user turn from the supplied JSONL path.
    prompt: readAntigravityTranscript(ev.transcriptPath as string | undefined)
      .filter((turn) => turn.role === "user")
      .at(-1)?.content,
    cwd: antigravityCwd(ev),
    sessionId: ev.conversationId as string | undefined,
  }),
  emit: (context) => ({ injectSteps: context ? [{ ephemeralMessage: context }] : [] }),
};

const cursorPrompt: HookSpec = {
  harness: "cursor-cli",
  parse: (ev) => ({
    prompt: (ev.prompt as string | undefined) ?? (ev.user_prompt as string | undefined),
    cwd: cursorCwd(ev),
    sessionId: (ev.conversation_id as string | undefined) ?? (ev.session_id as string | undefined),
  }),
  emit: (context) => ({ continue: true, additional_context: context }),
};

const copilotPrompt: HookSpec = {
  harness: "copilot-cli",
  parse: (ev) => ({
    prompt: ev.prompt as string | undefined,
    cwd: ev.cwd as string | undefined,
    sessionId: ev.sessionId as string | undefined,
  }),
  emit: (context, _notice, ev) => ({
    // Copilot's userPromptTransformed hook replaces model-facing content rather than appending
    // hook context. Preserve its transformed prompt and add the shared Hindsight injection.
    modifiedTransformedPrompt:
      `${(ev?.transformedPrompt as string | undefined) ?? ""}\n\n${context}`.trim(),
  }),
};

const devinCwd = (): string | undefined => process.env.DEVIN_PROJECT_DIR;
const devinPrompt: HookSpec = {
  harness: "devin-cli",
  requireCwd: true,
  parse: (ev) => ({
    prompt: ev.prompt as string | undefined,
    cwd: devinCwd(),
    sessionId: ev.session_id as string | undefined,
  }),
  emit: (context) => ({
    hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context },
  }),
};

const standardSessionStart = (harness: string): SessionStartHookSpec => ({
  harness,
  parse: (ev) => ({
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
  emit: (out) => ({
    ...(out.systemMessage ? { systemMessage: out.systemMessage } : {}),
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      ...(out.additionalContext ? { additionalContext: out.additionalContext } : {}),
    },
  }),
});

export const HOOK_HARNESSES: Record<HookHarnessName, HookHarnessSpec> = {
  "claude-code": {
    configStyle: "nested",
    install: {
      sessionStart: { event: "SessionStart", entry: "claude-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "claude-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "claude-stop-hook.js", timeout: 60 },
    },
    sessionStart: standardSessionStart("claude-code"),
    prompt: claudePrompt,
    retain: {
      hostTimeoutSec: 60,
      harness: "claude-code",
      parse: (ev) => ({
        sessionId: ev.session_id as string | undefined,
        transcriptPath: ev.transcript_path as string | undefined,
        cwd: ev.cwd as string | undefined,
      }),
    },
  },
  codex: {
    configStyle: "nested",
    install: {
      sessionStart: { event: "SessionStart", entry: "codex-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "codex-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "codex-stop-hook.js", timeout: 60 },
    },
    sessionStart: standardSessionStart("codex"),
    prompt: codexPrompt,
    retain: {
      hostTimeoutSec: 60,
      harness: "codex",
      parse: (ev) => ({
        sessionId: ev.session_id as string | undefined,
        transcriptPath: ev.transcript_path as string | undefined,
        cwd: ev.cwd as string | undefined,
      }),
      readTranscript: readCodexTranscript,
    },
  },
  dcode: {
    configStyle: "nested",
    install: {
      sessionStart: { event: "SessionStart", entry: "dcode-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "dcode-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "dcode-stop-hook.js", timeout: 60 },
    },
    sessionStart: standardSessionStart("dcode"),
    prompt: dcodePrompt,
    retain: {
      hostTimeoutSec: 60,
      harness: "dcode",
      parse: (ev) => ({
        sessionId: ev.session_id as string | undefined,
        transcriptPath: ev.transcript_path as string | undefined,
        cwd: ev.cwd as string | undefined,
        lastAssistantMessage: ev.last_assistant_message as string | undefined,
      }),
      readTranscript: readDcodeTranscript,
      readLastMessage: dcodeAssistantText,
    },
  },
  "antigravity-cli": {
    configStyle: "flat",
    install: {
      // PreInvocation is Antigravity's only lifecycle point that can inject context. Its first
      // invocation also performs the SessionStart responsibilities through runHook's seed guard.
      sessionStart: { event: "PreInvocation", entry: "antigravity-hook.js", timeout: 30 },
      prompt: { event: "PreInvocation", entry: "antigravity-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "antigravity-stop-hook.js", timeout: 30 },
    },
    sessionStart: {
      harness: "antigravity-cli",
      parse: (ev) => ({
        cwd: antigravityCwd(ev),
        sessionId: ev.conversationId as string | undefined,
      }),
      emit: () => ({}),
    },
    prompt: antigravityPrompt,
    retain: {
      hostTimeoutSec: 30,
      harness: "antigravity-cli",
      parse: (ev) => ({
        sessionId: ev.conversationId as string | undefined,
        transcriptPath: ev.transcriptPath as string | undefined,
        cwd: antigravityCwd(ev),
      }),
      readTranscript: readAntigravityTranscript,
    },
  },
  "cursor-cli": {
    configStyle: "flat",
    install: {
      sessionStart: { event: "sessionStart", entry: "cursor-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "beforeSubmitPrompt", entry: "cursor-hook.js" },
      stop: { event: "stop", entry: "cursor-stop-hook.js", timeout: 30 },
    },
    sessionStart: {
      harness: "cursor-cli",
      parse: (ev) => ({
        cwd: cursorCwd(ev),
        sessionId:
          (ev.conversation_id as string | undefined) ?? (ev.session_id as string | undefined),
      }),
      emit: (out) => ({
        ...(out.additionalContext ? { additional_context: out.additionalContext } : {}),
      }),
    },
    prompt: cursorPrompt,
    retain: {
      hostTimeoutSec: 30,
      harness: "cursor-cli",
      parse: (ev) => ({
        sessionId:
          (ev.conversation_id as string | undefined) ?? (ev.session_id as string | undefined),
        transcriptPath: ev.transcript_path as string | undefined,
        cwd: cursorCwd(ev),
      }),
      readTranscript: readCursorTranscript,
    },
  },
  "copilot-cli": {
    configStyle: "flat",
    install: {
      sessionStart: { event: "sessionStart", entry: "copilot-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "userPromptTransformed", entry: "copilot-hook.js", timeout: 30 },
      stop: { event: "agentStop", entry: "copilot-stop-hook.js", timeout: 60 },
    },
    sessionStart: {
      harness: "copilot-cli",
      parse: (ev) => ({
        cwd: ev.cwd as string | undefined,
        sessionId: ev.sessionId as string | undefined,
      }),
      // Copilot CLI's SessionStart response only has model-facing `additionalContext`; unlike
      // Claude/Cursor it has no supported in-TUI system-message/banner channel. Keep memory
      // quiet rather than auto-submitting a synthetic prompt or showing an OS notification. When
      // Copilot exposes a real TUI extension point, add the banner there without changing this
      // shared lifecycle output.
      emit: (out) => ({
        ...(out.additionalContext ? { additionalContext: out.additionalContext } : {}),
      }),
    },
    prompt: copilotPrompt,
    retain: {
      hostTimeoutSec: 60,
      harness: "copilot-cli",
      parse: (ev) => ({
        sessionId: ev.sessionId as string | undefined,
        transcriptPath: ev.transcriptPath as string | undefined,
        cwd: ev.cwd as string | undefined,
      }),
      readTranscript: readCopilotTranscript,
    },
  },
  "devin-cli": {
    configStyle: "nested",
    install: {
      sessionStart: { event: "SessionStart", entry: "devin-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "devin-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "devin-stop-hook.js", timeout: 60 },
    },
    sessionStart: {
      harness: "devin-cli",
      parse: (ev) => ({ cwd: devinCwd(), sessionId: ev.session_id as string | undefined }),
      emit: (out) => ({
        hookSpecificOutput: {
          hookEventName: "SessionStart",
          ...(out.additionalContext ? { additionalContext: out.additionalContext } : {}),
        },
      }),
    },
    prompt: devinPrompt,
    retain: {
      hostTimeoutSec: 60,
      harness: "devin-cli",
      parse: (ev) => {
        const sessionId = ev.session_id as string | undefined;
        return {
          sessionId,
          // RetainHook calls the supplied reader with transcriptPath; Devin's reader uses its
          // session id because the CLI persists conversations in sessions.db rather than a file.
          transcriptPath: sessionId,
          cwd: devinCwd(),
        };
      },
      readTranscript: readDevinTranscript,
    },
  },
  "grok-build": {
    configStyle: "nested",
    install: {
      sessionStart: { event: "SessionStart", entry: "grok-sessionstart-hook.js", timeout: 30 },
      prompt: { event: "UserPromptSubmit", entry: "grok-hook.js", timeout: 30 },
      stop: { event: "Stop", entry: "grok-stop-hook.js", timeout: 60 },
    },
    sessionStart: {
      ...standardSessionStart("grok-build"),
      // Grok's wire envelope is camelCase, unlike Claude's similarly named hook events.
      parse: (ev) => ({
        cwd: ev.cwd as string | undefined,
        sessionId: ev.sessionId as string | undefined,
      }),
    },
    prompt: {
      ...claudePrompt,
      harness: "grok-build",
      parse: (ev) => ({
        prompt: ev.prompt as string | undefined,
        cwd: ev.cwd as string | undefined,
        sessionId: ev.sessionId as string | undefined,
      }),
    },
    retain: {
      hostTimeoutSec: 60,
      harness: "grok-build",
      parse: (ev) => ({
        sessionId: ev.sessionId as string | undefined,
        transcriptPath:
          typeof ev.cwd === "string" && typeof ev.sessionId === "string"
            ? grokTranscriptPath(ev.cwd, ev.sessionId)
            : undefined,
        cwd: ev.cwd as string | undefined,
      }),
      readTranscript: readGrokTranscript,
    },
  },
  "qwen-code": {
    configStyle: "nested",
    timeoutUnit: "milliseconds",
    // TIMEOUTS ARE MILLISECONDS HERE, not seconds like every other harness in this table: Qwen's
    // hookRunner does `setTimeout(..., hookConfig.timeout ?? DEFAULT_HOOK_TIMEOUT)` with
    // DEFAULT_HOOK_TIMEOUT = 60_000 ("Timeout in milliseconds, default 60000"). Writing 30/60
    // registers 30ms/60ms hooks, which die before a Node process starts — and Qwen spawns hooks
    // `detached` and terminates the whole process TREE on timeout, so the retain is genuinely lost
    // rather than merely orphaned. `retain.hostTimeoutSec` below stays SECONDS, as its name says;
    // these two numbers are the same budget in different units for this harness alone.
    // The prompt timeout must also stay above core/hook.ts's HOOK_REFLECT_CAP_MS (25_000).
    install: {
      sessionStart: { event: "SessionStart", entry: "qwen-sessionstart-hook.js", timeout: 30_000 },
      prompt: { event: "UserPromptSubmit", entry: "qwen-hook.js", timeout: 30_000 },
      stop: { event: "Stop", entry: "qwen-stop-hook.js", timeout: 60_000 },
    },
    sessionStart: standardSessionStart("qwen-code"),
    prompt: qwenPrompt,
    retain: {
      hostTimeoutSec: 60,
      harness: "qwen-code",
      parse: (ev) => ({
        sessionId: ev.session_id as string | undefined,
        // Qwen supplies the path, but as the EMPTY STRING (not null, not absent) when chat
        // recording is off — runRetainHook's `if (!transcriptPath) return` already covers that.
        transcriptPath: ev.transcript_path as string | undefined,
        cwd: ev.cwd as string | undefined,
      }),
      readTranscript: readQwenTranscript,
    },
  },
};

export const runHarnessSessionStart = (harness: HookHarnessName): Promise<void> =>
  runSessionStartHook(HOOK_HARNESSES[harness].sessionStart);
export const runHarnessPrompt = (harness: HookHarnessName): Promise<void> =>
  runHook(HOOK_HARNESSES[harness].prompt);
export const runHarnessRetain = (harness: HookHarnessName): Promise<void> =>
  runRetainHook(HOOK_HARNESSES[harness].retain);
