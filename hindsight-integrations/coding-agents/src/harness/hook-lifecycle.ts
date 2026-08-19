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

export type HookHarnessName =
  | "claude-code"
  | "codex"
  | "antigravity-cli"
  | "cursor-cli"
  | "copilot-cli"
  | "devin-cli"
  | "grok-build";
export type HookLifecycle = "sessionStart" | "prompt" | "stop";
export type HookConfigStyle = "nested" | "flat";

export interface HookInstallSpec {
  event: string;
  entry: string;
  timeout?: number;
}

export interface HookHarnessSpec {
  configStyle: HookConfigStyle;
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
};

export const runHarnessSessionStart = (harness: HookHarnessName): Promise<void> =>
  runSessionStartHook(HOOK_HARNESSES[harness].sessionStart);
export const runHarnessPrompt = (harness: HookHarnessName): Promise<void> =>
  runHook(HOOK_HARNESSES[harness].prompt);
export const runHarnessRetain = (harness: HookHarnessName): Promise<void> =>
  runRetainHook(HOOK_HARNESSES[harness].retain);
