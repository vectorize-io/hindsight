#!/usr/bin/env node
/**
 * hindsight-codex-hook — the OpenAI Codex CLI entry point (a `UserPromptSubmit` hook).
 *
 * Codex CLI (v0.116+, `codex_hooks = true`) speaks a Claude-Code-compatible hook protocol
 * (see the hindsight-codex integration): event on stdin with session_id and prompt (or
 * user_prompt), output via hookSpecificOutput.additionalContext.
 *
 * Install (~/.codex/hooks.json):
 *   { "hooks": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "hindsight-codex-hook" } ] } ] } }
 *
 * Behavior (shared hook runtime, core/hook.ts): reflect once per session, cache, re-inject;
 * reflect outcomes recorded in the diagnostic file. Config: the layered files, harness name
 * "codex".
 */
import { runHook } from "./core/hook";

void runHook({
  harness: "codex",
  parse: (ev) => ({
    prompt: (ev.prompt as string | undefined) ?? (ev.user_prompt as string | undefined),
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
  emit: (context) => ({
    hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context },
  }),
});
