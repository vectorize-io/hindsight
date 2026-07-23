#!/usr/bin/env node
/**
 * hindsight-claude-hook — the Claude Code entry point (a `UserPromptSubmit` hook).
 *
 * Install (Claude Code settings.json):
 *   { "hooks": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "hindsight-claude-hook" } ] } ] } }
 *
 * Behavior (shared hook runtime, core/hook.ts): reflect once per Claude session on the first
 * prompt, cache, re-inject on later prompts; reflect outcomes recorded in the diagnostic file.
 * Config: the layered files (~/.hindsight/coding-agent.json + nearest project file), harness
 * name "claude-code".
 */
import { runHook } from "./core/hook";

void runHook({
  harness: "claude-code",
  parse: (ev) => ({
    prompt: ev.prompt as string | undefined,
    cwd: ev.cwd as string | undefined,
    sessionId: ev.session_id as string | undefined,
  }),
  emit: (context) => ({
    hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context },
  }),
});
