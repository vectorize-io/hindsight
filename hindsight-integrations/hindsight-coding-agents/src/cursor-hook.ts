#!/usr/bin/env node
/**
 * hindsight-cursor-hook — the Cursor CLI entry point (a `beforeSubmitPrompt` hook).
 *
 * Install (Cursor hooks.json):
 *   { "hooks": { "beforeSubmitPrompt": [ { "command": "hindsight-cursor-hook" } ] } }
 *
 * Cursor's hook contract (see the hindsight-cursor-cli integration): event on stdin carries the
 * prompt (`prompt` or `user_prompt`) and a `conversation_id`; output is
 *   { "continue": true, "additional_context": "..." }
 * — `continue` is always true: a memory failure must never block the user's prompt.
 *
 * Behavior (shared hook runtime, core/hook.ts): reflect once per conversation, cache, re-inject;
 * reflect outcomes recorded in the diagnostic file. Config: the layered files, harness name
 * "cursor-cli".
 */
import { runHook } from "./core/hook";

void runHook({
  harness: "cursor-cli",
  parse: (ev) => ({
    prompt: (ev.prompt as string | undefined) ?? (ev.user_prompt as string | undefined),
    cwd:
      (ev.cwd as string | undefined) ??
      (ev.workspace_root as string | undefined) ??
      (Array.isArray(ev.workspace_roots) ? (ev.workspace_roots[0] as string) : undefined),
    sessionId: (ev.conversation_id as string | undefined) ?? (ev.session_id as string | undefined),
  }),
  emit: (context) => ({ continue: true, additional_context: context }),
});
