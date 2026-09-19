#!/usr/bin/env node
/**
 * hindsight-cursor-hook — the Cursor CLI entry point (a `beforeSubmitPrompt` hook).
 *
 * Install (Cursor hooks.json):
 *   { "version": 1, "hooks": { "beforeSubmitPrompt": [ { "command": "hindsight-cursor-hook" } ] } }
 *
 * Cursor's hook contract (see the hindsight-cursor-cli integration): event on stdin carries the
 * prompt (`prompt` or `user_prompt`) and a `conversation_id`; output is
 *   { "continue": true, "additional_context": "..." }
 * — `continue` is always true: a memory failure must never block the user's prompt.
 *
 * Behavior (shared hook runtime, core/hook.ts): recall every prompt; reflect once per
 * conversation on the first prompt and cache the outcome so later prompts recall only. Reflect
 * outcomes recorded in the diagnostic file. Config: the layered files, harness name "cursor-cli".
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("cursor-cli");
