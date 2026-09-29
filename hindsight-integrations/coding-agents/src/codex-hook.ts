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
 * Behavior (shared hook runtime, core/hook.ts): recall every prompt; reflect once per session on
 * the first prompt and cache the outcome so later prompts recall only. Reflect outcomes recorded
 * in the diagnostic file. Config: the layered files, harness name "codex".
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("codex");
