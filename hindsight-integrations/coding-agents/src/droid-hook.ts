#!/usr/bin/env node
/**
 * hindsight-droid-hook - the Factory Droid entry point (a `UserPromptSubmit` hook).
 *
 * Install (Droid, user scope): ~/.factory/hooks.json
 *   { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "hindsight-droid-hook", "timeout": 30 } ] } ] }
 *
 * Behavior (shared hook runtime, core/hook.ts): recall every prompt; reflect once per session on
 * the first prompt and cache the outcome so later prompts recall only. Droid's hook protocol
 * matches Claude Code's on the wire (session_id / transcript_path / cwd in,
 * hookSpecificOutput.additionalContext out), so this is a one-line delegation.
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("factory-droid");
