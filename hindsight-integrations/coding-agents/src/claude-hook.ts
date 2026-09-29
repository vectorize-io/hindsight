#!/usr/bin/env node
/**
 * hindsight-claude-hook — the Claude Code entry point (a `UserPromptSubmit` hook).
 *
 * Install (Claude Code settings.json):
 *   { "hooks": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "hindsight-claude-hook" } ] } ] } }
 *
 * Behavior (shared hook runtime, core/hook.ts): recall every prompt; reflect once per Claude
 * session on the first prompt and cache the outcome so later prompts recall only. Reflect
 * outcomes recorded in the diagnostic file. Config: the layered files (~/.hindsight/coding-agent.json
 * + nearest project file), harness name "claude-code".
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("claude-code");
