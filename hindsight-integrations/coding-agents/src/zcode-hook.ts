#!/usr/bin/env node
/**
 * hindsight-zcode-hook - the ZCode entry point (a `UserPromptSubmit` hook).
 *
 * Install (ZCode, user scope): ~/.zcode/cli/config.json
 *   { "hooks": { "enabled": true, "events": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "process", "command": "node", "args": ["…/zcode-hook.js"],
 *         "timeoutMs": 30000 } ] } ] } } }
 *
 * Behavior (shared hook runtime, core/hook.ts): recall every prompt; reflect once per session on
 * the first prompt and cache the outcome so later prompts recall only. ZCode embeds the Claude Code
 * agent runtime, so the wire protocol is Claude's — the one addition is that this hook also appends
 * the prompt to the session's turn journal, because ZCode's `Stop` payload carries no user text and
 * its transcript does not survive the hook (see core/turn-journal.ts).
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("zcode");
