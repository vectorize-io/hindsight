#!/usr/bin/env node
/**
 * hindsight-traecode-hook - the TraeCode entry point (a `UserPromptSubmit` hook).
 *
 * Install (TraeCode, user scope): ~/.trae-cn/hooks.json
 *   { "version": 1, "hooks": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "node \"…/traecode-hook.js\"", "timeout": 30 } ] } ] } }
 *
 * Behavior (shared hook runtime, core/hook.ts): recall every prompt; reflect once per session on
 * the first prompt and cache the outcome so later prompts recall only. TraeCode speaks Claude
 * Code's hook protocol, so the wire format is Claude's — the one addition is that this hook also
 * appends the prompt to the session's turn journal, because TraeCode supplies no transcript at all
 * (sessions live in an encrypted local DB / the cloud; see core/turn-journal.ts).
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("traecode");
