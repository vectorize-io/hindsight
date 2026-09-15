#!/usr/bin/env node
/**
 * hindsight-codebuddy-hook - the CodeBuddy Code entry point (a `UserPromptSubmit` hook).
 *
 * Install (CodeBuddy, user scope): ~/.codebuddy/settings.json
 *   { "hooks": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "node \"…/codebuddy-hook.js\"", "timeout": 30 } ] } ] } }
 *
 * CodeBuddy Code runs the SAME `@genie/agent-cli` engine as WorkBuddy — WorkBuddy ships that engine
 * with `dataFolderName: ".workbuddy"` (see its product.json), CodeBuddy uses its default
 * `~/.codebuddy` — so it speaks Claude Code's hook protocol field for field. This therefore shares
 * the shared hook runtime unchanged: recall every prompt, reflect once per session on the first one
 * (see core/hook.ts).
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("codebuddy");
