#!/usr/bin/env node
/**
 * hindsight-workbuddy-hook - the WorkBuddy entry point (a `UserPromptSubmit` hook).
 *
 * Install (WorkBuddy, user scope): ~/.workbuddy/settings.json
 *   { "hooks": { "UserPromptSubmit": [ { "hooks": [
 *       { "type": "command", "command": "node \"…/workbuddy-hook.js\"", "timeout": 30 } ] } ] } }
 *
 * WorkBuddy (Tencent's AI workbench, built on the @genie/agent-cli engine) speaks Claude Code's
 * hook protocol field for field, so this shares the shared hook runtime unchanged — recall every
 * prompt; reflect once per session on the first prompt (see core/hook.ts).
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("workbuddy");
