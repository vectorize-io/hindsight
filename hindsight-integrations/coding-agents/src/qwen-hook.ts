#!/usr/bin/env node
/**
 * hindsight-qwen-hook — the Qwen Code entry point (a `UserPromptSubmit` hook).
 *
 * Install (~/.qwen/settings.json):
 *   { "hooks": { "UserPromptSubmit": [ { "hooks": [ { "type": "command",
 *     "command": "hindsight-qwen-hook", "timeout": 30000 } ] } ] } }
 *
 * Qwen speaks Claude Code's hook protocol field for field, with two differences that matter here:
 * `timeout` is MILLISECONDS (see HOOK_HARNESSES), and the genuine-submission marker is
 * `submitted_prompt` rather than `prompt` — `UserPromptSubmit` also fires on tool-result
 * continuations, where `prompt` holds model-bound tool output.
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("qwen-code");
