#!/usr/bin/env node
/**
 * Hindsight Kimi Code `UserPromptSubmit` hook.
 *
 * Install (~/.kimi-code/config.toml) — the entry schema is STRICT, exactly these four keys:
 *   [[hooks]]
 *   event = "UserPromptSubmit"
 *   command = "hindsight-kimi-hook"
 *   timeout = 30
 * A fifth key drops EVERY hook in the file at warning severity while the CLI still boots.
 */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("kimi-code");
