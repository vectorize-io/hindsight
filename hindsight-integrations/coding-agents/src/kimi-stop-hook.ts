#!/usr/bin/env node
/**
 * Hindsight Kimi Code `Stop` hook.
 *
 * Install (~/.kimi-code/config.toml) — the entry schema is STRICT, exactly these four keys:
 *   [[hooks]]
 *   event = "Stop"
 *   command = "hindsight-kimi-stop-hook"
 *   timeout = 30
 * A fifth key drops EVERY hook in the file at warning severity while the CLI still boots.
 */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("kimi-code");
