#!/usr/bin/env node
/**
 * Qwen Code `SessionStart` hook. Runs the shared session-start lifecycle: configures the bank's
 * knowledge pages, starts cold-repo seeding/deepening, and supplies the page roster. Qwen consumes
 * the Claude-compatible `hookSpecificOutput.additionalContext` + `systemMessage` envelope.
 */
import { runHarnessSessionStart } from "./harness/hook-lifecycle";

void runHarnessSessionStart("qwen-code");
