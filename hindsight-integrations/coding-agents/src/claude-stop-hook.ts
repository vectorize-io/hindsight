#!/usr/bin/env node
/** hindsight-claude-stop-hook — Claude Code `Stop` hook: writes the session transcript back to memory. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("claude-code");
