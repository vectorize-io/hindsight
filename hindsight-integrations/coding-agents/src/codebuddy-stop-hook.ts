#!/usr/bin/env node
/** CodeBuddy Stop hook: reads the session transcript (core/transcript-codebuddy-ide.ts — the IDE's
 *  index.json directory, or the CLI's JSONL that it hands to the WorkBuddy reader) and writes the
 *  whole conversation back to memory. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("codebuddy");
