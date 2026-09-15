#!/usr/bin/env node
/** CodeBuddy Stop hook: reads the session transcript (the shared `@genie/agent-cli` reader,
 *  core/transcript-workbuddy.ts — CodeBuddy writes the same schema at ~/.codebuddy/projects) and
 *  writes the whole conversation back to memory. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("codebuddy");
