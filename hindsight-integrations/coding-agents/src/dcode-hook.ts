#!/usr/bin/env node
/** Native DeepAgents Dcode Hooks V2 UserPromptSubmit entrypoint. */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("dcode");
