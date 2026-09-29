#!/usr/bin/env node
/** Devin CLI UserPromptSubmit hook: record its documented prompt payload, then use the shared recall path. */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("devin-cli");
