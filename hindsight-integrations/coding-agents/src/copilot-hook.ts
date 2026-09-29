#!/usr/bin/env node
/** GitHub Copilot CLI userPromptTransformed hook — shared Hindsight prompt lifecycle. */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("copilot-cli");
