#!/usr/bin/env node
/** Antigravity CLI PreInvocation hook. It obtains the current user task from the documented
 * transcriptPath payload and injects Hindsight context as an ephemeral model message. */
import { runHarnessPrompt } from "./harness/hook-lifecycle";

void runHarnessPrompt("antigravity-cli");
