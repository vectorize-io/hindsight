#!/usr/bin/env node
/** GitHub Copilot CLI sessionStart hook — shared Hindsight lifecycle, intentionally model-context
 * only: Copilot has no supported TUI banner response field (see hook-lifecycle.ts). */
import { runHarnessSessionStart } from "./harness/hook-lifecycle";

void runHarnessSessionStart("copilot-cli");
