#!/usr/bin/env node
/** Antigravity CLI Stop hook: retain the completed transcript using the native camelCase payload. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("antigravity-cli");
