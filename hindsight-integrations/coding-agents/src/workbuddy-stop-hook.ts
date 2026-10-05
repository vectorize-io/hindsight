#!/usr/bin/env node
/** WorkBuddy Stop hook: reads the session transcript (core/transcript-workbuddy.ts) and writes the
 *  whole conversation back to memory. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("workbuddy");
