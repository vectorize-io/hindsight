#!/usr/bin/env node
/** Native DeepAgents Dcode Hooks V2 Stop entrypoint. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("dcode");
