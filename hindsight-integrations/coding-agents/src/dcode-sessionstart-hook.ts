#!/usr/bin/env node
/** Native DeepAgents Dcode Hooks V2 SessionStart entrypoint. */
import { runHarnessSessionStart } from "./harness/hook-lifecycle";

void runHarnessSessionStart("dcode");
