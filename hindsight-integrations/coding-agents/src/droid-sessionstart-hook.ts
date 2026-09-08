#!/usr/bin/env node
/** hindsight-droid-sessionstart-hook - on session start, deterministically starts a background
 *  seed of a cold repo's bank and injects the knowledge-page bank-mission (see core/session-start.ts). */
import { runHarnessSessionStart } from "./harness/hook-lifecycle";

void runHarnessSessionStart("factory-droid");
