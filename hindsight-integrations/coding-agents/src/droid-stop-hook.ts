#!/usr/bin/env node
/** Factory Droid Stop/idle notification hook: writes the session transcript back to memory. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("factory-droid");
