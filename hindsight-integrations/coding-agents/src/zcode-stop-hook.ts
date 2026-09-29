#!/usr/bin/env node
/** ZCode Stop hook: closes the turn in the session journal with the assistant reply, then writes
 *  the whole journaled conversation back to memory (see core/turn-journal.ts). */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("zcode");
