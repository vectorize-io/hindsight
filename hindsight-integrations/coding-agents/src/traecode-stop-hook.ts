#!/usr/bin/env node
/** TraeCode Stop hook: closes the turn in the session journal with the assistant reply, then writes
 *  the whole journaled conversation back to memory (see core/turn-journal.ts). TraeCode keeps no
 *  readable transcript — its Stop payload carries the reply itself in `last_assistant_message`. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("traecode");
