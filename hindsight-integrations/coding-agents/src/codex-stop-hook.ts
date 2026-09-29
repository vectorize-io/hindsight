#!/usr/bin/env node
/** hindsight-codex-stop-hook — Codex CLI `Stop` hook: writes the session's rollout transcript back
 *  to memory. Same runtime as the Claude Stop hook, but with the Codex rollout reader. */
import { runHarnessRetain } from "./harness/hook-lifecycle";

void runHarnessRetain("codex");
