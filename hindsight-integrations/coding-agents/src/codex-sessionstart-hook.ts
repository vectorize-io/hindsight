#!/usr/bin/env node
/** hindsight-codex-sessionstart-hook — Codex CLI `SessionStart` hook: deterministically starts a
 *  background seed of a cold repo's bank and injects the knowledge-page bank-mission. Mirrors the
 *  Claude entry point; harness is "codex" so config/diag/bank resolution use the codex identity. */
import { runHarnessSessionStart } from "./harness/hook-lifecycle";

void runHarnessSessionStart("codex");
