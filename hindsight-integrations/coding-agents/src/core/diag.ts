/**
 * Shared diagnostics helper for hook-based harnesses (Claude Code, Cursor CLI, ...): appends a
 * JSON line to the diagnostic file so a silently memory-less session can't masquerade as one that
 * worked. Never throws — diagnostics must not break the agent.
 *
 * Deliberately neutral (no dependency on recall/reflect or retain specifics) so every lifecycle
 * hook (UserPromptSubmit, Stop, and future ones like SessionStart) can depend on it without
 * reaching into another hook's module.
 */
import { appendFileSync } from "node:fs";
import { log } from "./log";

/** Where the JSONL trail lands. Exported so a failure notice can name the file to open. */
export function diagFilePath(): string {
  return process.env.HINDSIGHT_DIAG_FILE || "/tmp/hindsight-plugin.log";
}

export function diag(harness: string, event: string, extra: Record<string, unknown> = {}): void {
  log.debug(harness, `diag:${event}`, extra); // debug level shows the full story in ONE file
  try {
    appendFileSync(
      diagFilePath(),
      JSON.stringify({ ts: new Date().toISOString(), harness, event, ...extra }) + "\n"
    );
  } catch {
    /* diagnostics must not break the agent */
  }
}
